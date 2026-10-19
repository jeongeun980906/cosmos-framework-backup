# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style action SFT dataset: ``DROIDLeRobotDataset`` → ``ActionTransformPipeline``.

The base ``DROIDLeRobotDataset.__getitem__`` returns the raw sample
(``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
``idle_frames``). The model expects each sample to be passed through
``ActionTransformPipeline`` (spatial resize/pad, text tokenization, action
padding to ``max_action_dim``, and ``sequence_plan`` construction). This thin
wrapper composes the two so the experiment can hand a single map-style dataset
to ``RankPartitionedDataLoader`` (mirroring how the vision recipe uses
``get_sft_dataset``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import ConcatDataset, Dataset, IterableDataset, get_worker_info

from cosmos_framework.utils import log

from cosmos_framework.data.generator.action.datasets.droid_merged_lerobot_dataset import DROIDMergedLeRobotDataset
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
from cosmos_framework.data.generator.action.datasets.libero_lerobot_delta_dataset import (
    LIBEROLeRobotDataset as LIBERODeltaLeRobotDataset,
)
from cosmos_framework.data.generator.action.datasets.pusht_lerobot_dataset import PushTLeRobotDataset
from cosmos_framework.data.generator.action.datasets.robomind_franka_dataset import (
    RoboMINDFrankaDataset,
)
from cosmos_framework.data.generator.action.datasets.robocasa_hdf5_dataset import (
    DEFAULT_CAMERAS as ROBOCASA_DEFAULT_CAMERAS,
)
from cosmos_framework.data.generator.action.datasets.robocasa_hdf5_dataset import RoboCasaHDF5Dataset
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

if TYPE_CHECKING:
    from cosmos_framework.data.generator.action.datasets.robomind_paired_dataset import (
        RoboMINDPairedDataset,
    )

# The four LIBERO eval suites trained jointly (one policy for all four).
_LIBERO_DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ``ActionTransformPipeline`` per sample.

    **Rank sharding** — ``RankPartitionedDataLoader`` assigns each rank to a dataset
    and sets ``shard_world_size`` / ``shard_rank`` on this wrapper *after*
    construction. Without using them, every rank would iterate the full dataset in
    the same order, so all data-parallel ranks process identical batches: the
    global batch is just ``max_samples_per_batch`` and the extra GPUs do redundant
    work (they only help fit the model via FSDP). We honor the shard here so each
    rank sees a disjoint ``1/shard_world_size`` slice → the effective global batch
    becomes ``max_samples_per_batch * shard_world_size`` (matching the DROID recipe's
    "global batch = batch × world_size"), and N GPUs give ~N× data throughput.

    The partition is **strided** (rank ``r`` takes global indices ``r, r+W, r+2W, …``)
    so each rank's samples are spread across all episodes rather than a contiguous
    block. All ranks get exactly ``len // W`` samples (the ``< W`` remainder is
    dropped) so batch counts stay balanced across ranks.

    (When wrapped in ``ActionIterableShuffleDataset``, the shard fields are set on
    the wrapper instead and these defaults stay 1/0 — no double sharding.)
    """

    def __init__(self, dataset: Dataset, transform: ActionTransformPipeline, resolution: str | int | None):
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution
        # Overwritten by RankPartitionedDataLoader; default = single shard (no-op).
        self.shard_world_size: int = 1
        self.shard_rank: int = 0

    def __len__(self) -> int:
        return len(self._dataset) // max(1, int(self.shard_world_size))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        world = max(1, int(self.shard_world_size))
        rank = int(self.shard_rank) % world
        global_idx = int(idx) * world + rank  # strided: this rank's disjoint slice
        return self._transform(self._dataset[global_idx], self._resolution)

    def get_shuffle_blocks(self):
        """Delegate to the inner DROIDLeRobotDataset (per-episode/segment flat-index blocks)."""
        return self._dataset.get_shuffle_blocks()


class ActionIterableShuffleDataset(IterableDataset):
    """Streaming view of a map-style ``ActionSFTDataset``.

    Each ``(rank, worker)`` is assigned a DISJOINT subset of episodes (sharded over
    ``shard_world_size * num_workers``), shuffles its episode ORDER, and streams the
    windows WITHIN each episode sequentially -> within-rank batch diversity (the N
    workers of a rank stream N different episodes) AND cross-rank diversity, while
    keeping reads sequential (I/O locality + COW; no RandomSampler random-access OOM).
    Re-shuffles each epoch and streams indefinitely (the trainer stops at ``max_iter``).

    ``shard_world_size`` / ``shard_rank`` are set by ``RankPartitionedDataLoader``.
    """

    def __init__(self, dataset: "ActionSFTDataset", seed: int = 42):
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:  # informational only; iteration is infinite
        return len(self._dataset)

    def __iter__(self):
        import torch

        blocks = self._dataset.get_shuffle_blocks()
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        global_shard = int(self.shard_rank) * nw + wid
        total_shards = max(1, int(self.shard_world_size) * nw)
        epoch = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch)  # same permutation across all (rank,worker) -> disjoint shard
            order = torch.randperm(len(blocks), generator=g).tolist()
            for b in order[global_shard::total_shards]:
                start, length = blocks[b]
                for idx in range(start, start + length):
                    yield self._dataset[idx]
            epoch += 1


def get_action_droid_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    action_space: str = "joint_pos",
    mode: str = "wam",
    use_state: bool = True,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    use_image_augmentation: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
    use_success_only: bool = True,
) -> Dataset:
    """Build the DROID action SFT dataset: ``action_space='joint_pos'`` (8D) +
    ``use_state`` (raw/un-normalized), concat_view, chunk_length 32.

    Reads ``root`` (a merged/versioned DROID LeRobot root) as a single flat
    dataset; ``use_success_only=True`` filters to the ``success/`` split."""
    shard_kwargs = dict(
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        mode=mode,
        use_state=use_state,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,  # i4: bundles random-crop+resize+ColorJitter
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
        use_success_only=use_success_only,
    )
    dataset: Dataset = DROIDLeRobotDataset(root=root, **shard_kwargs)
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_droid_merged_lerobot_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 16,
    action_space: str = "ee_pose",
    mode: str = "forward_dynamics",
    use_state: bool = False,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    split: str = "train",
    use_success_only: bool = False,
    use_image_augmentation: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    idle_frames_dropout: float = 0.05,
    format_prompt_as_json: bool = True,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build the DROID-Merged LeRobot SFT dataset for action FD recipes."""
    dataset = DROIDMergedLeRobotDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        mode=mode,
        use_state=use_state,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
        split=split,
        use_success_only=use_success_only,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        idle_frames_dropout=idle_frames_dropout,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_libero_sft_dataset(
    *,
    root: str,
    fps: float = 20.0,
    chunk_length: int = 16,
    image_size: int = 256,
    mode: str = "wam",
    camera_mode: str = "concat_view",
    action_space: str = "frame_wise_relative",
    rotation_space: str = "6d",
    pose_coordinate_frame: str = "native",
    action_normalization: str | None = "quantile_rot",
    action_stats_path: str | None = None,
    split: str = "train",
    val_ratio: float = 0.01,
    seed: int = 0,
    resolution: str | int | None = None,
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build the LIBERO action-policy SFT dataset (GA reproduction defaults).

    Feeds ``LIBEROLeRobotDataset`` (frame-wise-relative rot6d actions,
    ``quantile_rot``-normalized, concat_view third-person + wrist at 256x256 each
    → 256x512) through ``ActionTransformPipeline``. ``root`` is a LOCAL LeRobot dir
    (read parquet + video directly); pre-sync the HF dataset once, e.g.
    ``hf download lerobot/libero_10 --repo-type dataset --local-dir <root>``. Point
    ``root`` at libero_10 alone. The
    dataset is FPS-agnostic (decodes at real frame timestamps); ``fps`` is metadata
    for ``conditioning_fps`` / prompt duration.
    """
    dataset = LIBEROLeRobotDataset(
        root=root,
        image_size=image_size,
        chunk_length=chunk_length,
        fps=fps,
        mode=mode,
        split=split,
        val_ratio=val_ratio,
        seed=seed,
        camera_mode=camera_mode,
        action_space=action_space,
        rotation_space=rotation_space,
        pose_coordinate_frame=pose_coordinate_frame,
        action_normalization=action_normalization,
        action_stats_path=action_stats_path,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_pusht_sft_dataset(
    *,
    root: str,
    fps: float = 10.0,
    chunk_length: int = 16,
    mode: str = "policy",
    use_state: bool = True,
    action_normalization: str | None = "minmax",
    viewpoint: str = "third_person_view",
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
) -> ActionSFTDataset:
    """Build the PushT 2D-action SFT dataset (single view, ``[pos_x, pos_y]``,
    min-max normalized). ``mode='policy'`` trains the policy task; ``'joint'``
    samples FD/ID/policy per item for multi-task training."""
    dataset = PushTLeRobotDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        use_state=use_state,
        action_normalization=action_normalization,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    return ActionSFTDataset(dataset, transform, resolution)


def get_action_libero_delta_sft_dataset(
    *,
    root: str,
    suites: list[str] | tuple[str, ...] | None = _LIBERO_DEFAULT_SUITES,
    fps: float = 20.0,
    chunk_length: int = 16,
    mode: str = "policy",
    action_normalization: str | None = "minmax",
    normalization_path: str | None = None,
    viewpoint: str = "third_person_view",
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    use_state: bool = False,
    use_wrist: bool = False,
    wrist_key: str = "observation.images.wrist_image",
) -> ActionSFTDataset:
    """Build the local LIBERO 7D-delta-action SFT dataset (single agent-view,
    ``[dpos(3), daxang(3), gripper]``, min-max normalized via the shared stats
    JSON). Distinct from the upstream ``get_action_libero_sft_dataset`` (GA
    reproduction: concat_view + rot6d + quantile_rot).

    When ``suites`` is given, ``root`` is the ``LIBERO_LeRobot_v3`` parent dir and
    the suites are concatenated into a SINGLE map-style dataset (frame-count
    proportional mixing, and — unlike one dataloader entry per suite — it works
    for any ``world_size``). When ``suites=None``, ``root`` is one suite dir.
    ``mode='policy'`` trains the policy task; ``'joint'`` samples FD/ID/policy
    per item for multi-task training."""
    if suites:
        roots = [str(Path(root) / s) for s in suites]
    else:
        roots = [root]
    per_suite = [
        LIBERODeltaLeRobotDataset(
            root=r,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            normalization_path=normalization_path,
            use_state=use_state,
            use_wrist=use_wrist,
            wrist_key=wrist_key,
        )
        for r in roots
    ]
    dataset: Dataset = per_suite[0] if len(per_suite) == 1 else ConcatDataset(per_suite)
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    return ActionSFTDataset(dataset, transform, resolution)


def _build_robomind_per_task(
    *,
    root: str,
    embodiment_type: str,
    tasks: list[str] | tuple[str, ...] | None,
    fps: float,
    chunk_length: int,
    mode: str,
    viewpoint: str,
    action_normalization: str | None,
    sample_stride: int,
    droplist_path: str | None,
) -> list[RoboMINDFrankaDataset]:
    """One ``RoboMINDFrankaDataset`` per LeRobot task repo under ``root``.

    The BAAI-DataCube republication splits RoboMIND into one LeRobot v3 repo per task, so
    ``root`` is the parent holding those task directories and ``tasks=None`` picks up every
    one that has a readable ``meta/info.json``. Shared by the BC factory (which concatenates
    the list into one flat map-style dataset) and the ICL factory (which also needs the task
    boundaries, because "same task" is what scopes demo sampling).
    """
    base = Path(root)

    def _complete(d: Path) -> bool:
        # info.json alone is not enough: an interrupted `hf download` leaves the small
        # files behind and the parquet/episode index missing, and the failure then
        # surfaces as a FileNotFoundError deep inside the first __getitem__.
        return (
            (d / "meta" / "info.json").is_file()
            and (d / "meta" / "stats.json").is_file()
            and (d / "meta" / "tasks.parquet").is_file()
            and (d / "meta" / "episodes").is_dir()
            and (d / "videos").is_dir()
            and any((d / "data").glob("chunk-*/*.parquet"))
        )

    if tasks:
        names = [t for t in tasks]
        incomplete = [t for t in names if not _complete(base / t)]
        if incomplete:
            raise FileNotFoundError(f"Incomplete RoboMIND task dirs under {root!r}: {incomplete}")
    else:
        names = sorted(p.name for p in base.iterdir() if p.is_dir() and _complete(p))
        skipped = sorted(p.name for p in base.iterdir() if p.is_dir() and not _complete(p))
        if skipped:
            log.warning(f"RoboMIND: skipping {len(skipped)} incomplete task dir(s) under {root}: {skipped}")
    if not names:
        raise FileNotFoundError(f"No complete LeRobot task dirs under {root!r}")
    # {task_name: [episode_index, ...]} of windows whose video will not decode. Built by
    # scanning every episode offline; see tools/ and the dataset docs. Filtering here
    # keeps the task and drops only the broken episodes, instead of banning a whole repo
    # over a handful of them.
    drops: dict[str, list[int]] = {}
    if droplist_path:
        import json as _json

        with open(droplist_path) as fh:
            drops = _json.load(fh)
        n_ep = sum(len(v) for v in drops.values())
        log.info(f"RoboMIND droplist {droplist_path}: {n_ep} episode(s) across {len(drops)} task(s)")
    per_task = [
        RoboMINDFrankaDataset(
            root=str(base / n),
            embodiment_type=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
            drop_episodes=drops.get(n),
        )
        for n in names
    ]
    return per_task


def get_action_robomind_sft_dataset(
    *,
    root: str,
    embodiment_type: str = "robomind-franka",
    tasks: list[str] | tuple[str, ...] | None = None,
    fps: float = 30.0,
    chunk_length: int = 16,
    mode: str = "wam",
    viewpoint: str = "concat_view",
    action_normalization: str | None = "quantile",
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    sample_stride: int = 1,
    droplist_path: str | None = None,
) -> ActionSFTDataset:
    """RoboMIND as a plain BC stream — one LeRobot root per task, concatenated.

    The BAAI-DataCube republication splits RoboMIND into one LeRobot v3 repo per task, so
    ``root`` is the parent holding those task directories and ``tasks=None`` picks up every
    one that has a readable ``meta/info.json``. Concatenating them into a SINGLE map-style
    dataset (rather than one dataloader entry each) mixes tasks in proportion to their
    frame counts and works at any ``world_size`` — the same reasoning as the LIBERO suites.

    ``mode="wam"`` on purpose: this stream exists to keep the RoboCasa post-train contract
    alive during mid-training — first frame in, video and actions jointly denoised out —
    rather than to add forward/inverse-dynamics tasks. ``mode="joint"`` would sample all
    three per item.

    ``embodiment_type`` selects the head row: ``robomind-franka`` is domain 8 at 10-D and
    ``robomind-agilex-dual`` is domain 12 at 20-D. Both rows are already trained in the
    public Edge base, which is what makes this replay rather than a new row to fill.
    """
    per_task = _build_robomind_per_task(
        root=root,
        embodiment_type=embodiment_type,
        tasks=tasks,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        action_normalization=action_normalization,
        sample_stride=sample_stride,
        droplist_path=droplist_path,
    )
    dataset: Dataset = per_task[0] if len(per_task) == 1 else ConcatDataset(per_task)
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    return ActionSFTDataset(dataset, transform, resolution)


def get_action_robomind_icl_sft_dataset(
    *,
    root: str,
    embodiment_type: str = "robomind-franka",
    tasks: list[str] | tuple[str, ...] | None = None,
    fps: float = 30.0,
    chunk_length: int = 16,
    mode: str = "wam",
    viewpoint: str = "concat_view",
    action_normalization: str | None = "quantile",
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    sample_stride: int = 1,
    droplist_path: str | None = None,
    share_temporal: bool = True,
    demo_same_episode: bool = False,
    demo_match: str = "state",
    gripper_weight: float = 2.0,
    motion_weight: float = 2.0,
    progress_weight: float = 0.0,
    demo_top_k: int = 5,
    demo_candidate_episodes: int = 8,
    seed: int = 0,
) -> "RoboMINDPairedDataset":
    """RoboMIND as an ICL ``[demo, query]`` stream — the paired counterpart of
    :func:`get_action_robomind_sft_dataset`.

    Same windows, same task discovery, same normalization, same flat index order; the one
    difference is that each sample also carries a demo — a window from the SAME task repo
    and a DIFFERENT episode — as a fully clean second vision item and action segment. That
    is the layout the EgoVerse pool that owns the rest of the mid-train trains in, so the
    replay rows are refreshed under the same contract instead of a BC one.

    ``demo_match="state"`` (default) also picks WHEN in that episode: the window whose t0 EE
    pose and gripper, and net displacement across the window, are closest to the query's, so
    the demo shows the same phase of the task and moves the same way rather than landing at
    a uniformly random point. ``"random"`` restores the unmatched draw for the ablation arm.

    The caption is tokenized by the paired dataset, not by the pipeline: the demo clause is
    only known once the pair is assembled, so ``tokenizer_config`` is deliberately withheld
    from ``ActionTransformPipeline`` here and passed downstream instead.
    """
    from cosmos_framework.data.generator.action.datasets.robomind_paired_dataset import (
        RoboMINDPairedDataset,
    )

    per_task = _build_robomind_per_task(
        root=root,
        embodiment_type=embodiment_type,
        tasks=tasks,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        action_normalization=action_normalization,
        sample_stride=sample_stride,
        droplist_path=droplist_path,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=None,  # tokenization moves to RoboMINDPairedDataset (see above)
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    return RoboMINDPairedDataset(
        per_task=per_task,
        transform=transform,
        resolution=resolution,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        share_temporal=share_temporal,
        demo_same_episode=demo_same_episode,
        demo_match=demo_match,
        gripper_weight=gripper_weight,
        motion_weight=motion_weight,
        progress_weight=progress_weight,
        demo_top_k=demo_top_k,
        demo_candidate_episodes=demo_candidate_episodes,
        seed=seed,
    )


def get_action_robocasa_sft_dataset(
    *,
    root: str,
    fps: float = 20.0,
    chunk_length: int = 32,
    mode: str = "wam",
    cameras: tuple[str, ...] | list[str] = ROBOCASA_DEFAULT_CAMERAS,
    action_normalization: str | None = "minmax",
    normalization_path: str | None = None,
    use_state: bool = True,
    task_split: list[str] | tuple[str, ...] | None = None,
    pad_tail: bool = True,
    sample_stride: int = 1,
    index_cache_path: str | None = None,
    domain_name: str = "robocasa",
    action_channel_offset: int = 0,
    viewpoint: str = "third_person_view",
    resolution: str | int | None = None,
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = True,
    episode_shuffle_seed: int = 42,
    dummy_ret_frames: int | None = None,
) -> Dataset:
    """Build the RoboCasa 24-task action-policy SFT dataset.

    Feeds :class:`RoboCasaHDF5Dataset` (7D delta-EE ``[dpos(3), daxang(3),
    gripper]``, min-max normalized from the committed stats JSON, three cameras
    vertically concatenated, absolute proprio as the ``use_state`` conditioning
    row) through ``ActionTransformPipeline``. ``root`` is the release's
    ``success_only`` dir, read in place — no LeRobot conversion.

    ``iterable_shuffle=True`` (default) streams rank x worker-sharded, episode-
    order-shuffled and sequential *within* an episode, which keeps the HDF5 reads
    sequential — worth much more here than for the LeRobot readers, since a sample
    is ~15 MB of raw frames.

    ``domain_name`` only selects which ``DomainAwareLinear`` row the action heads
    index (default ``"robocasa"`` -> id 5). See :class:`RoboCasaHDF5Dataset`.
    """
    dataset = RoboCasaHDF5Dataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        cameras=cameras,
        viewpoint=viewpoint,
        action_normalization=action_normalization,
        normalization_path=normalization_path,
        use_state=use_state,
        task_split=task_split,
        pad_tail=pad_tail,
        sample_stride=sample_stride,
        index_cache_path=index_cache_path,
        domain_name=domain_name,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        action_channel_offset=action_channel_offset,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    # Present each single-clip sample in the paired [empty demo, query] layout the mid
    # checkpoint was trained on; see dummy_ret_wrap for why. None leaves it untouched.
    if dummy_ret_frames is not None:
        from cosmos_framework.data.generator.action.datasets.dummy_ret_wrap import DummyRetPairedWrap

        sft = DummyRetPairedWrap(sft, dummy_frames=int(dummy_ret_frames))
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def get_action_aloha_tabletop_sft_dataset(
    root: str,
    fps: float = 20.0,
    chunk_length: int = 32,
    mode: str = "wam",
    viewpoint: str = "third_person_view",
    action_normalization: str | None = "minmax",
    normalization_path: str | None = None,
    task_split: list[str] | tuple[str, ...] | None = None,
    pad_tail: bool = True,
    sample_stride: int = 1,
    domain_name: str = "aloha-tabletop",
    resolution: str | int | None = "480",
    max_action_dim: int = 64,
    action_channel_offset: int = 0,
    cfg_dropout_rate: float = 0.1,
    tokenizer_config: dict | None = None,
    format_prompt_as_json: bool = False,
    append_viewpoint_info: bool = False,
    append_duration_fps_timestamps: bool = False,
    append_resolution_info: bool = False,
    append_idle_frames: bool = False,
    iterable_shuffle: bool = True,
    episode_shuffle_seed: int = 42,
    index_cache_path: str | None = None,
    use_state: bool = True,
) -> Dataset:
    """Tabletop-Sim ALOHA 20D bimanual SFT dataset, same pipeline as the RoboCasa recipe.

    Exists so the 400-step warm-start probe has a second target with a different action
    width, routed through a different untrained head row (16 rather than 5). See
    :mod:`aloha_tabletop_dataset`.
    """
    from cosmos_framework.data.generator.action.datasets.aloha_tabletop_dataset import AlohaTabletopDataset

    dataset = AlohaTabletopDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        action_normalization=action_normalization,
        normalization_path=normalization_path,
        task_split=task_split,
        pad_tail=pad_tail,
        sample_stride=sample_stride,
        domain_name=domain_name,
        use_state=use_state,
        index_cache_path=index_cache_path,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        action_channel_offset=action_channel_offset,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft

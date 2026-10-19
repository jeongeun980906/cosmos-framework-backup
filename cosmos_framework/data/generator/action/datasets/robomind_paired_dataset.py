# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboMIND as an ICL ``[demo, query]`` stream instead of a plain BC stream.

Why this exists. The mid-train's RoboMIND rows (``..._bcmix``) are BC: one clip in,
video+actions jointly denoised out, no demo in context. Everything else in that
mid-train -- the EgoVerse pool that owns 87.5% of the ranks -- is ICL: two vision
items, two action segments, demo fully clean, query generated. So the replay stream
that is supposed to keep domains 8/12 alive presents those rows in a layout the rest
of the run never uses, and the gripper rows are refreshed under a *different* contract
than the one the backbone is being shaped for.

This module presents the same RoboMIND windows in the ICL layout. Each query is paired
with a demo clip drawn from the SAME TASK and a DIFFERENT EPISODE, and the pair is
assembled exactly as :class:`EgoVersePairedTransform` assembles a retrieval pair --
``video=[demo, query]``, ``action=[demo, query]``, per-item ``domain_id`` /
``raw_action_dim`` / ``image_size`` / ``conditioning_fps``, and the query's own sequence
plan with the shared temporal grid on. The packer marks every vision item but the last
clean, so the demo (video AND action) is conditioning and only the query is supervised.

Retrieval runs through :class:`TaskPairDescriptor`, one per task. The BAAI-DataCube
republication is one LeRobot repo per task, so "same task" is "same sub-dataset", and
inside a repo the rows are index-sorted so every episode is one contiguous run of
windows. The descriptor holds that layout AND a per-window key, and a demo is picked in
two steps:

  1. WHICH episode -- several drawn uniformly among the OTHER episodes of the task, the
     best-matching one kept. A different episode is the point: a window from the query's
     own episode overlaps the query in time and is very nearly its answer, so the model
     would learn to copy the demo instead of reading it as a demonstration.
  2. WHEN inside it -- the nearest windows by t0 state (absolute EE position + gripper
     closedness) AND net displacement across the window, sampled among the top k.

Step 2 exists because step 1 alone is not enough. Drawing the window uniformly makes the
demo land at a uniformly random PHASE of the task, and measured on ``insert_marker`` that
put the median demo 0.34 of an episode away from the query's own phase, with 31% of pairs
more than half an episode apart -- a 0.57 s demo of the release while the query is still
approaching. Same task, but not a demonstration of what the query is about to do.

Matching on raw absolute coordinates is correct here and needs no canonicalization: demo
and query are the same robot in the same task repo, so a pose means the same thing on both
sides. The gripper is weighted up because it is the sharpest phase landmark available --
approach, grasp, transport and release are one bit apart, while the arm can sit near the
same point in space on either side of a grasp.

What state matching does NOT fix, and is left alone deliberately: it aligns the ROBOT, not
the scene. A repetitive task brings the arm back through the same poses once per object, so
on ``packplate`` the retrieved demo can be racking its third plate while the query is on its
first -- same motion, different amount of the task already done. The descriptor therefore
also carries a ``grasps`` channel (closures completed in the episode), but it is weighted 0
by default: at weight 2 it moved only 6 of 45 sampled pairs, and on those it bought cycle
agreement (64% -> 71%) while costing t0 scene similarity (L1 0.176 -> 0.188). Six pairs is
not a basis for a default, and cycle mismatch is a milder defect than the phase mismatch
that motivated matching in the first place -- the motion being demonstrated is still the
right motion. ``progress_weight > 0`` turns it on for the ablation.

Query ORDER is byte-identical to the BC stream's: the same flat index space, the same
strided per-rank shard, no shuffle. Only the demo slot is added, so ICL-vs-BC over these
rows differs in exactly one thing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.utils import log


@dataclass(frozen=True)
class TaskPairDescriptor:
    """Retrieval index for ONE task: where each episode's windows are, and what each
    window looks like to match against.

    Attributes:
        episode_ids: Episode index of each entry, ascending.
        starts / ends: Half-open window range ``[start, end)`` of that episode, in the
            task dataset's own index space. Only windows lying entirely inside the
            episode are included (see ``episode_window_spans``).
        keys: ``[n_windows, F]`` whitened, weighted retrieval key per window, or ``None``
            when matching is off. Built from absolute EE position + gripper closedness at
            the window's t0 (``window_retrieval_keys``), z-scored per channel over the task
            so metres and a [0, 1] gripper are comparable, then the gripper channels scaled
            by ``gripper_weight``. Retrieval is then a plain L2 nearest-neighbour.
    """

    episode_ids: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    keys: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        dataset: Any,
        *,
        match: bool = True,
        gripper_weight: float = 2.0,
        motion_weight: float = 2.0,
        progress_weight: float = 0.0,
    ) -> "TaskPairDescriptor":
        ids, starts, ends = dataset.episode_window_spans()
        keys = None
        if match:
            raw = dataset.window_retrieval_keys()
            std = raw.std(axis=0)
            std[std < 1e-6] = 1.0  # a channel that never moves carries no information
            keys = (raw - raw.mean(axis=0)) / std
            # Weighted by what each channel MEANS, read off the dataset's own column names
            # rather than a stride. The gripper is the sharpest phase landmark a manipulation
            # episode has -- approach, grasp, transport and release are one bit apart --
            # though measured on insert_marker the result is flat between weight 1 and 4.
            # The displacement block is what actually earns its weight: it decides direction
            # of travel, and without it the retrieved demo moved the opposite way 29% of the
            # time against 12.5% with it. The grasps block is weighted 0 by default; it is
            # computed regardless because it costs one pass over a column already in memory,
            # and leaving it in the key makes the ablation a config change.
            weights = np.ones(keys.shape[1], dtype=np.float32)
            for i, name in enumerate(dataset.retrieval_key_names()):
                if name.endswith("gripper"):
                    weights[i] = float(gripper_weight)
                elif "disp_" in name:
                    weights[i] = float(motion_weight)
                elif name.endswith("grasps"):
                    weights[i] = float(progress_weight)
            keys = np.ascontiguousarray(keys * weights, dtype=np.float32)
        return cls(episode_ids=ids, starts=starts, ends=ends, keys=keys)

    def __len__(self) -> int:
        return int(self.episode_ids.shape[0])

    def slot_of(self, episode_id: int) -> int:
        """Descriptor slot holding ``episode_id``, or ``-1`` when it holds none.

        An episode shorter than one window contributes no slot, so a query can legitimately
        belong to an episode the descriptor does not list.
        """
        i = int(np.searchsorted(self.episode_ids, episode_id))
        if i < len(self) and int(self.episode_ids[i]) == int(episode_id):
            return i
        return -1

    def sample_window(
        self,
        slot: int,
        rng: random.Random,
        query_key: np.ndarray | None = None,
        top_k: int = 5,
    ) -> int:
        """A window from the episode in ``slot``.

        With a ``query_key`` (and keys built), one of the ``top_k`` windows whose t0 state
        is closest to the query's — the episode is already chosen, so this only decides
        WHEN inside it, which is exactly the phase choice. Drawing among the top k rather
        than taking the argmin keeps some variety without letting the phase drift.

        Without a key, uniform over the episode: the unmatched behaviour, kept for the
        ablation arm.
        """
        lo, hi = int(self.starts[slot]), int(self.ends[slot])
        if query_key is None or self.keys is None:
            return rng.randrange(lo, hi)
        dist = np.linalg.norm(self.keys[lo:hi] - query_key, axis=1)
        k = min(int(top_k), int(dist.shape[0]))
        nearest = np.argpartition(dist, k - 1)[:k] if k < dist.shape[0] else np.arange(dist.shape[0])
        return lo + int(nearest[rng.randrange(k)])

    def retrieve(
        self,
        exclude_slot: int,
        rng: random.Random,
        query_key: np.ndarray | None = None,
        *,
        top_k: int = 5,
        candidate_episodes: int = 8,
    ) -> int:
        """Retrieve a demo window: the best match among ``candidate_episodes`` draws.

        Each candidate is a different episode (never ``exclude_slot``) with its own best
        window; the closest of them wins. Searching more than one episode is what actually
        improves the match -- one draw can land on an episode that simply never passes
        through the query's configuration, and taking the best window of a bad episode is
        still a bad demo. Measured on insert_marker, going from 1 to 8 candidates takes the
        share of demos moving in the OPPOSITE direction from 10.4% to 4.4% and the median
        match distance from 2.38 to 1.31.

        It does not concentrate demos on a few "easy" episodes: over the same sweep the
        episode-usage entropy stayed at 0.95-0.97 of uniform and 259 of 292 episodes were
        still used, because the candidates themselves are drawn uniformly.

        Adding channels to the key, by contrast, did nothing -- wrist rotation and a
        mid-window path point were both flat-to-worse on every metric, on both tasks.
        """
        if self.keys is None or query_key is None:
            return self.sample_window(self.sample_other_episode(exclude_slot, rng), rng)
        best_window, best_dist = -1, float("inf")
        for _ in range(max(1, int(candidate_episodes))):
            slot = self.sample_other_episode(exclude_slot, rng)
            window = self.sample_window(slot, rng, query_key, top_k=top_k)
            dist = float(np.linalg.norm(self.keys[window] - query_key))
            if dist < best_dist:
                best_window, best_dist = window, dist
        return best_window

    def sample_other_episode(self, exclude_slot: int, rng: random.Random) -> int:
        """A slot drawn uniformly from every episode except ``exclude_slot``.

        Drawn over ``n - 1`` slots and shifted past the excluded one, so the exclusion is
        exact and uniform in one draw. Falls back to ``exclude_slot`` only for a
        single-episode task, where no other episode exists.
        """
        n = len(self)
        if n <= 1:
            return 0 if exclude_slot < 0 else exclude_slot
        if exclude_slot < 0:
            return rng.randrange(n)
        j = rng.randrange(n - 1)
        return j if j < exclude_slot else j + 1


def _to_float_video(video: torch.Tensor) -> torch.Tensor:
    """uint8 ``[C,T,H,W]`` -> float in ``[-1, 1]``.

    Both items of a multi-vision list must share a dtype: the model treats an all-float
    list as already preprocessed and an all-uint8 list as raw, and a mixed list takes the
    raw path and then asserts on the float item.
    """
    if video.dtype == torch.uint8:
        return video.float().div_(127.5).sub_(1.0)
    return video


class RoboMINDPairedDataset(Dataset):
    """Per-task RoboMIND datasets served as ``[demo, query]`` ICL samples.

    Args:
        per_task: One ``RoboMINDFrankaDataset`` per LeRobot task repo. Concatenated into
            one flat index space (the same one the BC stream's ``ConcatDataset`` builds),
            and also the unit of "same task" for demo retrieval.
        transform: The single-clip ``ActionTransformPipeline``, applied to query and demo
            independently. Must be built WITHOUT a tokenizer: the caption is only final
            once the demo clause is appended, so tokenization happens here.
        resolution: Resolution tier handed to ``transform``.
        tokenizer_config: VLM tokenizer config. ``None`` leaves ``ai_caption`` untokenized
            (smoke tests).
        cfg_dropout_rate: Probability of emptying the caption for classifier-free guidance.
        share_temporal: Put demo and query on one temporal mRoPE grid. Sound here because
            both clips come from the same repo -- same cameras, same frame size, same chunk
            length -- so their latent grids are identical by construction.
        demo_same_episode: Draw the demo from the query's OWN episode instead. Off by
            default; it exists to measure the copying shortcut the different-episode rule
            removes, not as a training setting.
        demo_match: ``"state"`` picks the window in the chosen episode whose t0 EE pose and
            gripper are closest to the query's, so the demo shows the same PHASE of the
            task. ``"random"`` draws the window uniformly (the unmatched ablation).
        gripper_weight: Weight on the gripper channels of the whitened key. Measured flat
            between 1 and 4 on insert_marker, so this is not a sensitive knob.
        motion_weight: Weight on the net-displacement channels. This one matters: it is what
            keeps the demo moving in the same direction as the query.
        progress_weight: Weight on the grasps-so-far channels. OFF by default — see the
            module docstring for the measurement that did not justify turning it on.
        demo_top_k: Sample among this many nearest windows rather than taking the argmin,
            so the phase stays put but the demo is not a single fixed clip.
        demo_candidate_episodes: Draw this many different episodes and keep the best match.
            The single most effective knob here (see :meth:`TaskPairDescriptor.retrieve`),
            and nearly free: a few hundred distance computations against two video decodes.
        seed: Mixed with the query index into a per-item RNG, so a given query always gets
            the same demo, in every worker and every re-run.
    """

    _MATCH_MODES = ("state", "random")

    def __init__(
        self,
        *,
        per_task: list[Dataset],
        transform: ActionTransformPipeline,
        resolution: str | int | None,
        tokenizer_config: dict | None = None,
        cfg_dropout_rate: float = 0.1,
        share_temporal: bool = True,
        demo_same_episode: bool = False,
        demo_match: str = "state",
        gripper_weight: float = 2.0,
        motion_weight: float = 2.0,
        progress_weight: float = 0.0,
        demo_top_k: int = 5,
        demo_candidate_episodes: int = 8,
        text_token_key: str = "text_token_ids",
        seed: int = 0,
    ) -> None:
        super().__init__()
        assert per_task, "RoboMINDPairedDataset needs at least one task dataset"
        self._per_task = list(per_task)
        self._transform = transform
        self._resolution = resolution
        self._share_temporal = bool(share_temporal)
        self._demo_same_episode = bool(demo_same_episode)
        if demo_match not in self._MATCH_MODES:
            raise ValueError(f"demo_match must be one of {self._MATCH_MODES}, got {demo_match!r}")
        self._demo_match = demo_match
        self._demo_top_k = max(1, int(demo_top_k))
        self._demo_candidate_episodes = max(1, int(demo_candidate_episodes))
        self._seed = int(seed)
        # Prefix sums over the per-task lengths: the flat index space the BC stream's
        # ConcatDataset builds, rebuilt here because the task boundaries are what scope
        # demo retrieval and ConcatDataset does not expose them as a lookup.
        self._cumulative: list[int] = []
        total = 0
        for ds in self._per_task:
            total += len(ds)
            self._cumulative.append(total)
        self._total = total
        # Built EAGERLY, before the DataLoader forks its workers: each descriptor scans its
        # task's row index once, and building lazily would repeat that scan in every worker
        # instead of inheriting it copy-on-write.
        self._descriptors = [
            TaskPairDescriptor.build(
                ds,
                match=self._demo_match == "state",
                gripper_weight=gripper_weight,
                motion_weight=motion_weight,
                progress_weight=progress_weight,
            )
            for ds in self._per_task
        ]
        n_eps = sum(len(d) for d in self._descriptors)
        solo = sum(1 for d in self._descriptors if len(d) <= 1)
        log.info(
            f"RoboMINDPairedDataset: {len(self._per_task)} task(s), {n_eps} episode(s), "
            f"{self._total} query window(s)"
            + f", demo_match={self._demo_match}"
            + (f"; {solo} task(s) have a single usable episode (self-task demos)" if solo else "")
        )
        self._text_tokenizer = None
        if tokenizer_config is not None:
            from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransform

            self._text_tokenizer = TextTokenizerTransform(
                input_keys=["ai_caption"],
                output_keys=[text_token_key],
                args={"tokenizer_config": tokenizer_config, "cfg_dropout_rate": cfg_dropout_rate},
            )
        # Set by RankPartitionedDataLoader after construction; default = single shard.
        self.shard_world_size: int = 1
        self.shard_rank: int = 0

    def __len__(self) -> int:
        return self._total // max(1, int(self.shard_world_size))

    def _locate(self, gi: int) -> tuple[int, int]:
        """Flat index -> ``(task, index within that task)``."""
        import bisect

        t = bisect.bisect_right(self._cumulative, gi)
        base = self._cumulative[t - 1] if t else 0
        return t, gi - base

    def _retrieve_demo(self, task: int, local: int, rng: random.Random) -> int:
        """Window index of the demo for query ``local`` of ``task``.

        Episode first (a different one), then phase inside it (nearest t0 state).
        """
        desc = self._descriptors[task]
        if len(desc) == 0:  # no episode long enough for a whole window: self-condition
            return local
        slot = desc.slot_of(self._per_task[task].episode_of(local))
        key = None if desc.keys is None else desc.keys[local]
        if self._demo_same_episode:
            slot = slot if slot >= 0 else rng.randrange(len(desc))
            return desc.sample_window(slot, rng, query_key=key, top_k=self._demo_top_k)
        return desc.retrieve(
            slot,
            rng,
            key,
            top_k=self._demo_top_k,
            candidate_episodes=self._demo_candidate_episodes,
        )

    def _pair_indices(self, i: int) -> tuple[int, int, int]:
        """``(task, query window, demo window)`` for sample ``i``.

        The single source of truth for the pairing, so ``pair_provenance`` reports exactly
        what ``__getitem__`` reads instead of re-deriving it.
        """
        world = max(1, int(self.shard_world_size))
        gi = int(i) * world + (int(self.shard_rank) % world)  # strided per-rank slice
        task, local = self._locate(gi)
        rng = random.Random((self._seed * 1000003) ^ (gi * 2654435761))
        return task, local, self._retrieve_demo(task, local, rng)

    def pair_provenance(self, i: int) -> dict[str, Any]:
        """Where sample ``i``'s demo and query come from, decoding nothing.

        For inspection (``tools/robomind_icl_dump_pairs.py``) and for asserting the
        same-task/different-episode property without reading video.

        One caveat: the task dataset steps to a random window when a window will not
        decode, so for the rare undecodable window the sample actually built differs from
        what this reports.
        """
        task, q, c = self._pair_indices(i)
        ds, desc = self._per_task[task], self._descriptors[task]
        return {
            "task": ds.repo_name,
            "task_index": task,
            "query_window": q,
            "demo_window": c,
            "query_episode": ds.episode_of(q),
            "demo_episode": ds.episode_of(c),
            "embodiment": ds.domain_name,
            # How close the demo's t0 state is to the query's, in whitened key units.
            # None when matching is off. Small = same phase of the task.
            "match_distance": (
                None
                if desc.keys is None
                else float(np.linalg.norm(desc.keys[c] - desc.keys[q]))
            ),
        }

    def __getitem__(self, i: int) -> dict[str, Any]:
        task, local, demo_local = self._pair_indices(i)
        query = self._transform(self._per_task[task][local], self._resolution)
        demo = self._transform(self._per_task[task][demo_local], self._resolution)
        return self._assemble(query, demo, self._per_task[task].domain_name)

    def _assemble(self, q: dict[str, Any], c: dict[str, Any], emb: str) -> dict[str, Any]:
        q_v, c_v = _to_float_video(q["video"]), _to_float_video(c["video"])
        if self._share_temporal and c_v.shape != q_v.shape:
            # Same repo => same cameras and frame size, so this cannot happen for RoboMIND.
            # Checked anyway because a mismatch otherwise surfaces as an assert deep inside
            # the packer's shared-grid path, where the cause is no longer visible.
            raise ValueError(
                f"share_temporal needs equal demo/query shapes, got {tuple(c_v.shape)} vs {tuple(q_v.shape)}"
            )

        sample = dict(q)
        sample.update(
            # Order is [demo, query] -- the packer marks every vision item but the last
            # fully clean, so the demo conditions and only the query is supervised.
            video=[c_v, q_v],
            image_size=[c["image_size"], q["image_size"]],
            action=[c["action"], q["action"]],
            domain_id=[c["domain_id"], q["domain_id"]],
            raw_action_dim=[c["raw_action_dim"], q["raw_action_dim"]],
            # Per-item fps [demo, query] (NOT per-sample): get_data_and_condition flattens
            # this to one fps per vision item / action segment.
            conditioning_fps=torch.stack([c["conditioning_fps"], q["conditioning_fps"]]),
            num_frames=int(q_v.shape[1]),
            dataset_name="robomind_paired",
            # The query's own plan, so the action<->vision alignment the pipeline derived
            # for this window is preserved verbatim; only the shared grid is switched on.
            sequence_plan=replace(q["sequence_plan"], share_vision_temporal_positions=self._share_temporal),
        )
        # ``action_raw`` / ``action_processing_record`` stay the QUERY's: the demo segment
        # is a clean conditioning input with no target, so there is nothing to evaluate it
        # against (same as DummyRetPairedWrap).
        #
        # Caption grammar matches the EgoVerse ICL stream -- demo clause then target
        # clause, both at the tail -- so the two streams describe the layout the same way.
        sample["ai_caption"] = f"{q['ai_caption']} demo {emb} target {emb}"
        if self._text_tokenizer is not None:
            sample = self._text_tokenizer(sample)
        return sample

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal LIBERO LeRobot dataset for Cosmos Action 7D-policy post-training.

LIBERO is a language-conditioned tabletop manipulation benchmark (Franka Panda).
The ``nvidia/LIBERO_LeRobot_v3`` release ships four task suites
(``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``), each a
self-contained LeRobot v3.0 dataset. The framework already registers LIBERO as a
first-class embodiment (``domain_id=5`` in
:mod:`cosmos_framework.data.generator.action.domain_utils`); this dataset is the
LeRobot-backed reader that feeds it.

It mirrors :class:`~cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset.DROIDLeRobotDataset`
and :class:`~cosmos_framework.data.generator.action.datasets.pusht_lerobot_dataset.PushTLeRobotDataset`
(same compact lazy frame index, same per-sample window builder, same
``_build_result`` contract consumed by ``ActionTransformPipeline``) but is
tailored for LIBERO:

  * **Single view** (the agent-view ``observation.images.image``). The dataset
    also stores ``observation.images.wrist_image``, but the OSS LIBERO eval
    server (``action_policy_server_libero.py``) is single-image, so training
    stays single-view for train/eval consistency.
  * **7D delta end-effector action** ``[dx, dy, dz, daxang_x, daxang_y,
    daxang_z, gripper]`` via ``build_action_spec(Pos(dim=3), Rot("axisangle"),
    Gripper())``. The stored ``action`` column is **already** the per-step delta
    command the LIBERO controller consumes (OpenVLA-OFT convention), so — unlike
    DROID ``ee_pose`` — no pose-delta reconstruction is needed.
  * **min-max normalization to ``[-1, 1]``** loaded from a shared stats JSON
    (``libero_lerobot_normalization.json``, generated from the four suites'
    ``meta/stats.json`` by ``tools/gen_libero_normalization.py``). The SAME file
    is passed to the eval server via ``--action-stats-path`` so train-time
    normalization and eval-time denormalization are guaranteed identical.
  * ``use_state=False`` (default): the LIBERO action is a relative delta, so
    prepending the absolute 8D proprio state (a different space/dim) is not
    meaningful. The action tensor is ``(chunk_length, 7)``; the decoded video
    window is ``chunk_length + 1`` frames.

Adjust the feature-key constants below if your LeRobot dataset uses different
column names; the defaults follow the ``nvidia/LIBERO_LeRobot_v3`` layout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.utils import log
from cosmos_framework.data.generator.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.generator.action.action_spec import Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.pose_utils import compute_idle_frames

Viewpoint = Literal["third_person_view"]

# --- LeRobot feature keys (nvidia/LIBERO_LeRobot_v3). Adjust if needed. ---
_IMAGE_FEATURE = "observation.images.image"        # agent-view (third-person) camera (video)
_WRIST_FEATURE = "observation.images.wrist_image"  # wrist (eye-in-hand) camera (video)
# With ``use_wrist=True`` the two views are vertically concatenated into ONE frame
# (agent-view on top, wrist on bottom) — Cosmos3 is single-input-video, so multi-view
# is handled by spatial concat (the DROID ``concat_view`` pattern). The transform then
# resizes the 512x256 stack to the 256 target (each view ~128x128).
_ACTION_FEATURE = "action"                   # [7] delta EE command [dpos(3), daxang(3), gripper]
_STATE_FEATURE = "observation.state"         # [8] absolute proprio [pos(3), axisangle(3), gripper, gripper]
# state[:7] = [x, y, z, axisangle(3), gripper] matches the 7D action channel layout
# exactly (drops the duplicated finger gripper). With ``use_state=True`` it is
# normalized with its OWN min/max (the absolute pose range — e.g. z up to ~1.37,
# axisangle up to ~3.67 — is far outside the delta-action range, so the delta
# stats would clamp it away) and prepended as the conditioning action[0].
_STATE_DIM = 7
_LIST_COLUMNS = {_ACTION_FEATURE, _STATE_FEATURE}

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")
_NORM_METHODS = ("minmax", "meanstd", "quantile")
_NORMALIZER_PATH = Path(__file__).parent / "libero_lerobot_normalization.json"


class LIBEROLeRobotDataset(Dataset):
    """LIBERO 7D delta-EE action dataset (single agent-view).

    Args:
        root: LeRobot v3.0 dataset root for ONE suite (must contain
            ``meta/info.json``). Mix suites at the dataloader level by listing
            multiple ``get_action_libero_sft_dataset`` entries.
        fps: Frames per second (LIBERO_LeRobot_v3 is 20).
        chunk_length: Number of commanded action steps per sample. The decoded
            window is ``chunk_length + 1`` frames (the extra leading frame is the
            conditioning observation).
        mode: ``"policy"`` (default) trains the policy task (predict actions +
            video from the first frame). ``"joint"`` randomly samples one of
            forward/inverse-dynamics/policy per item for multi-task training.
        action_normalization: ``"minmax"`` (default), ``"meanstd"``,
            ``"quantile"`` or ``None``. Stats are loaded from
            ``libero_lerobot_normalization.json`` (shared with the eval server).
        normalization_path: Override the stats JSON path.
        viewpoint: Only ``"third_person_view"`` (the agent-view camera) is
            supported.
    """

    def __init__(
        self,
        root: str,
        fps: float = 20.0,
        chunk_length: int = 16,
        mode: str = "policy",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "third_person_view",
        action_normalization: str | None = "minmax",
        normalization_path: str | None = None,
        use_state: bool = False,
        use_wrist: bool = False,
        wrist_key: str = _WRIST_FEATURE,
    ) -> None:
        super().__init__()
        if viewpoint != "third_person_view":
            raise NotImplementedError("LIBERO dataset only supports the single agent-view third_person_view.")
        if action_normalization is not None and action_normalization not in _NORM_METHODS:
            raise ValueError(f"action_normalization must be one of {_NORM_METHODS} or None.")

        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._action_normalization = action_normalization
        self._normalization_path = Path(normalization_path) if normalization_path else _NORMALIZER_PATH
        self._use_state = bool(use_state)
        self._use_wrist = bool(use_wrist)
        self._wrist_key = str(wrist_key)  # "...wrist_image" (LIBERO_LeRobot_v3) or "...image2" (lerobot/libero)
        self._domain_id = get_domain_id("libero")
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._state_stats: dict[str, torch.Tensor] | None = None

        self._root = Path(root)
        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        self._episodes = {
            int(row["episode_index"]): row
            for path in sorted((self._root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path).to_pylist()
        }
        # LIBERO_LeRobot_v3 stores the task string as the parquet (pandas) index
        # column ``__index_level_0__`` rather than a ``task`` column; accept either.
        tasks_table = pq.read_table(self._root / "meta" / "tasks.parquet")
        task_col = "task" if "task" in tasks_table.column_names else "__index_level_0__"
        self._tasks = {
            int(row["task_index"]): str(row[task_col]) for row in tasks_table.to_pylist()
        }

        # Compact, lazy frame index: read only the columns the sample builder
        # needs into contiguous numpy arrays (read-only after init, so DataLoader
        # worker forks share them copy-on-write). Mirrors DROID/PushT datasets.
        feature_cols = [_ACTION_FEATURE] + ([_STATE_FEATURE] if self._use_state else [])
        columns = ["index", "episode_index", "task_index", "timestamp", *feature_cols]
        index_parts, episode_parts, task_parts, ts_parts = [], [], [], []
        feature_parts: dict[str, list] = {c: [] for c in feature_cols}
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=columns)
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            for c in feature_cols:
                if c in _LIST_COLUMNS:
                    feature_parts[c].append(np.asarray(table[c].to_pylist(), dtype=np.float32))
                else:
                    feature_parts[c].append(np.asarray(table[c].to_numpy(), dtype=np.float32))
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._feat = {c: np.concatenate(feature_parts[c], axis=0).astype(np.float32)[order] for c in feature_cols}

        # Group frames into episodes; keep only within-episode chunk windows so a
        # sample never straddles an episode boundary (LeRobot v3 orders the global
        # frame index by episode, so episodes are contiguous after sorting).
        assert np.all(np.diff(self._row_episode) >= 0), "episode_index is not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        self._ep_vals = ep_vals.astype(np.int64)
        self._ep_starts = ep_starts.astype(np.int64)
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts - self._chunk_length)).astype(np.int64)

    # --- Properties mirroring DROID/PushT datasets (consumed by the experiment) ---

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def action_dim(self) -> int:
        return 7

    def _action_spec(self):
        return build_action_spec(Pos(dim=3), Rot("axisangle"), Gripper())

    @property
    def action_names(self) -> list[str]:
        return self._action_spec().names

    def _choose_mode(self) -> str:
        return random.choice(_MODE_CHOICES) if self._mode == "joint" else self._mode

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def _window_rows(self, start: int, stop: int, episode_index: int) -> list[dict[str, Any]]:
        """Reconstruct per-frame dicts for the half-open window ``[start, stop)``
        (guaranteed within a single episode)."""
        return [
            {
                "episode_index": episode_index,
                "task_index": int(self._row_task[j]),
                "timestamp": float(self._row_timestamp[j]),
                **{c: self._feat[c][j] for c in self._feat},
            }
            for j in range(start, stop)
        ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]

        observation_rows = self._window_rows(start, start + self._chunk_length + 1, episode_index)
        video = self._load_video(episode, observation_rows)
        raw_action = self._build_action(observation_rows)
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice(task.split(" | ")) if task else "complete the manipulation task"
        if self._use_wrist:
            # Tell the model the two-view layout (top = third-person, bottom = wrist).
            ai_caption = (
                "The top is the third-person agent view; the bottom is the wrist-mounted "
                f"camera view. Task: {ai_caption}"
            )

        return self._build_result(mode=mode, video=video, action=raw_action, ai_caption=ai_caption)

    def _build_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """7D delta-EE action ``[dpos(3), daxang(3), gripper]`` over the chunk.

        The decoded window is ``chunk+1`` frames; ``row[0]`` is the conditioning
        observation and ``rows[1:]`` carry the ``chunk`` commanded delta actions
        (the LIBERO controller consumes the stored ``action`` column directly).

        With ``use_state``, the current absolute proprio ``state[:7]`` (same channel
        layout as the action) is normalized with its own min/max and prepended as
        ``action[0]`` → ``(chunk+1, 7)``; the transform then conditions on it
        (``condition_frame_indexes_action=[0]``), matching the DROID/PushT framing."""
        action_rows = observation_rows[1:]
        action = np.asarray([r[_ACTION_FEATURE] for r in action_rows], dtype=np.float32)  # [chunk, 7]
        action_t = torch.from_numpy(action).float()
        if not self._use_state:
            return action_t
        init_state = np.asarray(observation_rows[0][_STATE_FEATURE], dtype=np.float32)[:_STATE_DIM]  # [7] absolute
        state_t = torch.from_numpy(init_state).float()
        if self._action_normalization is not None:
            state_t = normalize_action(state_t, self._action_normalization, self._load_state_stats())
        return torch.cat([state_t.unsqueeze(0), action_t], dim=0)  # [chunk+1, 7]

    def _decode_view(self, episode, video_key, timestamps):
        from_ts = float(episode.get(f"videos/{video_key}/from_timestamp", 0.0))
        # Clamp requested times to the view's stored range. The wrist videos for
        # libero_object/libero_10 are stored as ONE long concatenated mp4 (per-episode
        # ``from_timestamp`` offsets of 1000s+), and decoding at/just past the episode's
        # ``to_timestamp`` boundary can land torchcodec on an invalid packet. Clamping
        # to ``to_ts - 1/fps`` keeps every requested time strictly inside the segment.
        to_ts = episode.get(f"videos/{video_key}/to_timestamp")
        hi = (float(to_ts) - 1.0 / self._fps) if to_ts is not None else None
        wants = [from_ts + ts if hi is None else min(from_ts + ts, hi) for ts in timestamps]
        return decode_video_frames(self._video_path(episode, video_key), wants, self._tolerance_s)

    def _load_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        agent = self._decode_view(episode, _IMAGE_FEATURE, timestamps)  # (T,C,H,W)
        if not self._use_wrist:
            return agent
        # Multi-view: vertically concat agent-view (top) + wrist (bottom) into one
        # frame (Cosmos3 is single-input-video → DROID-style spatial concat). The
        # transform resizes the 2H x W stack down to the 256 target.
        try:
            wrist = self._decode_view(episode, self._wrist_key, timestamps)
        except Exception as e:  # noqa: BLE001 — rare torchcodec decode failure on the
            # long concatenated wrist mp4s; fall back to the agent view so training never
            # crashes on a bad packet. Logged (rate-limited) so the frequency is visible.
            LIBEROLeRobotDataset._wrist_decode_fail += 1
            if LIBEROLeRobotDataset._wrist_decode_fail <= 20 or LIBEROLeRobotDataset._wrist_decode_fail % 200 == 0:
                ep_i = episode.get("episode_index", "?")
                log.warning(
                    f"[LIBERO] wrist decode failed (ep={ep_i}, #{LIBEROLeRobotDataset._wrist_decode_fail}): "
                    f"{type(e).__name__}: {str(e)[:80]} — using agent view as wrist for this sample."
                )
            wrist = agent
        return torch.cat([agent, wrist], dim=-2)  # (T, C, 2H, W)

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        chunk_idx = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        file_idx = int(
            episode.get(
                f"videos/{video_key}/file_index",
                episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            episode_chunk=chunk_idx,
            episode_file=file_idx,
        )
        return self._root / rel

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
    ) -> dict[str, Any]:
        spec = self._action_spec()
        idle_frames = compute_idle_frames(
            action,
            spec,
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )
        if self._action_normalization is None:
            out_action = action
        else:
            out_action = normalize_action(action, self._action_normalization, self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": out_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
        }

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        """Load per-dim action stats from the shared normalization JSON.

        Using a committed stats file (rather than computing in-memory) guarantees
        the eval server denormalizes with the exact min/max the policy was
        trained on — pass the same file via ``--action-stats-path``."""
        if self._norm_stats is not None:
            return self._norm_stats
        self._norm_stats = {
            key: torch.from_numpy(value).float()
            for key, value in load_action_stats(str(self._normalization_path)).items()
        }
        return self._norm_stats

    def _load_state_stats(self) -> dict[str, torch.Tensor]:
        """Load the proprio-``state[:7]`` normalization stats (the ``"state"`` entry
        of the shared JSON). Separate from the delta-action stats so the absolute
        pose conditioning token maps cleanly into ``[-1, 1]``. The eval server loads
        the same entry via ``--state-stats-key state`` for train/eval consistency."""
        if self._state_stats is not None:
            return self._state_stats
        self._state_stats = {
            key: torch.from_numpy(value).float()
            for key, value in load_action_stats(str(self._normalization_path), stats_key="state").items()
        }
        return self._state_stats

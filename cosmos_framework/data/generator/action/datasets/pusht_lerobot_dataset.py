# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal PushT LeRobot dataset for Cosmos Action 2D-policy post-training.

PushT is a planar pushing benchmark: a single top-down camera and a **2D action**
``[pos_x, pos_y]`` (the commanded end-effector target in the image plane). The
framework already registers PushT as a first-class embodiment
(``domain_id=4``, ``raw_action_dim=2`` in
:mod:`cosmos_framework.data.generator.action.domain_utils`); this dataset is the
LeRobot-backed reader that feeds it.

It mirrors :class:`~cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset.DROIDLeRobotDataset`
(same compact lazy frame index, same per-sample window builder, same
``_build_result`` contract consumed by ``ActionTransformPipeline``) but is
simplified for PushT:

  * **Single view** (no wrist/exterior concat).
  * **2D action** via ``build_action_spec(Pos(dim=2))`` — both columns are
    ``DimType.POS``.
  * **min-max normalization to ``[-1, 1]``** computed in-memory from the dataset
    (PushT coordinates are large pixel values, so unlike DROID ``joint_pos`` they
    must be normalized for the action heads to converge). Override via
    ``action_normalization`` (``"minmax"`` | ``"meanstd"`` | ``"quantile"`` | ``None``).
  * ``use_state=True`` prepends the initial observed agent position so the action
    tensor is ``(chunk + 1, 2)`` — matching the DROID ``use_state`` framing.

Adjust the feature-key constants below if your LeRobot dataset uses different
column names; the defaults follow the canonical ``lerobot/pusht`` v3.0 layout.
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
from torchvision.transforms import functional as transforms_F

from cosmos_framework.data.generator.action.action_normalization import normalize_action
from cosmos_framework.data.generator.action.action_spec import Pos, build_action_spec
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.pose_utils import compute_idle_frames

Viewpoint = Literal["third_person_view"]

# --- LeRobot feature keys (PushT-Cosmos-Policy-lerobot v3.0). Adjust if needed. ---
_IMAGE_FEATURE = "observation.images.image"  # single top-down camera (video)
_STATE_FEATURE = "observation.state"         # [2] observed agent (x, y)
_ACTION_FEATURE = "action"                   # [2] commanded target (x, y)
_TRAIN_RESOLUTION = 256                       # upscale native 128 → 256 (fill frame, no padding)
# Columns whose parquet dtype is list<float> (need to_pylist -> stacked array).
_LIST_COLUMNS = {_STATE_FEATURE, _ACTION_FEATURE}

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")
_NORM_METHODS = ("minmax", "meanstd", "quantile")


class PushTLeRobotDataset(Dataset):
    """PushT 2D-action dataset (single view, ``[pos_x, pos_y]``).

    Args:
        root: LeRobot v3.0 dataset root (must contain ``meta/info.json``).
        fps: Frames per second (canonical PushT is 10).
        chunk_length: Number of commanded action steps per sample. The decoded
            window is ``chunk_length + 1`` frames (the extra frame is the initial
            observation, prepended as state when ``use_state``).
        mode: ``"policy"`` (default) trains the policy task (predict actions + video
            from the first frame). ``"joint"`` randomly samples one of
            forward/inverse-dynamics/policy per item for multi-task training
            (matching the DROID recipe).
        action_normalization: ``"minmax"`` (default), ``"meanstd"``, ``"quantile"``
            or ``None``. Stats are computed in-memory from the action column.
        use_state: Prepend the initial observed agent position → ``(chunk+1, 2)``.
        viewpoint: Only ``"single_view"`` is supported.
    """

    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "policy",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "third_person_view",
        action_normalization: str | None = "minmax",
        use_state: bool = True,
    ) -> None:
        super().__init__()
        if viewpoint != "third_person_view":
            raise NotImplementedError("PushT dataset only supports the single top-down third_person_view.")
        if action_normalization is not None and action_normalization not in _NORM_METHODS:
            raise ValueError(f"action_normalization must be one of {_NORM_METHODS} or None.")

        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._action_normalization = action_normalization
        self._use_state = bool(use_state)
        self._domain_id = get_domain_id("pusht")
        self._norm_stats: dict[str, torch.Tensor] | None = None

        self._root = Path(root)
        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        self._episodes = {
            int(row["episode_index"]): row
            for path in sorted((self._root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path).to_pylist()
        }
        self._tasks = {
            int(row["task_index"]): str(row["task"])
            for row in pq.read_table(self._root / "meta" / "tasks.parquet").to_pylist()
        }

        # Compact, lazy frame index: read only the columns the sample builder needs
        # into contiguous numpy arrays (read-only after init, so DataLoader worker
        # forks share them copy-on-write). Mirrors DROIDLeRobotDataset.
        feature_cols = [_STATE_FEATURE, _ACTION_FEATURE]
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

    # --- Properties mirroring DROIDLeRobotDataset (consumed by the experiment) ---

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
        return 2

    def _action_spec(self):
        return build_action_spec(Pos(dim=2))

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
        ai_caption = random.choice(task.split(" | ")) if task else "push the T block to the target"

        return self._build_result(mode=mode, video=video, action=raw_action, ai_caption=ai_caption)

    def _build_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """2D action ``[pos_x, pos_y]`` over the chunk. The window is ``chunk+1``
        frames: ``row[0]`` is the initial observed state (prepended when
        ``use_state``); ``rows[1:]`` are the ``chunk`` commanded actions."""
        action_rows = observation_rows[1:]
        action = np.asarray([r[_ACTION_FEATURE] for r in action_rows], dtype=np.float32)  # [chunk, 2]
        if self._use_state:
            init_state = np.asarray(observation_rows[0][_STATE_FEATURE], dtype=np.float32)[None, :]  # [1, 2]
            action = np.concatenate([init_state, action], axis=0)  # [chunk + 1, 2]
        return torch.from_numpy(action).float()

    def _load_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        from_ts = float(episode.get(f"videos/{_IMAGE_FEATURE}/from_timestamp", 0.0))
        video = decode_video_frames(  # (T, C, H, W) float in [0, 1], native 128x128
            self._video_path(episode, _IMAGE_FEATURE),
            [from_ts + ts for ts in timestamps],
            self._tolerance_s,
        )
        # Upscale the native 128 frame to the 256 training resolution so the scene
        # FILLS the frame. The transform's aspect-preserving path caps scaling at 1.0
        # and reflection-pads 128 into a 256 corner (~3/4 of the frame is padding);
        # resizing here makes image_size = [256,256,256,256] (no padding) and keeps
        # prompt/resolution metadata consistent with eval, where the server upscales
        # the 128 frame to 256 (send image_size=256). PushT views are square, so the
        # 1:1 resize introduces no distortion.
        if video.shape[-1] != _TRAIN_RESOLUTION or video.shape[-2] != _TRAIN_RESOLUTION:
            video = transforms_F.resize(
                video,
                size=[_TRAIN_RESOLUTION, _TRAIN_RESOLUTION],
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
        return video

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
        """Compute per-dim action stats in-memory from the full action column."""
        if self._norm_stats is not None:
            return self._norm_stats
        a = torch.from_numpy(self._feat[_ACTION_FEATURE]).float()  # [N, 2]
        self._norm_stats = {
            "mean": a.mean(dim=0),
            "std": a.std(dim=0),
            "min": a.amin(dim=0),
            "max": a.amax(dim=0),
            "q01": torch.quantile(a, 0.01, dim=0),
            "q99": torch.quantile(a, 0.99, dim=0),
        }
        return self._norm_stats

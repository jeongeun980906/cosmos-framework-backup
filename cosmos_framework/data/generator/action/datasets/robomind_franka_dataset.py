# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboMIND Franka LeRobot dataset — single-arm and dual-arm variants."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_normalization import load_action_stats
from cosmos_framework.data.generator.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.utils import log
from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view", "third_person_view"]

# Dual-arm uses camera_front as the top view; single-arm uses camera_top.
_IMAGE_FEATURES_DUAL = {
    "top": "observation.images.camera_front",
    "left": "observation.images.camera_left",
    "right": "observation.images.camera_right",
}
_IMAGE_FEATURES_SINGLE = {
    "top": "observation.images.camera_top",
    "left": "observation.images.camera_left",
    "right": "observation.images.camera_right",
}
_STATE_FEATURE = "observation.states.end_effector"
_ACTION_FEATURE = "actions.joint_position"

# --- AgileX Cobot Magic (RoboMIND's dual-arm robot; domain 12, same 20-D contract) ----
#
# RoboMIND has no dual-arm Franka -- its bimanual robot is the AgileX, and the
# BAAI-DataCube LeRobot conversion lays it out differently from the Franka repos:
# every per-arm quantity is a SEPARATE feature rather than one concatenated vector,
# and the wrist cameras are named ..._wrist. Measured on 1.6M frames of the local
# slice:
#   observation.states.end_effector_{left,right}  [7] = pos(3) + euler(3) + grip(1)
#   actions.end_effector_{left,right}             [7] = same, as a command
#   actions.joint_position_{left,right}           [7] = 6 joints + the grip in raw
#                                                       motor units (0..7.5)
# The action-side end_effector gripper is the clean one: it tracks the state channel at
# r=0.957 and lives in metres of jaw opening, p50 = -0.002 (closed), p99 = 0.588 (open),
# with 74.6% of frames below 0.05. The Franka path emits ``1 - raw`` so that 1 means
# CLOSED; the same convention here needs the width divided by its open extent first.
_IMAGE_FEATURES_AGILEX = {
    "top": "observation.images.camera_front",
    "left": "observation.images.camera_left_wrist",
    "right": "observation.images.camera_right_wrist",
}
_STATE_FEATURE_AGILEX = "observation.states.end_effector_{side}"
_ACTION_FEATURE_AGILEX = "actions.end_effector_{side}"
_AGILEX_GRIP_OPEN = 0.588   # p99 of the jaw-opening channel over the local slice

# 90-degree clockwise rotation about the Z axis in the local frame. This matches
# the production RoboMIND Franka wrapper conversion to OpenCV coordinates.
_ROBOMIND_FRANKA_TO_OPENCV: np.ndarray = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

_NORMALIZER_PATH_DUAL = Path(__file__).parent.parent / "normalizer_stats/robomind_franka_dual_stats.json"
_NORMALIZER_PATH_SINGLE = Path(__file__).parent.parent / "normalizer_stats/robomind_franka_stats.json"
_NORMALIZER_PATH_AGILEX = Path(__file__).parent.parent / "normalizer_stats/robomind_agilex_dual_stats.json"


def _video_backend() -> str | None:
    """Decoder backend for LeRobot video reads, overridable by env.

    LeRobot's ``get_safe_default_codec`` picks torchcodec whenever the package merely
    IMPORTS, not when it actually works — and torchcodec needs ffmpeg shared libraries
    (libavutil.so.56 and friends) that a bare box may not have. There the default fails
    at read time with an opaque "Could not load this library", not at import. Setting
    ``LEROBOT_VIDEO_BACKEND=pyav`` selects the pure-Python path instead; unset keeps
    LeRobot's own choice, which is right inside the training container.
    """
    return os.environ.get("LEROBOT_VIDEO_BACKEND") or None


def _dual_arm_action_spec():
    return build_action_spec(
        Pos(prefix="left"),
        Rot("rot6d", prefix="left"),
        Gripper(prefix="left"),
        Pos(prefix="right"),
        Rot("rot6d", prefix="right"),
        Gripper(prefix="right"),
    )


class RoboMINDFrankaDataset(ActionBaseDataset):
    """RoboMIND Franka dataset — single-arm (10D) or dual-arm (20D) cartesian actions.

    Single-arm (``robomind-franka``): ``[pos_delta(3), rot6d_delta(6), gripper(1)]``
    Dual-arm (``robomind-franka-dual``):
        ``[left_pos(3), left_rot6d(6), left_gripper(1),
           right_pos(3), right_rot6d(6), right_gripper(1)]``
    """

    _SUPPORTED_EMBODIMENTS = ("robomind-franka", "robomind-franka-dual", "robomind-agilex-dual")

    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "joint",
        embodiment_type: str = "robomind-franka-dual",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "concat_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
        drop_episodes: set[int] | list[int] | None = None,
    ) -> None:
        if embodiment_type not in self._SUPPORTED_EMBODIMENTS:
            raise ValueError(
                f"RoboMINDFrankaDataset only supports {self._SUPPORTED_EMBODIMENTS}, "
                f"got embodiment_type={embodiment_type!r}."
            )
        if viewpoint not in ("concat_view", "third_person_view"):
            raise NotImplementedError(f"RoboMINDFrankaDataset does not support viewpoint={viewpoint!r}.")
        self._embodiment_type = embodiment_type
        self._image_features = {
            "robomind-franka": _IMAGE_FEATURES_SINGLE,
            "robomind-franka-dual": _IMAGE_FEATURES_DUAL,
            "robomind-agilex-dual": _IMAGE_FEATURES_AGILEX,
        }[embodiment_type]
        super().__init__(
            root=root,
            domain_name=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        # Drop specific episodes rather than whole task repos. RoboMIND's AV1 videos have
        # frames the decoder refuses, but they cluster into a handful of episodes — 5 of
        # 105 in one measured repo — so banning the repo throws away 95% of a task and,
        # with it, the task diversity this replay stream exists to provide. LeRobot v3
        # packs every episode of a repo into one mp4 per camera, so the data cannot be
        # re-fetched episode-wise; filtering the row index is the only way to keep the
        # task and lose only what is broken.
        if drop_episodes:
            drop = {int(e) for e in drop_episodes}
            before = len(self._rows)
            self._rows = [r for r in self._rows if int(r["episode_index"]) not in drop]
            log.info(
                f"{embodiment_type} {Path(root).name}: dropped {len(drop)} bad episode(s), "
                f"{before - len(self._rows)} of {before} rows removed"
            )

    @property
    def repo_name(self) -> str:
        """Directory name of this task's LeRobot repo — the task's human-readable label."""
        return self._root.name

    @property
    def action_dim(self) -> int:
        return 10 if self._embodiment_type == "robomind-franka" else 20

    def _action_spec(self) -> ActionSpec:
        if self._embodiment_type == "robomind-franka":
            return build_action_spec(Pos(), Rot("rot6d"), Gripper())
        return _dual_arm_action_spec()

    @classmethod
    def _stats_path(cls) -> Path:
        # Class-level default (no instance to branch on) — matches the
        # constructor's default embodiment_type ("robomind-franka-dual").
        return _NORMALIZER_PATH_DUAL

    def load_action_stats(self) -> dict[str, torch.Tensor]:
        """Instance-aware override: respects ``self._embodiment_type``.

        The inherited classmethod always resolves via ``_stats_path()``, which
        has no instance to branch on and is hardcoded to the dual-arm file. A
        single-arm instance calling ``load_action_stats()`` would otherwise
        silently get 20D dual-arm stats instead of its own 10D stats.
        """
        path = {
            "robomind-franka": _NORMALIZER_PATH_SINGLE,
            "robomind-franka-dual": _NORMALIZER_PATH_DUAL,
            "robomind-agilex-dual": _NORMALIZER_PATH_AGILEX,
        }[self._embodiment_type]
        return {
            key: torch.from_numpy(value).float()
            for key, value in load_action_stats(str(path)).items()
        }

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            self._norm_stats = self.load_action_stats()
        return self._norm_stats

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Build one sample, stepping over windows whose video will not decode.

        RoboMIND's mp4s are not uniformly readable. torchcodec raises
        ``Could not push packet to decoder: Invalid data found when processing input``
        on some AgileX windows — twice now, at iter 50 and iter 773, each time killing a
        32-rank job through ChildFailedError because one DataLoader worker on one of the
        two agilex ranks hit a bad window. PyAV opens all 115 files and decodes their
        head and tail, so the defect is neither a truncated download nor a whole broken
        file; it is specific frames that the stricter decoder refuses.

        A replay stream at 6.25% of ranks does not justify losing the run over a handful
        of unreadable windows, so step to the next window instead — loudly, once per
        offending index, and only a bounded number of times. Exhausting the budget still
        raises, because a stream that cannot produce ANY sample is a real failure and
        must not be silently skipped into an empty epoch.
        """
        pick = int(idx)
        for attempt in range(self._DECODE_RETRIES):
            try:
                return self._build_sample(pick)
            except RuntimeError as e:                                # noqa: PERF203
                if "decoder" not in str(e) and "Invalid data" not in str(e):
                    raise
                log.warning(
                    f"{self._embodiment_type}: undecodable window at idx {pick} "
                    f"({attempt + 1}/{self._DECODE_RETRIES}) — {str(e).splitlines()[-1][:90]}"
                )
                # Jump at RANDOM, not by a fixed stride. The failures cluster — whole
                # runs of consecutive episodes at the tail of a file go bad together, in
                # one measured case 2,347 consecutive windows — so a fixed step only
                # escapes if it happens to exceed the run length, and nothing guarantees
                # that for every repo. A uniform redraw makes each retry an independent
                # trial, which is what turns a ~4% per-window failure rate into ~4e-12
                # for eight in a row.
                pick = random.randrange(len(self))
        raise RuntimeError(
            f"{self._embodiment_type}: {self._DECODE_RETRIES} consecutive windows from idx "
            f"{idx} failed to decode; the dataset root is likely unusable, not just sparse."
        )

    _DECODE_RETRIES = 16

    def episode_of(self, idx: int) -> int:
        """Episode index of the window at ``idx``, from its first observation row.

        Same index arithmetic as ``_build_sample``, so the answer is the episode the
        sample would actually be built from. Reads the already-materialized row index,
        so it costs a list lookup and no decode.
        """
        row_idx = (int(idx) % len(self)) * self._sample_stride
        return int(self._rows[row_idx]["episode_index"])

    def _row_arm_features(self) -> np.ndarray:
        """Per-ROW ``[pos_x, pos_y, pos_z, gripper_closedness]``, one block per arm.

        Absolute end-effector position in metres and gripper closedness in [0, 1], read
        straight from the row index — the same columns ``_build_raw_action_*`` reads, so
        the key and the action it retrieves against speak the same units. No video decode.
        """
        rows = self._rows
        if self._embodiment_type == "robomind-agilex-dual":
            blocks: list[np.ndarray] = []
            for side in ("left", "right"):
                state = np.asarray([r[_STATE_FEATURE_AGILEX.format(side=side)] for r in rows], dtype=np.float32)
                cmd = np.asarray([r[_ACTION_FEATURE_AGILEX.format(side=side)] for r in rows], dtype=np.float32)
                blocks += [state[:, 0:3], 1.0 - np.clip(cmd[:, [6]] / _AGILEX_GRIP_OPEN, 0.0, 1.0)]
            return np.concatenate(blocks, axis=-1)
        state = np.asarray([r[_STATE_FEATURE] for r in rows], dtype=np.float32)
        gripper = np.asarray([r[_ACTION_FEATURE] for r in rows], dtype=np.float32)
        if self._embodiment_type == "robomind-franka":
            return np.concatenate([state[:, 0:3], 1.0 - gripper[:, [7]]], axis=-1)
        return np.concatenate(
            [state[:, 0:3], 1.0 - gripper[:, [7]], state[:, 6:9], 1.0 - gripper[:, [15]]], axis=-1
        )

    _RETRIEVAL_KEY_PER_ARM = (
        "pos_x", "pos_y", "pos_z", "gripper", "disp_x", "disp_y", "disp_z", "grasps",
    )
    _GRASP_CLOSED = 0.5   # closedness above which the gripper counts as holding something

    def retrieval_key_names(self) -> list[str]:
        """Column names of :meth:`window_retrieval_keys`, so callers can weight channels by
        meaning instead of by a hardcoded stride."""
        arms = ("",) if self._embodiment_type == "robomind-franka" else ("left_", "right_")
        return [f"{a}{n}" for a in arms for n in self._RETRIEVAL_KEY_PER_ARM]

    def window_retrieval_keys(self) -> np.ndarray:
        """Per-window retrieval key, ``[n_windows, 7 * n_arms]`` float32.

        Per arm: absolute EE position and gripper closedness at the window's FIRST
        observation row, the net EE displacement ACROSS the window, and the number of
        gripper closures completed so far in the episode.

        t0 state because that is the frame the query is conditioned on, so matching on it
        asks the question that matters: did this demo start from the configuration the
        query starts from? Displacement because t0 state alone does not pin down what
        happens next -- the arm passes through the same point on the way out and on the way
        back, and matching position only, the retrieved demo moved in the OPPOSITE direction
        29% of the time (measured on insert_marker; 12.5% once displacement is in the key).

        Absolute coordinates are used as-is. Demo and query are the SAME robot in the SAME
        task repo, so a pose already means the same thing on both sides and there is nothing
        to canonicalize away.
        """
        feats = self._row_arm_features()
        grasps = self._row_grasp_counts(feats)
        first = np.arange(len(self), dtype=np.int64) * self._sample_stride
        last = np.minimum(first + self._chunk_length, feats.shape[0] - 1)
        blocks: list[np.ndarray] = []
        for arm in range(feats.shape[1] // 4):
            pos0 = feats[first, arm * 4 : arm * 4 + 3]
            blocks += [
                pos0,
                feats[first, arm * 4 + 3 : arm * 4 + 4],
                feats[last, arm * 4 : arm * 4 + 3] - pos0,
                grasps[first, arm : arm + 1],
            ]
        return np.ascontiguousarray(np.concatenate(blocks, axis=-1), dtype=np.float32)

    def _row_grasp_counts(self, feats: np.ndarray) -> np.ndarray:
        """Per-row count of gripper closures so far in the episode, one column per arm.

        Task progress that robot state alone cannot express. A repetitive task brings the
        arm back through the same poses once per object, so matching on pose and motion can
        land the demo on the wrong CYCLE -- visibly so on ``packplate``, where the demo had
        plates already racked while the query was still on its first. Closures completed is
        what separates those cycles, and it is robust to pacing in a way relative time is
        not: a slow demonstration and a fast one agree on how many plates are done.
        """
        rows = self._rows
        ep = np.fromiter((int(r["episode_index"]) for r in rows), dtype=np.int64, count=len(rows))
        new_episode = np.empty(len(ep), dtype=bool)
        new_episode[0] = True
        np.not_equal(ep[1:], ep[:-1], out=new_episode[1:])
        n_arms = feats.shape[1] // 4
        counts = np.zeros((len(ep), n_arms), dtype=np.float32)
        for arm in range(n_arms):
            closed = feats[:, arm * 4 + 3] > self._GRASP_CLOSED
            rising = np.empty(len(closed), dtype=bool)
            rising[0] = closed[0]
            np.logical_and(closed[1:], ~closed[:-1], out=rising[1:])
            rising[new_episode] = closed[new_episode]  # an episode never inherits the last one's state
            total = np.cumsum(rising)
            # cumsum restarts at each episode: subtract the running total at its first row.
            base = np.maximum.accumulate(np.where(new_episode, total - rising, 0))
            counts[:, arm] = total - base
        return counts

    def episode_window_spans(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-episode window ranges in this dataset's own index space.

        The descriptor the ICL stream retrieves demos through: for each episode, the
        half-open range of window indices ``[start, end)`` whose ENTIRE span lies inside
        that episode. Window ``i`` is built from rows ``[i*stride, i*stride + chunk]``
        inclusive, so a window near an episode boundary reaches into the next
        demonstration; those are excluded here because a clip made of two different
        demonstrations is not a demonstration. (Queries are unaffected — the query index
        space stays exactly the BC stream's, so ICL and BC see the same query stream.)

        Rows are index-sorted, so every episode is one contiguous run and the whole
        descriptor is three small arrays rather than a per-window table.

        Returns:
            ``(episode_ids, starts, ends)``, episodes with no fully-contained window dropped.
        """
        rows = self._rows
        ep = np.fromiter((int(r["episode_index"]) for r in rows), dtype=np.int64, count=len(rows))
        change = np.flatnonzero(np.diff(ep)) + 1
        row_starts = np.concatenate(([0], change))
        row_ends = np.concatenate((change, [len(ep)]))  # exclusive
        stride, span = self._sample_stride, self._chunk_length
        n_windows = len(self)
        # First window at or after the episode's first row; last window whose final
        # observation row (i*stride + span) still belongs to the episode.
        starts = -(-row_starts // stride)
        ends = (row_ends - 1 - span) // stride + 1
        starts = np.clip(starts, 0, n_windows)
        ends = np.clip(ends, 0, n_windows)
        keep = ends > starts
        return ep[row_starts][keep], starts[keep], ends[keep]

    def _build_sample(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx) % len(self)
        row_idx = idx * self._sample_stride
        observation_rows = self._rows[row_idx : row_idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        episode = self._episodes[int(observation_rows[0]["episode_index"])]
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        if self._viewpoint == "concat_view":
            video = self._load_concat_video(episode, observation_rows)
            if self._embodiment_type == "robomind-franka":
                view_desc = (
                    "The top row shows a third-person perspective looking towards the single-arm Franka robot from above. "
                    "The bottom-left view looks at the scene from the left side, and the bottom-right view looks at the scene from the right side."
                )
            elif self._embodiment_type == "robomind-agilex-dual":
                view_desc = (
                    "The top row shows a third-person perspective looking towards the dual-arm AgileX robot from the front. "
                    "The bottom-left and bottom-right views are the left and right wrist cameras."
                )
            else:
                view_desc = (
                    "The top row shows a third-person perspective looking towards the dual-arm Franka robot from the front. "
                    "The bottom-left view looks at the scene from the left side, and the bottom-right view looks at the scene from the right side."
                )
        else:
            video = self._load_single_video(episode, observation_rows)
            view_desc = None

        if self._embodiment_type == "robomind-franka":
            raw_action, initial_pose = self._build_raw_action_single(observation_rows, action_rows)
            extras: dict[str, Any] = {"initial_pose": initial_pose}
        elif self._embodiment_type == "robomind-agilex-dual":
            raw_action, initial_pose_left, initial_pose_right = self._build_raw_action_agilex(
                observation_rows, action_rows)
            extras = {"initial_pose": initial_pose_left, "initial_pose_right": initial_pose_right}
        else:
            raw_action, initial_pose_left, initial_pose_right = self._build_raw_action_dual(observation_rows, action_rows)
            extras = {"initial_pose": initial_pose_left, "initial_pose_right": initial_pose_right}

        if view_desc is not None:
            extras["additional_view_description"] = view_desc

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            **extras,
        )

    def _load_single_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        video_key = self._image_features["top"]
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        return decode_video_frames(
            self._video_path(episode, video_key),
            [float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
            self._tolerance_s,
            backend=_video_backend(),
        )

    def _load_concat_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        timestamps = [float(row["timestamp"]) for row in observation_rows]
        frames_by_view = {
            name: decode_video_frames(
                self._video_path(episode, video_key),
                [float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
                self._tolerance_s,
                backend=_video_backend(),
            )
            for name, video_key in self._image_features.items()
        }

        top = frames_by_view["top"]
        left = frames_by_view["left"]
        right = frames_by_view["right"]
        _, _, h_top, w_top = top.shape
        half_h, half_w = h_top // 2, w_top // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    # Quantile normalization divides by the 1%-99% spread, and AgileX's frame-to-frame
    # rot6d diagonal is a delta at 1.0: that spread is 0.0016, so the scaler is a 1246x
    # amplifier. The 0.009% of frames carrying a physically impossible rotation (27-64
    # degrees inside one 0.1 s step) then land at |z| up to 876, where one sample's
    # squared error is ~760,000x a normal one. grad_clip is applied as ONE global scalar
    # over every parameter, so a single such sample renormalises the entire step onto its
    # own direction. Rebuilding the stats cannot fix this: swapping the quantiles for
    # min/max widens the denominator to 0.74 and crushes real rotation signal to 0.004,
    # which is worse. The tail is the defect, so only the tail is cut. Franka's whole
    # distribution maxes at 2.1, so this bound never touches clean data.
    _ACTION_CLIP = 5.0

    def _build_result(self, **kwargs: Any) -> dict[str, Any]:
        result = super()._build_result(**kwargs)
        result["action"] = result["action"].clamp(-self._ACTION_CLIP, self._ACTION_CLIP)
        return result

    def _build_relative_poses(
        self,
        positions: np.ndarray,
        euler_xyz: np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor]:
        poses_abs = build_abs_pose_from_components(positions, euler_xyz, "euler_xyz")
        poses_abs[:, :3, :3] = poses_abs[:, :3, :3] @ _ROBOMIND_FRANKA_TO_OPENCV
        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        return poses_rel, initial_pose

    def _build_raw_action_single(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        gripper = np.asarray([row[_ACTION_FEATURE] for row in action_rows], dtype=np.float32)
        poses_rel, initial_pose = self._build_relative_poses(state[:, 0:3], state[:, 3:6])
        action = np.concatenate(
            [poses_rel[-self._chunk_length :], 1.0 - gripper[-self._chunk_length :, [7]]],
            axis=-1,
        )  # [T, 10]
        return torch.from_numpy(action).float(), initial_pose

    def _build_raw_action_dual(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = np.asarray([row[_STATE_FEATURE] for row in observation_rows], dtype=np.float32)
        gripper = np.asarray([row[_ACTION_FEATURE] for row in action_rows], dtype=np.float32)

        poses_rel_left, initial_pose_left = self._build_relative_poses(state[:, 0:3], state[:, 3:6])
        poses_rel_right, initial_pose_right = self._build_relative_poses(state[:, 6:9], state[:, 9:12])
        action = np.concatenate(
            [
                poses_rel_left[-self._chunk_length :],
                1.0 - gripper[-self._chunk_length :, [7]],
                poses_rel_right[-self._chunk_length :],
                1.0 - gripper[-self._chunk_length :, [15]],
            ],
            axis=-1,
        )  # [T, 20]
        return torch.from_numpy(action).float(), initial_pose_left, initial_pose_right

    def _build_raw_action_agilex(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AgileX dual-arm -> the same 20-D contract, from per-arm split features."""
        parts: list[np.ndarray] = []
        initial: list[torch.Tensor] = []
        for side in ("left", "right"):
            state = np.asarray([row[_STATE_FEATURE_AGILEX.format(side=side)] for row in observation_rows],
                               dtype=np.float32)
            cmd = np.asarray([row[_ACTION_FEATURE_AGILEX.format(side=side)] for row in action_rows],
                             dtype=np.float32)
            poses_rel, initial_pose = self._build_relative_poses(state[:, 0:3], state[:, 3:6])
            # Jaw opening (metres) -> closedness in [0, 1], matching the Franka path's
            # ``1 - raw`` so that 1 is CLOSED on both arms of the contract.
            grip = 1.0 - np.clip(cmd[-self._chunk_length :, [6]] / _AGILEX_GRIP_OPEN, 0.0, 1.0)
            parts += [poses_rel[-self._chunk_length :], grip]
            initial.append(initial_pose)
        action = np.concatenate(parts, axis=-1)  # [T, 20]
        return torch.from_numpy(action).float(), initial[0], initial[1]

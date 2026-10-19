# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tabletop-Sim ALOHA bimanual dataset (20D absolute end-effector, HDF5-backed).

Exists to give the RoboCasa 400-step probe a SECOND target with a different action
width. Every warm-start conclusion so far rests on one 7-DOF benchmark routed through
one untrained head row (domain 5); a 20D bimanual benchmark routed through a different
untrained row (16) says whether those conclusions are about the backbone or about
RoboCasa specifically.

Layout. ``actions/ee_6d_pos`` is ``[T, 20]`` = per arm ``[pos(3), rot6d(6), grip(1)]``,
left then right -- the same channel contract row 16 already carries for the YAM/eva
embodiments, and the same one RoboMIND's dual-arm sets use at row 12.

Row 16 is untrained in every checkpoint under comparison: ``llm2action.bias[16]`` is
identically 0 in the public Edge base, in the ICL mid, and in the BC-replay mid, so all
three start post-training from the same random head and the probe measures the backbone
alone. That is the same property that makes row 5 the right choice for RoboCasa.

Poses here are ABSOLUTE, not frame-to-frame deltas (positions sit at 0.18-0.64 m). That
differs from the RoboCasa recipe but is identical across the three arms being compared,
which is what the comparison requires. It also means the quantile widths are healthy
(the narrowest channel spans ~1.0), so this reader needs none of the tail-clipping the
AgileX delta-rot6d channels required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.generator.action.domain_utils import get_action_dim, get_domain_id

_ACTION_KEY = "actions/ee_6d_pos"
_STATE_KEY = "observations/states/ee_6d_pos"
_LANG_KEY = "observations/states/language_instruction"
# Scene view on top, the two wrists side by side underneath. ``back`` is dropped on purpose
# -- it is the most redundant of the four views and a fourth row only inflates the canvas.
#
# The layout matters for cost, not just framing. The source resolutions differ:
#   front 480x640, wrist_left 240x320, wrist_right 240x320
# The first version matched every view to the WIDEST one, which upscaled both wrists 2x to
# 480x640 and stacked all three vertically -> 1440x640. That quadruples each wrist's pixel
# count while adding no information, and it showed up directly in the bill: 44,928 vision
# tokens and 6.8 s/iter on 3 nodes, against RoboCasa's 21,168 tokens and 3.62 s/iter.
#
# Tiling the wrists in a row instead keeps every source pixel at its native scale and needs
# no interpolation at all:
#   front            480x640
#   wristL | wristR  240x640   (two 240x320 side by side)
#   ------------------------
#   canvas           720x640    <- exactly half the old area, ~2x fewer vision tokens
#
# Only the interpolated pixels are gone; nothing real is discarded. Dropping the front view
# to 240x320 as well would quarter the area, but that throws away actual scene detail.
# 3인칭 카메라를 front / back 중에 고른다. 기본은 front (기존 동작 유지).
#
# 왜 스위치가 필요한가: cosmos_policy 계열(자체 arm, GR00T·pi0.5·BeingH 포트)은
# convert_tabletop_to_egoverse57.py 의 GRID 에서 primary third-person 을 `back` 으로
# 잡는다. 우리만 `front` 를 쓰고 있어서, 지금 비교표는 모델 차이와 시점 차이가 섞여 있다.
# ALOHA_SCENE_CAM=back 으로 학습하면 시점 변수를 분리할 수 있다.
#
# hdf5 에는 back / front 둘 다 480x640 으로 들어 있어 레이아웃은 그대로 쓸 수 있다
# (손목 240x320 두 장이 정확히 폭을 채운다). 즉 이 스위치는 해상도·배치를 건드리지
# 않고 시점만 바꾼다.
_SCENE_CAM = os.environ.get("ALOHA_SCENE_CAM", "front").strip().lower()
if _SCENE_CAM not in ("front", "back"):
    raise ValueError(f"ALOHA_SCENE_CAM must be front|back, got {_SCENE_CAM!r}")

DEFAULT_CAMERAS: tuple[str, ...] = (
    f"observations/images/{_SCENE_CAM}",
    "observations/images/wrist_left",
    "observations/images/wrist_right",
)
# 캡션 스타일. 기본은 layout (기존 동작 유지).
#
#   layout  "Top: <cam> third-person camera. Bottom-left: ... Task: <instruction>"
#           이어붙인 캔버스라 어느 칸이 무엇인지 문장으로 알려준다.
#   plain   "<instruction>"  -- cosmos_policy 계열과 동일.
#
# cosmos_policy(자체 arm, GR00T·pi0.5·BeingH 포트)는 hdf5 의 language_instruction 을
# 그대로 넘긴다. 카메라마다 이름 있는 고정 슬롯(front_1/sideleft/sideright/center_top)에
# 넣기 때문에 배치를 텍스트로 알려줄 필요가 없다. 우리만 접두사를 붙이고 있어서
# 비교표에 변수가 하나 더 섞여 있다. plain 으로 두면 그 변수가 빠진다.
#
# 학습과 eval 이 반드시 같은 값을 써야 한다 -- 클라이언트도 같은 환경변수를 읽는다.
# 캔버스 해상도. 기본은 native (리사이즈 없음, 기존 동작 유지).
# "HxW" 를 주면 조립된 캔버스를 그 크기로 줄인다.
#
# 왜 필요한가: cosmos_policy pv0 는 뷰당 224x224 x 3뷰 = 150,528 px 로 돈다
# (뷰를 latent 슬롯으로 분리해서 그리드 448^2 의 우하단 공백을 없앤 판이다).
# 우리 native 캔버스는 720x640 = 460,800 px 라 3배가 넘는다. 비전 토큰으로 보면
# WAN2.2 VAE(16x) + patch 2 기준 H*W/1024 이므로
#   pv0   224^2/1024 x 3 = 147 토큰/프레임
#   native 720x640/1024   = 450 토큰/프레임
# 416x352 로 줄이면 143 토큰이라 pv0 와 사실상 같아진다.
#
# 종횡비 1.125 를 정확히 지키면서 32의 배수를 맞추면 288x256(72토큰) 아니면
# 576x512(288토큰)뿐이라 147 을 못 맞춘다. pv0 자신도 손목 240x320 을 224x224 로
# 늘려 왜곡하므로, 토큰 수를 맞추는 쪽을 택한다.
#   pv0     세 뷰를 각각 224x224 로 만들어 세로로 쌓는다 -> 672x224.
#           뷰당 해상도와 토큰 수(147)가 pv0 와 정확히 같고 공백 사분면도 없다.
#           RoboCasa 쪽이 쓰는 1x3 배치와 같은 형태다.
_CANVAS = os.environ.get("ALOHA_CANVAS", "native").strip().lower()
if _CANVAS == "pv0":
    _CANVAS_HW = None          # 조립 단계에서 뷰별로 처리한다
    _PER_VIEW = 224
elif _CANVAS != "native":
    try:
        _CANVAS_HW = tuple(int(x) for x in _CANVAS.split("x"))
        assert len(_CANVAS_HW) == 2 and all(v > 0 and v % 32 == 0 for v in _CANVAS_HW)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f'ALOHA_CANVAS must be "native" or "HxW" with H,W multiples of 32, got {_CANVAS!r}'
        ) from e
    _PER_VIEW = None
else:
    _CANVAS_HW = None
    _PER_VIEW = None

_CAPTION_STYLE = os.environ.get("ALOHA_CAPTION", "layout").strip().lower()
if _CAPTION_STYLE not in ("layout", "plain"):
    raise ValueError(f"ALOHA_CAPTION must be layout|plain, got {_CAPTION_STYLE!r}")

CAPTION_PREFIX = (
    f"Top: {_SCENE_CAM} third-person camera. Bottom-left: left wrist camera. "
    "Bottom-right: right wrist camera. Task: "
) if _CAPTION_STYLE == "layout" else ""


class _Episode:
    __slots__ = ("path", "task", "num_steps")

    def __init__(self, path: str, task: str, num_steps: int) -> None:
        self.path, self.task, self.num_steps = path, task, num_steps


class AlohaTabletopDataset(Dataset):
    """Sliding-window reader over ``tabletop-simulation-hdf5/<task>/episode_*.hdf5``."""

    def __init__(
        self,
        root: str,
        fps: float = 20.0,
        chunk_length: int = 32,
        mode: str = "wam",
        cameras: tuple[str, ...] | list[str] = DEFAULT_CAMERAS,
        viewpoint: str = "third_person_view",
        action_normalization: str | None = "minmax",
        normalization_path: str | None = None,
        task_split: list[str] | tuple[str, ...] | None = None,
        pad_tail: bool = True,
        sample_stride: int = 1,
        domain_name: str = "aloha-tabletop",
        index_cache_path: str | None = None,
        use_state: bool = True,
    ) -> None:
        super().__init__()
        self._root = Path(root)
        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._mode = str(mode)
        self._cameras = tuple(cameras)
        self._viewpoint = str(viewpoint)
        self._action_normalization = action_normalization
        self._pad_tail = bool(pad_tail)
        self._sample_stride = max(1, int(sample_stride))
        self._domain_name = str(domain_name)
        self._use_state = bool(use_state)
        self._domain_id = get_domain_id(self._domain_name)
        self._action_dim = get_action_dim(self._domain_name)
        self._norm_path = normalization_path or str(
            Path(__file__).parent.parent / "normalizer_stats" / "aloha_tabletop_stats.json"
        )
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._handles: dict[str, h5py.File] = {}

        keep = set(task_split) if task_split else None
        eps: list[_Episode] = []
        for task_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
            if keep is not None and task_dir.name not in keep:
                continue
            for f in sorted(task_dir.glob("episode_*.hdf5")):
                with h5py.File(f, "r") as h:
                    n = int(h[_ACTION_KEY].shape[0])
                eps.append(_Episode(str(f), task_dir.name, n))
        if not eps:
            raise ValueError(f"No episodes under {root} (task_split={task_split}).")
        self._episodes = eps

        # One window per start frame. ``pad_tail`` keeps the windows that run off the end
        # (edge-padded), which is where task completion and gripper release live.
        span = self._chunk_length + 1
        counts = [
            max(1, (e.num_steps + self._sample_stride - 1) // self._sample_stride)
            if self._pad_tail
            else max(0, (e.num_steps - span) // self._sample_stride + 1)
            for e in eps
        ]
        self._win_counts = counts
        self._win_cum = np.cumsum(counts)
        self._length = int(self._win_cum[-1])

    def __len__(self) -> int:
        return self._length

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """``(start, length)`` per episode over the flat window index.

        ``ActionIterableShuffleDataset`` shuffles these blocks and streams windows
        sequentially inside each one, which keeps the HDF5 reads sequential -- a window
        is ~30 MB of raw frames here, so random access across episodes would dominate.
        """
        starts = np.concatenate([[0], self._win_cum[:-1]]).astype(np.int64)
        return [(int(st), int(n)) for st, n in zip(starts, self._win_counts) if n > 0]

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _h5(self, path: str) -> h5py.File:
        # Worker-local handle cache: reopening a file per window dominates the read.
        h = self._handles.get(path)
        if h is None:
            h = self._handles[path] = h5py.File(path, "r")
        return h

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            self._norm_stats = {
                k: torch.from_numpy(v).float() for k, v in load_action_stats(self._norm_path).items()
            }
        return self._norm_stats

    def _locate(self, idx: int) -> tuple[_Episode, int]:
        ep_i = int(np.searchsorted(self._win_cum, idx, side="right"))
        prev = int(self._win_cum[ep_i - 1]) if ep_i > 0 else 0
        return self._episodes[ep_i], (idx - prev) * self._sample_stride

    @staticmethod
    def _read_span(dset: h5py.Dataset, t0: int, length: int, num_steps: int) -> np.ndarray:
        hi = min(t0 + length, num_steps)
        arr = np.asarray(dset[t0:hi])
        missing = length - (hi - t0)
        if missing > 0:
            arr = np.concatenate([arr, np.repeat(arr[-1:], missing, axis=0)], axis=0)
        return arr

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep, t0 = self._locate(int(idx) % self._length)
        h = self._h5(ep.path)
        window = self._chunk_length + 1

        views = []
        for cam in self._cameras:
            v = self._read_span(h[cam], t0, window, ep.num_steps)  # (T, H, W, 3)
            views.append(v)
        # Scene view on top, the remaining views tiled in a row beneath it (see the layout
        # note by DEFAULT_CAMERAS). The wrists are exactly half the scene view's width, so
        # two of them fill the row and nothing needs resampling.
        if _PER_VIEW is not None:
            # pv0 정합. 저쪽은 뷰마다 독립 latent 슬롯(224x224)을 주지만 우리 파이프라인은
            # 캔버스 한 장을 쓴다. 뷰당 픽셀을 224x224 로 맞추고 세로로 쌓으면
            # 672x224 = 150,528 px = 147 토큰/프레임으로 pv0 과 정확히 같아진다
            # (공백 사분면도 없다). RoboCasa 쪽이 쓰는 1x3 배치와 같은 형태다.
            # cv2.INTER_AREA 를 쓴다. 이유가 둘이다:
            #   (1) pv0 변환기(convert_tabletop_to_egoverse57.py)가 쓰는 필터와 같다.
            #   (2) eval 클라이언트는 Tabletop-Sim venv 에서 도는데 거기엔 torch 가 없다.
            #       같은 cv2 호출을 쓰면 학습과 eval 의 리사이즈가 비트 단위로 같아진다.
            import cv2

            sq = []
            for v in views:                                   # v: (T,H,W,3) uint8
                sq.append(np.stack([
                    cv2.resize(f, (_PER_VIEW, _PER_VIEW), interpolation=cv2.INTER_AREA)
                    for f in v
                ], axis=0))
            # 아래 조립 코드가 scene 위 + 나머지를 가로로 붙이는 구조라, pv0 에서는
            # 미리 세로로 합쳐 한 덩어리(scene)로 만들어 그 경로를 우회한다.
            views = [np.concatenate(sq, axis=1)]             # axis 1 = height

        scene, rest = views[0], views[1:]
        row = np.concatenate(rest, axis=2) if rest else None  # axis 2 = width
        if row is not None and row.shape[2] != scene.shape[2]:
            # Fallback for a non-default camera set whose row does not line up with the
            # scene width. Resample the ROW, not the scene, so the scene stays native.
            t = torch.from_numpy(np.ascontiguousarray(row)).permute(0, 3, 1, 2).float()
            scale = scene.shape[2] / row.shape[2]
            t = torch.nn.functional.interpolate(
                t,
                size=(max(1, int(round(row.shape[1] * scale))), scene.shape[2]),
                mode="bilinear",
                align_corners=False,
            )
            row = t.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        video_thwc = scene if row is None else np.concatenate([scene, row], axis=1)
        video = torch.from_numpy(np.ascontiguousarray(video_thwc)).permute(3, 0, 1, 2)  # (C,T,H,W) uint8
        if _CANVAS_HW is not None and video.shape[-2:] != _CANVAS_HW:
            # pv0 와 토큰 수를 맞추려고 캔버스를 줄인다. antialias=True 는 cv2 의
            # INTER_AREA(= pv0 변환기가 쓰는 것)와 같은 계열의 다운샘플이다.
            video = torch.nn.functional.interpolate(
                video.float(), size=_CANVAS_HW, mode="bilinear",
                align_corners=False, antialias=True,
            ).clamp(0, 255).to(torch.uint8)

        action = torch.from_numpy(
            self._read_span(h[_ACTION_KEY], t0, self._chunk_length, ep.num_steps).astype(np.float32)
        )
        if self._action_normalization is not None:
            action = normalize_action(action, self._action_normalization, self._load_norm_stats())

        if self._use_state:
            # 현재 절대 proprio 를 action[0] 으로 앞에 붙인다. 액션 길이가 chunk+1 이 되면
            # build_sequence_plan_from_mode 가 condition_frame_indexes_action=[0] 을 자동으로
            # 세워 그 행을 '노이즈 없는 조건'으로 다룬다 -- RoboCasa/DROID 와 같은 경로다.
            #
            # RoboCasa 는 액션이 delta 이고 state 는 절대값이라 robocasa_state 라는 별도
            # 통계 블록이 필요했지만, ALOHA 는 actions/ee_6d_pos 와
            # observations/states/ee_6d_pos 가 **같은 절대 EE 포즈 공간**(20D)이므로
            # 액션과 동일한 통계를 그대로 쓴다.
            state = torch.from_numpy(
                np.asarray(h[_STATE_KEY][min(t0, ep.num_steps - 1)], dtype=np.float32)
            )
            if self._action_normalization is not None:
                state = normalize_action(state, self._action_normalization, self._load_norm_stats())
            action = torch.cat([state.unsqueeze(0), action], dim=0)  # (chunk+1, 20)

        lang = h[_LANG_KEY][min(t0, ep.num_steps - 1)]
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8")
        task = str(lang) if lang else ep.task.replace("_", " ")

        return {
            "ai_caption": CAPTION_PREFIX + task,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": self._mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(0, dtype=torch.long),
        }

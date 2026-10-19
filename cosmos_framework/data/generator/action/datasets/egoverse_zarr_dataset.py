# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""EgoVerse zarr window reader + retrieval-paired dataset for conditioning policies.

EgoVerse ships per-episode zarr stores (raw frames as JPEG bytes + egomimic-
processed unified ``action{D}/action`` and ``state{D}/state`` arrays) plus a
retrieval index (``egoverse_pairs_v7/{windows,pairs}.npz``). Each **window** is a
fixed 150-frame clip ``(ep, w0, w1, emb, cls, keep)``; ``pairs.npz`` maps every
query window to its top-``K`` retrieved condition windows (same task ``cls``,
often a *different* embodiment — the cross-embodiment one-shot signal).

This module provides:

* :class:`EgoVerseWindowReader` — reads ONE window into a ``(video, action)``
  sample. The training unit is the leading ``chunk_length+1`` frames (≈1s) of the
  150-frame window. Video frames are JPEG-decoded (multi-cam vertically
  concatenated, DROID/LIBERO ``concat_view`` style). Actions (per-step deltas)
  are normalized with the per-embodiment ``"<emb>"`` stats; with ``use_state``
  the current absolute (waist-frame) state is normalized with the separate
  ``"<emb>_state"`` stats and prepended as the conditioning ``action[0]`` — the
  LIBERO/DROID framing (see ``libero_lerobot_dataset.py``).
* :class:`EgoVersePairedDataset` — for each query, samples one valid condition
  (online sample-1-of-K, skipping the ``-1`` no-match slots) and returns the
  ``{"query": ..., "condition": ...}`` raw pair. The sequence-plan / packing
  assembly (multi vision item + multi-domain action segment) is the next layer.

Embodiment registry: human rigs ``MECKA``/``scale``/``aria`` live under
``egoverse/`` with 57D action/state and a single ``images.front_1`` cam (domain
``hand_pose``); the ``eva`` bimanual robot lives under ``egoverse_eva/`` with 20D
action/state and three cams (domain ``eva``). ``max_action_dim`` padding to a
common width is handled downstream by the action transform, as for LIBERO.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.action_processing import pad_action_to_max_dim
from cosmos_framework.data.generator.action.transforms import (
    find_closest_target_size,
    reflection_pad_to_target,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan

# 8 EgoVerse task classes (windows.npz ``cls``) → a short instruction. Matched
# query/condition windows always share ``cls``, so this also names the demo.
_CLS_TO_PROMPT: dict[str, str] = {
    "put_in_container": "put the object in the container",
    "pick_place": "pick up the object and place it",
    "wipe": "wipe the surface",
    "dishes": "handle the dishes",
    "sort": "sort the objects",
    "pour": "pour from the container",
    "fold": "fold the cloth",
    "other": "complete the manipulation task",
}

# GR1 task tags → detailed language (the GR1 zarr only stores the terse task name;
# phrasings match the play subgoal `lang` so query/demo text live in one space).
_GR1_TASK_LANG: dict[str, str] = {
    "CloseLeftCabinetDoor": "close the left cabinet door",
    "PnPSinkToCabinet": "pick the object from the sink and place it in the cabinet",
    "PnPSinkToRightCounterPlate": "pick the object from the sink and place it on the plate",
    "TurnOnFaucet": "turn on the sink faucet",
}


def _resolve_instruction(attrs: dict, w0: int, n_frames: int, base_task: str) -> str:
    """Per-window language instruction.

    * mimicdroid play: the ``subgoals`` attr ([{start,end,lang,...}]) — pick the
      subgoal covering the window midpoint and use its ``lang`` (not the "play" tag).
    * GR1: map the terse task tag to a detailed phrase.
    * human rigs: ``task_description`` is already a detailed NL sentence — keep it.
    """
    sg = attrs.get("subgoals")
    if sg:
        segs = json.loads(sg) if isinstance(sg, str) else sg
        mid = int(w0) + n_frames // 2
        for s in segs:
            if int(s.get("start", -1)) <= mid <= int(s.get("end", -1)):
                return str(s.get("lang") or base_task)
        return base_task
    return _GR1_TASK_LANG.get(base_task, base_task)


@dataclass(frozen=True)
class EgoVerseEmbSpec:
    """Per-embodiment data layout (raw root, dims, cameras, action domain)."""

    root: str                       # raw-data subdir under data_root
    dim: int                        # action/state channel count
    domain: str                     # domain_utils key for action2llm routing
    cams: tuple[str, ...] = ("images.front_1",)
    fps: float = 30.0               # native capture rate (drives PE fps-modulation + per-emb frame count)


EMB_REGISTRY: dict[str, EgoVerseEmbSpec] = {
    "MECKA": EgoVerseEmbSpec(root="egoverse", dim=57, domain="hand_pose"),
    "scale": EgoVerseEmbSpec(root="egoverse", dim=57, domain="hand_pose"),
    "aria": EgoVerseEmbSpec(root="egoverse", dim=57, domain="hand_pose"),
    # robocasa/MimicDroid demos converted to the egomimic action57 format, stored
    # under egoverse/ (e.g. 003__SinkEnvForPlay_D_demo_1[__2x]); same 57-dim
    # hand_pose head as the human rigs, single front cam. Captured at 20 Hz (not
    # 30) → a 5 s clip is 100 frames, so it gets its own frame count + fps so the
    # PE temporal modulation and the VAE 4n+1 length stay correct.
    "mimicdroid": EgoVerseEmbSpec(root="egoverse", dim=57, domain="hand_pose", fps=20.0),
    # GR1 humanoid robot (MimicDroidDataset/GR1 → egoverse action57 zarr, 20 Hz).
    # Same 57-dim hand_pose head as the human rigs/mimicdroid (robot fingers map to
    # the predefined open/closed hand15); used as the post-training target embodiment,
    # conditioned on SinkEnvForPlay play demos (emb=mimicdroid).
    "gr1": EgoVerseEmbSpec(root="egoverse", dim=57, domain="hand_pose", fps=20.0),
    # EgoDex (Apple Vision Pro human hands) — held-out embodiment for OFFLINE eval only.
    # Same 57-dim hand_pose convention; zarr under dataset/egodex_egoverse (30 Hz).
    "egodex": EgoVerseEmbSpec(root="egodex_egoverse", dim=57, domain="hand_pose", fps=30.0),
    # Humanoid-Everyday (HE) held-out embodiment. zarr 레이아웃은 egodex 와 동일하다:
    # state57/action57 + images.front_1, 57-dim hand_pose, 30 fps. 등록이 없으면
    # EMB_REGISTRY[emb] 에서 KeyError 가 나 전 행이 실패한다.
    "he": EgoVerseEmbSpec(root="egoverse_he", dim=57, domain="hand_pose", fps=30.0),
    "eva": EgoVerseEmbSpec(
        root="egoverse_eva",
        dim=20,
        domain="eva",
        cams=("images.front_1", "images.left_wrist", "images.right_wrist"),
    ),
}


# --- action57 -> action20 (docs/action57_state57_convention.md §3f, §6) -------
#
# The 20-D contract is ``[Leef9 | Lgrip | Reef9 | Rgrip]`` and ``eef9`` is
# ``[Δpos(3), R[:,0](3), R[:,1](3)]``, which is channel-for-channel the layout
# ``robomind-franka-dual`` (domain 12) was trained on:
# ``[Lpos3, Lrot6d6, Lgrip, Rpos3, Rrot6d6, Rgrip]``. Both use rot6d = the first two
# columns of the rotation matrix (identity ``[1,0,0,0,1,0]``, matching ``IDENT9``) and
# both are body-frame frame-to-frame deltas (``backward_framewise`` is
# ``inv(P_t) @ P_{t+1}``), so no channel remap or frame change is needed.
#
# Why domain 12 rather than 16: 12 is already trained in the public Edge base (20
# non-zero bias entries, exactly its width), and 16 is not (bias identically 0). Sending
# this view to 12 makes it replay against forgetting rather than a fresh row to fill.
#
# The two grippers are the only nonlinear step. ``hand15`` is written as the lerp
# ``(1-g)·OPEN15 + g·CLOSED15``, so ``g`` inverts exactly by projecting onto the
# open→closed axis -- but on the RAW hand15, not the normalized one, which is why the
# caller carries both. ``2g-1`` then matches domain 12's own normalization: its stats
# put the gripper at ``q01=0.0, q99=1.0``, and an affine normalizer maps that to
# ``(g-0.5)/0.5``.
_OPEN15 = torch.tensor(
    [0.0005, 0.0958, -0.034, 0.0285, 0.0222, -0.1263, 0.0085, 0.0217, -0.1285,
     -0.0116, 0.0222, -0.1282, -0.0314, 0.0229, -0.1259], dtype=torch.float32)
_CLOSED15 = torch.tensor(
    [0.0052, 0.0775, -0.0701, 0.0266, 0.044, -0.0916, 0.0085, 0.0441, -0.1016,
     -0.0097, 0.044, -0.0937, -0.0278, 0.044, -0.0919], dtype=torch.float32)
_LEEF, _LHAND, _REEF, _RHAND = 9, 18, 33, 42
VIEW20_DIM = 20


def _grip_from_hand15(h15: torch.Tensor) -> torch.Tensor:
    """Invert the predefined-pose lerp ``hand15`` -> grasp scalar ``g`` in ``[0, 1]``."""
    axis = (_CLOSED15 - _OPEN15).to(h15.dtype)
    return ((h15 - _OPEN15.to(h15.dtype)) @ axis / (axis @ axis)).clamp(0.0, 1.0)


def action57_to_20(raw57: torch.Tensor, norm57: torch.Tensor) -> torch.Tensor:
    """``(..., 57) -> (..., 20)``: normalized eef blocks, grippers from the raw rows."""
    gl = _grip_from_hand15(raw57[..., _LHAND : _LHAND + 15]).unsqueeze(-1)
    gr = _grip_from_hand15(raw57[..., _RHAND : _RHAND + 15]).unsqueeze(-1)
    return torch.cat(
        [norm57[..., _LEEF : _LEEF + 9], 2.0 * gl - 1.0,
         norm57[..., _REEF : _REEF + 9], 2.0 * gr - 1.0], dim=-1)


@lru_cache(maxsize=4096)
def _open_array(path: str) -> zarr.Array:
    """Cache zarr array handles per worker process (opening is not free)."""
    return zarr.open(path, mode="r")


@lru_cache(maxsize=16384)
def _episode_attrs(ep_dir: str) -> tuple[tuple[str, Any], ...]:
    """Episode root zarr.json ``attributes`` (task_description, eef_speed_*, ...)."""
    try:
        attrs = json.loads((Path(ep_dir) / "zarr.json").read_text()).get("attributes", {})
        return tuple(sorted(attrs.items()))
    except Exception:  # noqa: BLE001 — missing/unreadable attrs → empty
        return ()


def _window_speed(ep_dir: str, w0: int, span: int) -> float | None:
    """Overall window speed = median over frames of the per-arm-mean eef speed."""
    try:
        arr = np.asarray(_open_array(f"{ep_dir}/meta/eef_speed")[w0 : w0 + span])
        return float(np.median(arr.mean(axis=1))) if arr.size else None
    except Exception:  # noqa: BLE001 — missing speed meta → None
        return None


def _decode_jpeg(buf: bytes) -> np.ndarray:
    """JPEG bytes → ``(H, W, 3)`` uint8 (RGB)."""
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))


def _read_frames(d: str, cam: str, start: int, stop: int) -> np.ndarray:
    """Decode ``[start, stop)`` frames of one camera → ``(T, H, W, 3)`` uint8.

    NOTE: index the zarr vlen-bytes array by SLICE (``arr[start:stop]``) — a
    single-element index returns a 0-d object array whose ``.item()`` is broken.
    """
    arr = _open_array(f"{d}/{cam}")
    raw = arr[start:stop]  # 1-d object array of bytes
    return np.stack([_decode_jpeg(bytes(raw[k])) for k in range(len(raw))], axis=0)


class EgoVerseWindowReader:
    """Reads one EgoVerse window into a ``(video, action)`` policy sample."""

    def __init__(
        self,
        *,
        data_root: str,
        pairs_dir: str,
        normalization_path: str,
        chunk_length: int = 28,           # 28 deltas → 29 video frames (≡1 mod 4, ≈0.97s @30fps)
        action_normalization: str = "quantile",
        action_source: str = "delta",     # "delta"=action57 diffs | "abs"=abs_action57 cmd rows | "state"=absolute NEXT-state57 chunk
        use_state: bool = False,          # state dropped entirely (v1): vision + action(deltas) only
        fps: float = 30.0,
        speed_stats_path: str | None = None,  # per-emb {lo,hi} for 0-1 speed conditioning; None = no speed
        cams_override: dict[str, tuple[str, ...]] | None = None,  # per-emb cam list (else EMB_REGISTRY default)
        view20_embs: tuple[str, ...] = (),     # embs emitted as the 20-D view instead of 57-D
        view20_domain: str = "robomind-franka-dual",   # domain 12 — trained in the base
    ) -> None:
        self._data_root = Path(data_root)
        self._chunk_length = int(chunk_length)
        self._action_normalization = action_normalization
        if action_source not in ("delta", "abs", "state"):
            raise ValueError(f"action_source must be 'delta', 'abs' or 'state', got {action_source!r}")
        # "abs" (v6 GR1 post recipe): rows are the recorded ABSOLUTE command chunk
        # (``abs_action57``: act[t] → s[t+1]) — same row indexing/count as delta,
        # different array + stats block (``<emb>_abs``).
        # "state" (2.5 SMID / state-chunk pivot, mid recipe): rows are the ABSOLUTE
        # NEXT-state chunk ``state57[w0+1 : w0+chunk+1]`` (row t = s[t+1], the
        # state the delta/command at t would produce) — available for EVERY
        # embodiment incl. human rigs (which have no commands), normalized with
        # the SAME ``<emb>_state`` stats as the use_state conditioning row.
        self._action_source = action_source
        self._use_state = bool(use_state)
        self._fps = float(fps)
        # Per-emb camera override: lets the GR1 post-train read 3 agentviews
        # (front_1 + sideleft + sideright, vertical-concat'd by the reader) while the
        # egoverse pre-train reads only the center front_1 from the SAME zarrs.
        self._cams_override = dict(cams_override) if cams_override else {}
        self._view20_embs = frozenset(view20_embs)
        self._view20_domain = str(view20_domain)
        self._speed_stats: dict[str, dict[str, float]] | None = (
            json.loads(Path(speed_stats_path).read_text()) if speed_stats_path else None
        )

        w = np.load(Path(pairs_dir) / "windows.npz", allow_pickle=True)
        self._ep = w["ep"]
        self._w0 = w["w0"].astype(np.int64)
        self._w1 = w["w1"].astype(np.int64)   # window end (builder-guaranteed valid span = w1-w0)
        self._emb = w["emb"]
        self._cls = w["cls"]

        # Pre-load per-embodiment action + state stats as torch tensors.
        # Always load the plain delta block AND ``<emb>_state`` (delta is needed
        # even in state mode: the paired DEMO segment conditions on delta rows,
        # the 2.5 ret_action contract); abs mode additionally loads ``<emb>_abs``.
        self._stats: dict[str, dict[str, torch.Tensor]] = {}
        _suffixes = ("", "_state") + (("_abs",) if self._action_source == "abs" else ())
        for emb in set(self._emb.tolist()):
            for suffix in _suffixes:
                key = f"{emb}{suffix}"
                st = {
                    k: torch.from_numpy(v).float()
                    for k, v in load_action_stats(normalization_path, stats_key=key).items()
                }
                # Degenerate-dim guard (2.5 action57_stats와 동일: "hi = lo + 1"):
                # 사실상 상수인 차원(q99-q01 or max-min < 1e-3)은 분모를 1로 강제.
                # 근거 사례: MECKA head9 회전 성분(q99-q01=6.8e-5)에 구버전 zero-head
                # 에피소드가 섞이면 (0-0.9999)/6.8e-5 ≈ -29,411 → FM loss ~1.67e7
                # 스파이크 (2026-08-06 v10 Edge mid에서 실측·규명).
                for lo_k, hi_k in (("q01", "q99"), ("min", "max")):
                    if lo_k in st and hi_k in st:
                        deg = (st[hi_k] - st[lo_k]) < 1e-3
                        if bool(deg.any()):
                            st[hi_k] = torch.where(deg, st[lo_k] + 1.0, st[hi_k])
                self._stats[key] = st

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def emb_of(self, widx: int) -> str:
        return str(self._emb[int(widx)])

    def fps_of(self, widx: int) -> float:
        """Native capture fps of window ``widx`` (20 for mimicdroid, else 30)."""
        return float(EMB_REGISTRY[str(self._emb[int(widx)])].fps)

    def _window_frames(self, widx: int) -> tuple[int, int, float]:
        """Per-window (target_frames, action_chunk, fps).

        ``self._chunk_length`` is the *reference* (30 fps) clip length, so it
        defines a wall-clock duration ``(chunk_length+1)/self._fps``. Each window
        is read at its own native fps for the same duration, then floored to the
        nearest VAE-legal ``4n+1`` length and capped by the builder-guaranteed
        window span. This keeps every clip the same real-time length (so the PE
        fps-modulation aligns demo↔query) while honoring each rig's frame rate
        (30 fps → 149 frames @5 s; 20 fps mimicdroid → 97 frames @5 s).
        """
        widx = int(widx)
        return self._frames_from_span(str(self._emb[widx]), int(self._w1[widx] - self._w0[widx]))

    def _frames_from_span(self, emb: str, span: int) -> tuple[int, int, float]:
        """(target_frames, action_chunk, fps) for an ``emb`` clip of ``span`` raw frames."""
        emb_fps = float(EMB_REGISTRY[emb].fps)
        duration_s = (self._chunk_length + 1) / self._fps
        scaled = int(round(duration_s * emb_fps))
        tgt = min(scaled, int(span))
        tgt = tgt - ((tgt - 1) % 4)            # floor to 4n+1 (VAE temporal requirement)
        return tgt, tgt - 1, emb_fps

    def _normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if self._action_normalization is None:
            return x
        # Hard clamp after normalization: legit per-emb deltas/states live in
        # |x| ≲ 5; anything beyond is data corruption (scale hand-glitch bursts
        # ~40-90, zero-head9 segments ~29k vs tight quantiles → FM loss up to
        # 8.5e9, 2026-08-06 실측). The stats-side degenerate guard + droplist
        # cannot enumerate every defect variant (partial in-episode bursts), so
        # bound the target space itself — corrupt rows become bounded noise the
        # optimizer shrugs off instead of loss bombs.
        return normalize_action(x, self._action_normalization, self._stats[key]).clamp_(-10.0, 10.0)

    def _speed01(self, emb: str, raw_speed: float | None) -> float | None:
        """Map raw window eef-speed → per-emb [0,1] control value."""
        if self._speed_stats is None or raw_speed is None or emb not in self._speed_stats:
            return None
        lo, hi = self._speed_stats[emb]["lo"], self._speed_stats[emb]["hi"]
        return float(np.clip((raw_speed - lo) / max(hi - lo, 1e-8), 0.0, 1.0))

    def view20_eligible(self, widx: int) -> bool:
        """Is window ``widx``'s embodiment one this reader may emit as the 20-D view?"""
        return str(self._emb[int(widx)]) in self._view20_embs

    def read_window(
        self,
        widx: int,
        speed_override: float | None = None,
        frame_offset: int = 0,
        action_source: str | None = None,
        use_state: bool | None = None,
        view20: bool | None = None,
    ) -> dict[str, Any]:
        """Read window ``widx`` → raw policy sample dict.

        Returns ``video`` ``(C, T, H, W)`` uint8, ``action`` ``(chunk+1, dim)``
        (when ``use_state``; else ``(chunk, dim)``) normalized, plus ``domain_id``,
        ``emb``, ``ai_caption`` (task_description + normalized speed), ``speed``
        (0–1 control; ``speed_override`` forces it for inference), ``conditioning_fps``.

        ``frame_offset`` (δ ≥ 0) shifts the read start forward by δ raw frames within
        the episode (video, action deltas AND state0 all advance in lockstep), without
        changing the clip length: the per-window ``(target_frames, chunk)`` come from
        the stored span ``w1-w0`` and are δ-invariant. Used by the GR1 post-train
        per-frame coverage (sub-stride offsets); ``frame_offset=0`` (default) reproduces
        the original read exactly, so all other paths are unaffected. The CALLER is
        responsible for clamping δ in-bounds (see :meth:`max_frame_offset`) — this keeps
        a paired query/demo on the SAME δ rather than each silently clipping to its own.
        """
        widx = int(widx)
        emb = str(self._emb[widx])
        spec = EMB_REGISTRY[emb]
        ep = str(self._ep[widx])
        w0 = int(self._w0[widx]) + max(0, int(frame_offset))
        d = str(self._data_root / spec.root / ep)

        target_frames, chunk, emb_fps = self._window_frames(widx)
        # Video: decode each cam, vertically concat (front on top), → (C, T, H, W).
        cams = self._cams_override.get(emb, spec.cams)
        views = [_read_frames(d, cam, w0, w0 + target_frames) for cam in cams]
        video_thwc = np.concatenate(views, axis=1) if len(views) > 1 else views[0]  # (T, H*ncam, W, 3)
        video = torch.from_numpy(video_thwc).permute(3, 0, 1, 2).contiguous()       # (C, T, H, W) uint8

        # Action rows over the chunk: measured-state deltas ("delta"), absolute
        # command rows ("abs", v6 post recipe), or the absolute NEXT-state chunk
        # ("state", 2.5 SMID mid recipe: row t = s[t+1], same act[t]→s[t+1]
        # alignment as delta/abs — indexing shifts by +1 into the state array).
        # ``action_source``/``use_state`` args override the reader defaults for
        # THIS window — the paired dataset reads the DEMO segment with
        # ('delta', False) when the query target is 'state', matching the 2.5
        # contract (demo slot = ret_video + ret_ACTION; state is query-only).
        _src = self._action_source if action_source is None else action_source
        _use_st = self._use_state if use_state is None else bool(use_state)
        # The 20-D view needs the RAW rows as well: the grippers invert a lerp defined in
        # raw hand15 coordinates, while the eef blocks pass through already normalized.
        # Eligibility is per-emb; the CALLER decides per item, because a query and its
        # conditioning demo have to land on the same contract -- a 57-D demo in front of a
        # 20-D query is a silently wrong sample, not an error.
        _view20 = (emb in self._view20_embs) and (True if view20 is None else bool(view20))
        if _view20 and spec.dim != 57:
            raise ValueError(
                f"view20 requested for emb={emb!r} whose spec.dim is {spec.dim}, not 57. "
                "The projection is defined only on the action57 layout."
            )
        if _src == "abs":
            act_arr = _open_array(f"{d}/abs_action{spec.dim}/action")
            raw = torch.from_numpy(np.asarray(act_arr[w0 : w0 + chunk], dtype=np.float32))
            action = self._normalize(raw, f"{emb}_abs")  # (chunk, dim)
        elif _src == "state":
            act_arr = _open_array(f"{d}/state{spec.dim}/state")
            raw = torch.from_numpy(np.asarray(act_arr[w0 + 1 : w0 + 1 + chunk], dtype=np.float32))
            action = self._normalize(raw, f"{emb}_state")  # (chunk, dim)
        else:
            act_arr = _open_array(f"{d}/action{spec.dim}/action")
            raw = torch.from_numpy(np.asarray(act_arr[w0 : w0 + chunk], dtype=np.float32))
            action = self._normalize(raw, emb)  # (chunk, dim)

        if _use_st:
            sta_arr = _open_array(f"{d}/state{spec.dim}/state")
            state0_raw = torch.from_numpy(np.asarray(sta_arr[w0], dtype=np.float32)).unsqueeze(0)  # (1, dim)
            state0 = self._normalize(state0_raw, f"{emb}_state")
            action = torch.cat([state0, action], dim=0)  # (chunk+1, dim) — state0 = conditioning action[0]
            raw = torch.cat([state0_raw, raw], dim=0)

        # Text: prefer the episode's natural-language task_description, fall back to cls.
        attrs = dict(_episode_attrs(d))
        task = str(attrs.get("task_description") or _CLS_TO_PROMPT.get(str(self._cls[widx]), _CLS_TO_PROMPT["other"]))
        task = _resolve_instruction(attrs, w0, target_frames, task)  # subgoal (md) / detailed (gr1) / NL (human)

        # Speed: per-emb 0–1 control value (training = GT window speed; inference may override).
        speed = speed_override if speed_override is not None else self._speed01(emb, _window_speed(d, w0, target_frames))
        caption = (
            f"{emb}: {task.rstrip('. ')}. speed: {speed:.2f}" if speed is not None else f"{emb}: {task}"
        )

        if _view20:
            action = action57_to_20(raw, action)  # (chunk[+1], 20)

        return {
            "video": video,
            "action": action,
            "domain_id": torch.tensor(
                get_domain_id(self._view20_domain if _view20 else spec.domain), dtype=torch.long),
            "emb": emb,
            "task": task,
            "raw_action_dim": VIEW20_DIM if _view20 else spec.dim,
            "ai_caption": caption,
            "speed": (float(speed) if speed is not None else None),
            "conditioning_fps": torch.tensor(emb_fps, dtype=torch.long),  # 20 for mimicdroid → PE fps-modulation
        }

    def _episode_len(self, widx: int) -> int:
        """Number of raw frames (= ``action{dim}`` length) of window ``widx``'s episode."""
        widx = int(widx)
        spec = EMB_REGISTRY[str(self._emb[widx])]
        d = str(self._data_root / spec.root / str(self._ep[widx]))
        return int(_open_array(f"{d}/action{spec.dim}/action").shape[0])

    def max_frame_offset(self, widx: int, max_offset: int) -> int:
        """Largest δ in ``[0, max_offset]`` for which reading ``widx`` at start ``w0+δ``
        stays in-bounds, i.e. ``w0 + δ + target_frames <= episode_len`` (the video read
        is the longest reach; the ``chunk`` action read is one frame shorter). Returns 0
        when ``max_offset <= 0``. The window builder guarantees ``w0 + span <= len`` at
        δ=0, so for the GR1 stride-4 pairs (span=32, target≈29) δ up to 3 is always safe;
        this is the defensive clamp that also covers demos sitting at the episode tail."""
        if max_offset <= 0:
            return 0
        target_frames, _chunk, _fps = self._window_frames(widx)
        room = self._episode_len(widx) - int(self._w0[widx]) - int(target_frames)
        return max(0, min(int(max_offset), room))

    def read_state0(self, widx: int) -> np.ndarray:
        """Raw (un-normalized) absolute ``state{dim}`` at the window start, shape ``(dim,)``.

        Used by offline eval to anchor a cumulative-sum EEF trajectory at the real start
        pose (state57 layout = head9 | Labs9 | Lhand15 | Rabs9 | Rhand15; abs9 = [pos, R0, R1],
        so left-EEF start pos = ``[9:12]``, right-EEF = ``[33:36]``)."""
        widx = int(widx)
        spec = EMB_REGISTRY[str(self._emb[widx])]
        d = str(self._data_root / spec.root / str(self._ep[widx]))
        sta_arr = _open_array(f"{d}/state{spec.dim}/state")
        return np.asarray(sta_arr[int(self._w0[widx])], dtype=np.float32)

    def read_clip(self, ep: str, emb: str, w0: int, span: int = 100,
                  speed_override: float | None = None, frame_offset: int = 0) -> dict[str, Any]:
        """Read a clip at an EXPLICIT ``(ep, emb, w0)`` — same output as :meth:`read_window`
        but without a stored window index. Used for online eval, where the retrieved demo
        is an arbitrary play (ep, w0) rather than a pair-built window.

        ``frame_offset`` (δ ≥ 0) advances the read start by δ raw frames (lockstep with the
        paired query); the clip length is fixed by ``span`` and is δ-invariant. ``frame_offset=0``
        (default) reproduces the original read exactly. The caller clamps δ in-bounds."""
        spec = EMB_REGISTRY[emb]
        d = str(self._data_root / spec.root / ep)
        w0 = int(w0) + max(0, int(frame_offset))
        target_frames, chunk, emb_fps = self._frames_from_span(emb, span)
        cams = self._cams_override.get(emb, spec.cams)
        # abs/state modes read the same arrays + stats as read_window, keeping the
        # online-eval demo byte-identical to training.
        _act_off = 0
        if self._action_source == "abs":
            act_arr = _open_array(f"{d}/abs_action{spec.dim}/action")
            act_key = f"{emb}_abs"
        elif self._action_source == "state":
            act_arr = _open_array(f"{d}/state{spec.dim}/state")
            act_key = f"{emb}_state"
            _act_off = 1  # next-state chunk: row t = s[t+1]
        else:
            act_arr = _open_array(f"{d}/action{spec.dim}/action")
            act_key = emb
        # Clamp w0 so the video read [w0:w0+target_frames] AND the action read [w0:w0+chunk]
        # stay in-bounds. Short demos (e.g. taskdemos ~255 frames) otherwise return a truncated,
        # non-4n+1 video clip near the end (w0 close to len) that the VAE encoder rejects.
        n_vid = int(_open_array(f"{d}/{cams[0]}").shape[0])
        w0 = max(0, min(int(w0), n_vid - target_frames, int(act_arr.shape[0]) - chunk - _act_off))
        views = [_read_frames(d, cam, int(w0), int(w0) + target_frames) for cam in cams]
        video_thwc = np.concatenate(views, axis=1) if len(views) > 1 else views[0]
        video = torch.from_numpy(video_thwc).permute(3, 0, 1, 2).contiguous()

        raw = torch.from_numpy(
            np.asarray(act_arr[int(w0) + _act_off: int(w0) + _act_off + chunk], dtype=np.float32)
        )
        action = self._normalize(raw, act_key)
        if self._use_state:
            sta_arr = _open_array(f"{d}/state{spec.dim}/state")
            state0_raw = torch.from_numpy(np.asarray(sta_arr[int(w0)], dtype=np.float32)).unsqueeze(0)
            state0 = self._normalize(state0_raw, f"{emb}_state")
            action = torch.cat([state0, action], dim=0)
            raw = torch.cat([state0_raw, raw], dim=0)
        # Same contract as the training path: a run that trains this emb as 20-D must
        # also read its online-eval demos as 20-D, or the demo slot silently carries a
        # 57-D vector into a 20-D head.
        _view20 = emb in self._view20_embs
        if _view20:
            action = action57_to_20(raw, action)

        attrs = dict(_episode_attrs(d))
        task = str(attrs.get("task_description") or _CLS_TO_PROMPT["other"])
        task = _resolve_instruction(attrs, int(w0), target_frames, task)
        speed = speed_override if speed_override is not None else self._speed01(emb, _window_speed(d, int(w0), target_frames))
        caption = (
            f"{emb}: {task.rstrip('. ')}. speed: {speed:.2f}" if speed is not None else f"{emb}: {task}"
        )
        return {
            "video": video,
            "action": action,
            "domain_id": torch.tensor(
                get_domain_id(self._view20_domain if _view20 else spec.domain), dtype=torch.long),
            "emb": emb,
            "task": task,
            "raw_action_dim": VIEW20_DIM if _view20 else spec.dim,
            "ai_caption": caption,
            "speed": (float(speed) if speed is not None else None),
            "conditioning_fps": torch.tensor(emb_fps, dtype=torch.long),
        }


class EgoVersePairedResize:
    """Minimal spatial transform: aspect-preserving resize + reflection-pad each
    side's video to ``resolution``, recording its ``image_size`` ``(4,)``.

    Reuses the action pipeline's ``find_closest_target_size`` /
    ``reflection_pad_to_target`` so EgoVerse video is prepared exactly like the
    LIBERO/DROID recipes. Query and condition are resized independently (rig
    resolutions differ, e.g. 480×640 vs 360×640). Sequence-plan / tokenization /
    multi-segment packing are added in the next layer.
    """

    def __init__(self, resolution: str | int = "256", keep_aspect_ratio: bool = True) -> None:
        self._resolution = resolution
        self._keep_aspect_ratio = bool(keep_aspect_ratio)

    def _resize_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        video = sample["video"]  # (C, T, H, W) uint8
        h, w = int(video.shape[-2]), int(video.shape[-1])
        target_w, target_h = find_closest_target_size(h, w, self._resolution)
        d: dict[str, Any] = {"video": video}
        reflection_pad_to_target(d, ["video"], self._keep_aspect_ratio, target_w, target_h)
        sample["video"] = d["video"]
        sample["image_size"] = d["image_size"]  # [target_h, target_w, orig_h_resized, orig_w_resized]
        return sample

    def __call__(self, paired: dict[str, Any]) -> dict[str, Any]:
        paired["query"] = self._resize_one(paired["query"])
        paired["condition"] = self._resize_one(paired["condition"])
        return paired


class EgoVersePairedTransform:
    """Assemble a reader pair into a multi-item conditioning-policy training sample.

    Produces the data_batch sample consumed by ``get_data_and_condition`` /
    ``pack_input_sequence``, mirroring the image-editing multi-vision format and
    the video-transfer conditioning layout:

      * ``video`` = ``[demo_v, query_v]`` — two vision items. The packer marks all
        but the last (demo) fully clean; the last (query) uses
        ``condition_frame_indexes_vision=[0]`` (frame-0 observation clean, rest
        generated).
      * ``action`` = ``[demo_a, query_a]`` — two action segments (per-domain). Demo
        fully clean, query fully generated (handled by the multi-segment packer).
      * ``sequence_plan.share_vision_temporal_positions=share_temporal`` — when
        True, demo & query vision items AND action segments share one temporal
        mRoPE grid (transfer-style aligned conditioning).

    Shared-grid requires identical latent grids, so both videos are resized to the
    SAME canvas. With ``keep_aspect_ratio=False`` (default for shared mode) the
    content fills the canvas exactly (no padding → latent-padding removal is a
    no-op → grids match), at the cost of mild aspect distortion across rigs.

    The caption is the query instruction (task + target speed) with the demo's
    speed appended, so both query and demo speed condition the model.
    """

    def __init__(
        self,
        *,
        resolution: str | int = "256",
        max_action_dim: int = 64,
        share_temporal: bool = True,
        keep_aspect_ratio: bool = False,
        tokenizer_config: dict | None = None,
        cfg_dropout_rate: float = 0.1,
        text_token_key: str = "text_token_ids",
        plain: bool = False,
        dummy_ret_prob: float = 0.0,
        dummy_ret_frames: int = 1,
    ) -> None:
        self._resolution = resolution
        self._max_action_dim = int(max_action_dim)
        self._share_temporal = bool(share_temporal)
        self._keep_aspect_ratio = bool(keep_aspect_ratio)
        # BC ("plain") arm: emit the QUERY ONLY — no demo vision item, no demo
        # action segment, caption without the "demo <emb>" clause. Single-item
        # samples take the standard single-clip packing path
        # (num_action_items_per_sample == 1), so plain is layout-equivalent to
        # the LIBERO/DROID recipes while keeping the identical v6 reader/target.
        self._plain = bool(plain)
        # Fraction of paired samples whose DEMO slot is emptied: the two-item ICL layout
        # is kept exactly, and only the demo's content is replaced by zeros.
        #
        # Why this is a separate arm rather than `plain`. A mid-trained backbone is asked,
        # at RoboCasa post-train, to predict actions with NO demo in context, because the
        # post-train recipe is single-clip. ICL mid-training never shows it that situation:
        # every sample carries an informative demo, so the model is free to route
        # prediction through the demo and let the query pathway atrophy. `plain` removes
        # the slot entirely, which changes the packing layout too (one vision item, one
        # action segment) and so cannot isolate the effect. Emptying the slot keeps the
        # layout byte-identical and removes only the information, which is exactly the
        # condition post-train presents.
        #
        # The demo action segment is a CLEAN conditioning input, not a target, so zeroing
        # it costs no supervision -- the query's action loss is untouched.
        self._dummy_ret_prob = float(dummy_ret_prob)
        # Frames in the emptied demo clip. An informative demo has to be as long as the
        # query, but an empty one carries the same nothing at any length, and the VAE
        # encodes T=1 at latent_T=1 with no padding (1 + (T-1)//4, and T=1 is already of
        # the form 1 + 4k). A 1-frame dummy therefore costs one latent frame instead of
        # the query's 23, which roughly halves the vision tokens in the sample.
        #
        # The cost is that demo and query no longer share a temporal grid, so
        # ``share_vision_temporal_positions`` must be off for these samples -- that flag
        # exists to align a demo against a query frame-for-frame, which is meaningless
        # when the demo is blank. Set ``dummy_ret_frames=0`` to keep the query's length
        # and the shared grid.
        self._dummy_ret_frames = int(dummy_ret_frames)
        self._text_tokenizer = None
        if tokenizer_config is not None:
            from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransform

            self._text_tokenizer = TextTokenizerTransform(
                input_keys=["ai_caption"],
                output_keys=[text_token_key],
                args={"tokenizer_config": tokenizer_config, "cfg_dropout_rate": cfg_dropout_rate},
            )

    def _resize(self, video: torch.Tensor, target_w: int, target_h: int) -> tuple[torch.Tensor, torch.Tensor]:
        d: dict[str, Any] = {"video": video}
        reflection_pad_to_target(d, ["video"], self._keep_aspect_ratio, target_w, target_h)
        return d["video"], d["image_size"]

    def __call__(self, paired: dict[str, Any]) -> dict[str, Any]:
        if self._plain:
            return self._call_plain(paired["query"])
        q, c = paired["query"], paired["condition"]
        dummy_ret = self._dummy_ret_prob > 0.0 and random.random() < self._dummy_ret_prob

        # Shared canvas for both items (from the query's aspect) so latent grids match.
        qh, qw = int(q["video"].shape[-2]), int(q["video"].shape[-1])
        target_w, target_h = find_closest_target_size(qh, qw, self._resolution)
        q_v, q_is = self._resize(q["video"], target_w, target_h)
        c_v, c_is = (None, q_is) if dummy_ret else self._resize(c["video"], target_w, target_h)
        # Pre-normalize uint8 → float [-1, 1]. For multi-vision items the model
        # flattens each to (1,C,T,H,W) and, when float, sets is_preprocessed=True —
        # which skips _normalize_video_databatch_inplace's extra stack (that would
        # otherwise produce a 6-D tensor the VAE misreads as T=C=3). Resize stays on
        # uint8 (clamped) so BICUBIC ringing can't push values past [-1, 1].
        q_v = q_v.float().div_(127.5).sub_(1.0)
        # Build the dummy AFTER the query is normalized, and in the normalized space. Both
        # halves of the list must share a dtype: the model treats a list of float vision
        # items as already preprocessed and a list of uint8 items as raw, and a mixed list
        # takes the raw path and then asserts on the float item ("Video data is not in
        # uint8 format"). Zeros here are mid-gray; created as uint8 zeros and normalized
        # they would instead be a saturated -1.0.
        if dummy_ret:
            n = self._dummy_ret_frames
            c_v = torch.zeros_like(q_v) if n <= 0 else torch.zeros_like(q_v[:, :1]).repeat(1, n, 1, 1)
        else:
            c_v = c_v.float().div_(127.5).sub_(1.0)

        q_a = pad_action_to_max_dim(q["action"], self._max_action_dim)
        c_a = torch.zeros_like(q_a) if dummy_ret else pad_action_to_max_dim(c["action"], self._max_action_dim)

        # Caption format: "<instruction> demo <emb> [speed] target <emb> [speed]" —
        # the task instruction, then the demo (conditioning source) embodiment, then
        # the target (query) embodiment. Speed is appended only when available
        # (speed_stats_path set + emb in the JSON); with speed off (e.g. GR1
        # post-training) it is omitted, giving "<instruction> demo <emb> target <emb>".
        def _spd(s: float | None) -> str:
            return f" {s:.2f}" if s is not None else ""

        instruction = q.get("task") or q["ai_caption"]
        # Same grammar, with the demo clause naming the empty slot. Keeping the clause (as
        # opposed to dropping it, which is what `plain` does) means the text layout does
        # not change either -- only what the slot says is in it.
        demo_clause = "none" if dummy_ret else f"{c['emb']}{_spd(c.get('speed'))}"
        caption = f"{instruction} demo {demo_clause} target {q['emb']}{_spd(q.get('speed'))}"

        # Action↔vision alignment, derived per-sample from the query's action vs
        # video length (mirrors transforms.build_sequence_plan_from_mode, no history):
        #   * delta-only  (action_len == video_len - 1): action[0] = delta1 aligns
        #     with vision frame 1 → offset 1, no clean action row (all generated).
        #   * use_state   (action_len == video_len):     action[0] = state0 is a clean
        #     CONDITIONING row aligned with vision frame 0 → offset 0, condition [0].
        # This makes the paired transform handle use_state=True (GR1 post-train) and
        # use_state=False (EgoVerse pre-train) without a separate flag.
        q_vid_len, q_act_len = int(q_v.shape[1]), int(q_a.shape[0])
        if q_act_len == q_vid_len:  # state0 prepended → clean conditioning row
            cond_action, action_offset = [0], 0
        else:  # delta-only (q_act_len == q_vid_len - 1)
            cond_action, action_offset = [], 1

        sample: dict[str, Any] = {
            # Order is [demo (condition), query] — demo is the all-clean context;
            # query is the last item (frame-0 clean vision, generated action+video).
            "video": [c_v, q_v],
            "image_size": [c_is, q_is],
            "action": [c_a, q_a],
            # A dummy slot carries the QUERY's domain and width: the demo segment is an
            # encoder input, so an unrelated domain row would be exercised on all-zero
            # content for no reason.
            "domain_id": [q["domain_id"] if dummy_ret else c["domain_id"], q["domain_id"]],
            "raw_action_dim": [
                torch.tensor(q["raw_action_dim"] if dummy_ret else c["raw_action_dim"], dtype=torch.long),
                torch.tensor(q["raw_action_dim"], dtype=torch.long),
            ],
            "ai_caption": caption,
            # Per-item fps [demo, query] (NOT per-sample) — demo & query may differ
            # (30 Hz human demo, 20 Hz mimicdroid query). get_data_and_condition
            # flattens this across the batch to per-vision-item / per-action-segment
            # fps so the PE fps-modulation places each clip on its own real-time grid.
            "conditioning_fps": torch.stack(
                [q["conditioning_fps"] if dummy_ret else c["conditioning_fps"], q["conditioning_fps"]]
            ),
            "num_frames": int(q_v.shape[1]),
            "dataset_name": "egoverse_paired",
            "sequence_plan": SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=True,
                condition_frame_indexes_vision=[0],  # query frame-0 clean, rest generated
                condition_frame_indexes_action=cond_action,  # [0]=state0 clean (use_state) else []
                # A shortened dummy has a different latent length than the query, so the
                # shared grid cannot be built for it.
                share_vision_temporal_positions=(
                    self._share_temporal and not (dummy_ret and self._dummy_ret_frames > 0)
                ),
                action_start_frame_offset=action_offset,
            ),
        }
        if self._text_tokenizer is not None:
            sample = self._text_tokenizer(sample)
        return sample

    def _call_plain(self, q: dict[str, Any]) -> dict[str, Any]:
        """BC ("plain") sample: the query clip alone — identical reader output,
        normalization, caption instruction, and action/vision alignment as the
        paired path, minus every demo artifact. Single vision item + single action
        segment → the standard single-clip packing path (bit-identical to the
        LIBERO/DROID layout), so ICL-vs-BC differs ONLY in demo conditioning."""
        qh, qw = int(q["video"].shape[-2]), int(q["video"].shape[-1])
        target_w, target_h = find_closest_target_size(qh, qw, self._resolution)
        q_v, q_is = self._resize(q["video"], target_w, target_h)
        # Keep uint8: a single-item sample does NOT take the multi-vision flatten
        # (is_preprocessed) branch — _normalize_video_databatch_inplace's standard
        # path asserts uint8 and normalizes the list-of-1 itself.

        q_a = pad_action_to_max_dim(q["action"], self._max_action_dim)

        def _spd(s: float | None) -> str:
            return f" {s:.2f}" if s is not None else ""

        instruction = q.get("task") or q["ai_caption"]
        # Same caption grammar as the paired path with the "demo <emb>" clause removed.
        caption = f"{instruction} target {q['emb']}{_spd(q.get('speed'))}"

        q_vid_len, q_act_len = int(q_v.shape[1]), int(q_a.shape[0])
        if q_act_len == q_vid_len:  # state0 prepended → clean conditioning row
            cond_action, action_offset = [0], 0
        else:  # delta-only (q_act_len == q_vid_len - 1)
            cond_action, action_offset = [], 1

        # Single-clip contract (matches the LIBERO/PushT ActionTransformPipeline
        # output exactly): PLAIN tensors, no per-item lists — video [C,T,H,W]
        # uint8, action [T,D], 0-dim domain_id / raw_action_dim. The collate
        # wraps per sample; the model's standard dense path consumes them.
        sample: dict[str, Any] = {
            "video": q_v,
            "image_size": q_is,
            "action": q_a,
            "domain_id": q["domain_id"],
            "raw_action_dim": torch.tensor(q["raw_action_dim"], dtype=torch.long),
            "ai_caption": caption,
            # Single item → per-sample scalar fps (standard single-clip semantics).
            "conditioning_fps": q["conditioning_fps"],
            "num_frames": int(q_v.shape[1]),
            "dataset_name": "egoverse_plain",
            "sequence_plan": SequencePlan(
                has_text=True,
                has_vision=True,
                has_action=True,
                condition_frame_indexes_vision=[0],  # frame-0 observation clean, rest generated
                condition_frame_indexes_action=cond_action,  # [0]=state0 clean (use_state) else []
                share_vision_temporal_positions=False,  # single item: nothing to share
                action_start_frame_offset=action_offset,
            ),
        }
        if self._text_tokenizer is not None:
            sample = self._text_tokenizer(sample)
        return sample


class EgoVersePairedDataset(Dataset):
    """Retrieval-paired EgoVerse dataset: ``{"query", "condition"}`` per item.

    For each query window, samples ONE valid condition window online (uniform
    over the non-``-1`` slots of ``match_idx``; ``match_dist``-weighting is a
    future knob). The query is the generation target; the condition (demo) is
    fully clean context. Sequence-plan / packing assembly is the next layer.
    """

    def __init__(
        self,
        *,
        reader: EgoVerseWindowReader,
        pairs_dir: str,
        transform: EgoVersePairedResize | EgoVersePairedTransform | None = None,
        seed: int = 0,
        per_frame_stride: int = 1,
        plain: bool = False,
        droplist_path: str | None = None,
        view20_prob: float = 1.0,   # share of view20-eligible items emitted as 20-D
    ) -> None:
        self._reader = reader
        self._transform = transform
        # BC ("plain") arm: skip demo sampling/reading entirely — the transform
        # (plain=True) consumes only the query. Pair rows / shuffle / sharding /
        # per-frame stride are identical to the paired arm, so ICL and BC see the
        # SAME query distribution in the SAME order.
        self._plain = bool(plain)
        self._view20_prob = float(view20_prob)
        # Per-frame coverage knob (GR1 post-train). The stored query windows sit on a
        # STRIDE-frame grid (w0 = 0, STRIDE, 2*STRIDE, …); with ``per_frame_stride=STRIDE``
        # each __getitem__ samples a sub-stride offset δ ∈ {0..STRIDE-1} and advances BOTH
        # the query target AND its paired demo by the SAME δ (lockstep), so every frame
        # becomes a reachable training example across epochs without re-retrieving a demo
        # or inflating the index. ``per_frame_stride=1`` (default) ⇒ δ≡0 ⇒ original behavior,
        # so non-GR1 configs/embodiments are untouched.
        self._per_frame_stride = max(1, int(per_frame_stride))
        p = np.load(Path(pairs_dir) / "pairs.npz", allow_pickle=True)
        self._query_idx = p["query_idx"].astype(np.int64)
        self._match_idx = p["match_idx"].astype(np.int64)  # (N, K), -1 = no match
        self._rng = np.random.default_rng(seed)
        # Global shuffle of the pair rows. The pairs file stores queries GROUPED by
        # embodiment (long contiguous aria/scale/MECKA/mimicdroid blocks), and the
        # loader iterates indices sequentially with strided per-rank sharding and no
        # sampler shuffle — so without this each rank would march through one
        # embodiment/task block at a time. That makes the loss oscillate: it descends
        # while overfitting a homogeneous block, then spikes at every block boundary
        # (distribution shift), at the SAME deterministic iters every run. Permuting
        # the rows once (shared seed → identical order on all ranks, so the strided
        # shards stay a disjoint partition) interleaves embodiments/tasks into every
        # batch and removes the block-boundary spikes.
        # Defect-episode droplist: known-bad episodes excluded as BOTH queries and
        # demo candidates (e.g. v10의 zero-head9 MECKA 31 eps — head9가 IDENT9가
        # 아니라 0으로 기록된 구버전 변환본; MECKA stats로 정규화하면 수백~수만
        # 크기로 폭발해 FM loss 스파이크(1.67e7)를 만들었음, 2026-08-06 규명).
        if droplist_path:
            import json as _json

            _drop = set(_json.load(open(droplist_path))["episodes"])
            _w_drop = np.isin(reader._ep, list(_drop))
            if _w_drop.any():
                self._match_idx = self._match_idx.copy()
                self._match_idx[np.where(_w_drop[np.clip(self._match_idx, 0, None)] & (self._match_idx >= 0))] = -1
                _q_keep = ~_w_drop[self._query_idx]
                n_drop = int((~_q_keep).sum())
                self._query_idx = self._query_idx[_q_keep]
                self._match_idx = self._match_idx[_q_keep]
                print(
                    f"[EgoVersePairedDataset] droplist: {len(_drop)} defect eps -> "
                    f"dropped {n_drop} query rows + masked demo candidates",
                    flush=True,
                )
        # State-array availability guard: a few legacy episodes (e.g. 4 old-format
        # MECKA eps in v10) ship ``action57`` but NO ``state57`` — any recipe that
        # reads state (``use_state`` conditioning row and/or the "state" action
        # target) would crash a DataLoader worker mid-run (2026-08-05: killed the
        # first 4-node SMID launch at iter 7). Filter them out ONCE at init:
        # drop query rows on bad eps and mask bad demo candidates to -1 (the
        # per-row sampler then falls back to the remaining candidates / self).
        if reader._use_state or reader._action_source == "state":
            _eps = reader._ep
            _uniq, _uidx = np.unique(_eps, return_index=True)  # one representative window per ep
            _bad = set()
            for _e, _wi in zip(_uniq, _uidx):
                _spec = EMB_REGISTRY[str(reader._emb[int(_wi)])]
                if not (reader._data_root / _spec.root / str(_e) / f"state{_spec.dim}").is_dir():
                    _bad.add(str(_e))
            if _bad:
                # QUERIES only: demos never read state (2.5 contract above), so
                # state-less episodes remain perfectly valid demo candidates.
                _w_bad = np.isin(_eps, list(_bad))  # per-window bad flag
                _q_keep = ~_w_bad[self._query_idx]
                n_drop = int((~_q_keep).sum())
                self._query_idx = self._query_idx[_q_keep]
                self._match_idx = self._match_idx[_q_keep]
                print(
                    f"[EgoVersePairedDataset] state guard: {len(_bad)} eps lack state arrays "
                    f"({sorted(_bad)[:4]}) -> dropped {n_drop} query rows (demo candidates kept)",
                    flush=True,
                )
        self._order = np.random.default_rng(seed).permutation(self._query_idx.shape[0])
        # Set by RankPartitionedDataLoader after construction; default = single shard.
        # Each rank takes a disjoint strided slice (same scheme as ActionSFTDataset).
        self.shard_world_size: int = 1
        self.shard_rank: int = 0

    def __len__(self) -> int:
        return int(self._query_idx.shape[0]) // max(1, int(self.shard_world_size))

    def _sample_condition(self, i: int) -> int:
        # Cross-fps demos are allowed: the fps-aware PE places demo and query on a
        # shared *physical-time* grid (each item modulated by its own fps), so a
        # 30 Hz human demo aligns with a 20 Hz mimicdroid query over the same ~5 s
        # span even though their frame counts differ (149 vs 97).
        cands = self._match_idx[i]
        cands = cands[cands >= 0]
        if cands.size == 0:  # no valid demo: fall back to self-conditioning
            return int(self._query_idx[i])
        return int(cands[self._rng.integers(cands.size)])

    def __getitem__(self, i: int) -> dict[str, Any]:
        world = max(1, int(self.shard_world_size))
        gi = int(i) * world + (int(self.shard_rank) % world)  # strided per-rank slice
        row = int(self._order[gi])  # map through the global shuffle (breaks emb blocks)
        q = int(self._query_idx[row])
        # Demo sampling + δ draw run in BOTH arms (identical RNG consumption and
        # identical joint δ clamp), so the BC arm's query stream is byte-identical
        # to the ICL arm's — plain merely skips READING the demo window.
        c = self._sample_condition(row)
        delta = self._sample_frame_offset(q, c)
        # Which contract this item speaks. Derived from `row` by an integer mix rather
        # than drawn, for two reasons: the demo/δ draws above are documented to consume
        # RNG identically in the plain and ICL arms, and a fixed partition of the pool is
        # more useful than a per-epoch re-roll -- a given window is always 57-D or always
        # 20-D, so the two streams stay disjoint and reproducible.
        # Both halves must be eligible, not just the coin flip. Retrieval is
        # CROSS-EMBODIMENT by design — a mimicdroid query is routinely paired with an
        # aria/scale/MECKA demo — and eligibility is decided per-emb inside the reader.
        # Flipping the item alone therefore produced a 20-D query in front of a 57-D
        # demo whenever the pair straddled embodiments: no error, just a sample that
        # teaches the wrong contract. Measured at 1 in 400 before this guard.
        _view20 = (
            ((row * 2654435761) & 0xFFFFFFFF) / 0x100000000 < self._view20_prob
            and self._reader.view20_eligible(q)
            and (self._plain or self._reader.view20_eligible(c))
        )
        if self._plain:
            paired = {"query": self._reader.read_window(q, frame_offset=delta, view20=_view20)}
        else:
            # 2.5 contract: when the QUERY target is the absolute state chunk
            # (SMID), the DEMO segment still conditions on DELTA action rows with
            # NO state row (ret_video + ret_action; state is query-only). This
            # also means demo-pool-only episodes never need state arrays.
            _demo_kw = (
                {"action_source": "delta", "use_state": False}
                if self._reader._action_source == "state"
                else {}
            )
            paired = {
                # SAME δ for query AND demo → the demo is the t=0-retrieved clip advanced in
                # lockstep, not a re-retrieval (see _sample_frame_offset / read_window docs).
                "query": self._reader.read_window(q, frame_offset=delta, view20=_view20),
                "condition": self._reader.read_window(c, frame_offset=delta, view20=_view20, **_demo_kw),
            }
        return self._transform(paired) if self._transform is not None else paired

    def _sample_frame_offset(self, q: int, c: int) -> int:
        """Sub-stride frame offset δ shared by the query (window ``q``) and demo (``c``).

        Returns 0 when per-frame coverage is off (``per_frame_stride==1``). Otherwise
        samples δ uniformly over ``{0..cap}`` where ``cap`` is the largest offset that
        keeps BOTH windows in-bounds (joint clamp = ``min`` of the per-window limits and
        ``STRIDE-1``), so a single δ never overflows either episode."""
        if self._per_frame_stride <= 1:
            return 0
        max_delta = self._per_frame_stride - 1
        cap = min(
            max_delta,
            self._reader.max_frame_offset(q, max_delta),
            self._reader.max_frame_offset(c, max_delta),
        )
        return int(self._rng.integers(cap + 1)) if cap > 0 else 0


def get_action_egoverse_paired_sft_dataset(
    *,
    data_root: str,
    pairs_dir: str,
    normalization_path: str,
    speed_stats_path: str | None = None,
    chunk_length: int = 28,
    action_normalization: str = "quantile",
    action_source: str = "delta",
    resolution: str | int = "256",
    max_action_dim: int = 64,
    share_temporal: bool = True,
    keep_aspect_ratio: bool = False,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    fps: float = 30.0,
    use_state: bool = False,
    cams_override: dict[str, tuple[str, ...]] | None = None,
    seed: int = 0,
    per_frame_stride: int = 1,
    plain: bool = False,
    droplist_path: str | None = None,
    view20_embs: tuple[str, ...] = (),
    view20_domain: str = "robomind-franka-dual",
    view20_prob: float = 1.0,
    dummy_ret_prob: float = 0.0,
    dummy_ret_frames: int = 1,
) -> EgoVersePairedDataset:
    """Build the EgoVerse retrieval-paired conditioning-policy SFT dataset.

    Mirrors ``get_action_libero_sft_dataset`` but yields ``[demo, query]`` paired
    samples (video-transfer-style shared-grid conditioning, multi-domain action).
    """
    reader = EgoVerseWindowReader(
        data_root=data_root,
        pairs_dir=pairs_dir,
        normalization_path=normalization_path,
        speed_stats_path=speed_stats_path,
        chunk_length=chunk_length,
        action_normalization=action_normalization,
        action_source=action_source,
        fps=fps,
        use_state=use_state,
        cams_override=cams_override,
        view20_embs=view20_embs,
        view20_domain=view20_domain,
    )
    transform = EgoVersePairedTransform(
        resolution=resolution,
        max_action_dim=max_action_dim,
        share_temporal=share_temporal,
        keep_aspect_ratio=keep_aspect_ratio,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        plain=plain,
        dummy_ret_prob=dummy_ret_prob,
        dummy_ret_frames=dummy_ret_frames,
    )
    return EgoVersePairedDataset(
        reader=reader,
        pairs_dir=pairs_dir,
        transform=transform,
        seed=seed,
        per_frame_stride=per_frame_stride,
        plain=plain,
        droplist_path=droplist_path,
        view20_prob=view20_prob,
    )


if __name__ == "__main__":  # smoke test
    import sys

    PAIRS = "/mnt/ddn/dataset/egoverse_pairs_v7"
    D = "cosmos_framework/data/generator/action/datasets"
    reader = EgoVerseWindowReader(
        data_root="/mnt/ddn/dataset",
        pairs_dir=PAIRS,
        normalization_path=f"{D}/egoverse_normalization.json",
        speed_stats_path=f"{D}/egoverse_speed_stats.json",
    )
    ds = EgoVersePairedDataset(
        reader=reader, pairs_dir=PAIRS,
        transform=EgoVersePairedTransform(resolution="256", max_action_dim=64, share_temporal=True),
    )
    print(f"len(dataset) = {len(ds):,}")
    for i in (0, len(ds) // 2, len(ds) - 1):
        s = ds[i]
        v, a, sp = s["video"], s["action"], s["sequence_plan"]
        print(
            f"[{i}] video items={len(v)} shapes={[tuple(x.shape) for x in v]} "
            f"action segs={len(a)} shapes={[tuple(x.shape) for x in a]} "
            f"domain_id={[d.item() for d in s['domain_id']]} raw_dim={[d.item() for d in s['raw_action_dim']]}\n"
            f"      image_size={[x.tolist() for x in s['image_size']]} share_temporal={sp.share_vision_temporal_positions} "
            f"cond_vis={sp.condition_frame_indexes_vision} cond_act={sp.condition_frame_indexes_action}\n"
            f"      caption: '{s['ai_caption']}'"
        )
    sys.exit(0)

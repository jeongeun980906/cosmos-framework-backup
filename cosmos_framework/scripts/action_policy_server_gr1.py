# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HTTP inference server for the GR1 conditioning-policy (closed-loop online eval).

The GR1 policy generates each 32-step action chunk conditioned on a ``play`` demo
retrieved ONLINE from the robot's recent proprioception (``gr1_online_retrieval``,
a live port of ``75_future_pairs.py``). The robocasa GR1 env runs as a separate
client (``eval_gr1_client.py``) — robocasa need not be installed here.

Endpoints
---------
POST /reset   {"task": "<GR1 task>"}                         -> {"ok": true}
POST /predict {"task": "<task>", "image": <HxWx3 uint8 list or base64 png>,
               "proprio": {"pL":[3],"pR":[3],"quatL":[4],"quatR":[4],
                           "gripL": float, "gripR": float}}
              -> {"action57": [[57]*32], "demo": {"ep","w0","dist"}}

The client streams proprio every control step (server buffers it); each /predict
re-matches a demo from the past window, conditions, and returns the next chunk in
the egomimic ``action57`` space (the client decodes it to env actions).

Launch::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_gr1 \\
        --checkpoint <GR1 post-trained ckpt dir> --port 8000 \\
        --pool /mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool.npz --pool-kind taskdemo \\
        --vae-path /mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as _R


def _to_thwc_uint8(video: torch.Tensor) -> np.ndarray:
    """(C,T,H,W) in [-1,1] or uint8 → (T,H,W,3) uint8."""
    v = video.detach().float().cpu()
    if v.min() < -0.01:  # [-1,1] float → [0,255]
        v = (v.clamp(-1, 1) + 1.0) * 127.5
    return v.permute(1, 2, 3, 0).clamp(0, 255).to(torch.uint8).numpy()


def _resize_h(frames: np.ndarray, h: int) -> np.ndarray:
    """(T,H,W,3) uint8 → resized to height h (keep aspect)."""
    out = []
    for f in frames:
        w = max(1, round(f.shape[1] * h / f.shape[0]))
        out.append(np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)))
    return np.stack(out)


def _panel(obs: np.ndarray, gen: np.ndarray, demo: np.ndarray, h: int = 256) -> np.ndarray:
    """Build per-chunk panel frames [obs | generated | retrieved-demo], horizontally
    concat'd at common height h; obs (a single current frame) is tiled to T = gen length."""
    T = gen.shape[0]
    obs_t = np.repeat(obs[None], T, axis=0)
    cols = [_resize_h(obs_t, h), _resize_h(gen, h), _resize_h(demo[:T] if len(demo) >= T else
            np.concatenate([demo, np.repeat(demo[-1:], T - len(demo), 0)]), h)]
    W = max(c.shape[2] for c in cols)
    cols = [np.pad(c, ((0, 0), (0, 0), (0, W - c.shape[2]), (0, 0))) for c in cols]
    return np.concatenate(cols, axis=2)  # (T, h, 3*W, 3)

from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.data.generator.action.action_normalization import load_action_stats
from cosmos_framework.data.generator.action.action_processing import ActionProcessingRecord
from cosmos_framework.data.generator.action.datasets.egoverse_zarr_dataset import (
    _GR1_TASK_LANG,
    _open_array,
    EgoVersePairedTransform,
    EgoVerseWindowReader,
    get_domain_id,
)
from cosmos_framework.scripts.gr1_online_retrieval import (
    OnlineRetriever,
    episode_windows,
    match_within_idx,
)
from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint

_NORM = "cosmos_framework/data/generator/action/datasets/egoverse_normalization.json"
DEMO_EMB = "mimicdroid"   # play demos
QUERY_EMB = "gr1"
CHUNK = 32                # action steps; video = 33 frames @20 Hz
SPAN = 100                # play window span (reader floors to 33 frames @20 Hz)

_LVL_RE = re.compile(r"L\d+$")


def _base_task(task: str) -> str:
    """Strip a trailing level suffix (``L2``/``L3``/…) so L3/L2 eval — the SAME task in an
    unseen layout/style — conditions on and is captioned as the BASE task (the taskdemo pool
    and ``_GR1_TASK_LANG`` are keyed by base names). ``PnPSinkToCabinetL3`` -> ``PnPSinkToCabinet``."""
    return _LVL_RE.sub("", task)


# --- GR1 query state57 from live env proprio (use_state) -----------------------
# Mirrors the absolute state57 written by tools/convert_taskdemos.py &
# 10_mimicdroid_convert.py: [HEAD9 | abs9_L | hand15_L | abs9_R | hand15_R]. MUST
# match the converter so the eval state0 lives in the same space the demo's state0
# (read from the zarr) was trained on. abs9(pos,quat) = [pos, R[:,0], R[:,1]];
# hand15(g) = lerp(OPEN15, CLOSED15, g) with g in [0,1].
_HEAD9 = np.array([-0.60, 0.0, 1.17, 0.033, -0.999, -0.003, 0.818, 0.028, -0.574], float)
_OPEN15 = np.array([0.0005, 0.0958, -0.034, 0.0285, 0.0222, -0.1263, 0.0085, 0.0217, -0.1285,
                    -0.0116, 0.0222, -0.1282, -0.0314, 0.0229, -0.1259], float)
_CLOSED15 = np.array([0.0052, 0.0775, -0.0701, 0.0266, 0.044, -0.0916, 0.0085, 0.0441, -0.1016,
                      -0.0097, 0.044, -0.0937, -0.0278, 0.044, -0.0919], float)


def _abs9_one(pos, quat_xyzw) -> np.ndarray:
    q = np.asarray(quat_xyzw, float)
    n = np.linalg.norm(q)
    q = q / n if (n > 1e-6 and np.isfinite(q).all()) else np.array([0.0, 0.0, 0.0, 1.0])
    Rm = _R.from_quat(q).as_matrix()
    return np.concatenate([np.asarray(pos, float), Rm[:, 0], Rm[:, 1]])


def _state57_from_proprio(p: dict) -> np.ndarray:
    """{pL,pR,quatL,quatR,gripL,gripR} (current env step) -> absolute state57."""
    def hand15(g):
        g = float(np.clip(g, 0.0, 1.0))
        return (1.0 - g) * _OPEN15 + g * _CLOSED15

    return np.concatenate([
        _HEAD9,
        _abs9_one(p["pL"], p["quatL"]), hand15(p["gripL"]),
        _abs9_one(p["pR"], p["quatR"]), hand15(p["gripR"]),
    ]).astype(np.float32)


# --- action57 delta -> absolute base-frame eef pose (server-side integration) -----
# Each arm's eef9 = [Δpos(3), 6D-rot(= ΔR columns 0,1)] is a per-step BODY-frame DELTA
# (tools/convert_taskdemos.py::_eef9). Integrate onto the CURRENT measured eef pose (from
# proprio) so every chunk re-anchors on the live state (no cross-chunk drift):
#     p_{t+1} = p_t + R_t·Δpos,   R_{t+1} = R_t·ΔR    (body/body, matches the converter).
# state0 (the absolute pose) IS the input, so we integrate here on the server. Output is the
# GR1 OSC_POSE action layout (input_type=absolute, input_ref_frame=base, ori=axis-angle):
#   [Rpos3, Raa3, Lpos3, Laa3, grip12, grip13]  (pose block 0:6=right, 6:12=left — verified
#   empirically against robot0_base_to_{right,left}_eef_pos). grip = 2g-1 with g∈[0,1] (1=closed)
#   recovered by projecting hand15 onto the predefined OPEN15→CLOSED15 axis; idx12←Lhand15,
#   idx13←Rhand15 (mirrors the converter's GRIP_CMD=(12,13) so train/eval stay self-consistent).
def _quat_mat(quat_xyzw) -> np.ndarray:
    q = np.asarray(quat_xyzw, float)
    n = np.linalg.norm(q)
    q = q / n if (n > 1e-6 and np.isfinite(q).all()) else np.array([0.0, 0.0, 0.0, 1.0])
    return _R.from_quat(q).as_matrix()


def _sixd_to_R(c0, c1) -> np.ndarray:
    """Two stored ΔR columns → full rotation matrix (Gram-Schmidt)."""
    a = c0 / (np.linalg.norm(c0) + 1e-9)
    b = c1 - (a @ c1) * a
    b = b / (np.linalg.norm(b) + 1e-9)
    return np.stack([a, b, np.cross(a, b)], axis=1)  # columns [a, b, a×b]


def _grip_from_hand15(h15: np.ndarray) -> float:
    """Invert the predefined-pose lerp hand15 = (1-g)·OPEN15 + g·CLOSED15 → grasp signal g∈[0,1]."""
    axis = _CLOSED15 - _OPEN15
    return float(np.clip(((np.asarray(h15, float) - _OPEN15) @ axis) / (axis @ axis), 0.0, 1.0))


def _integrate_arm(p0, R0, eef9) -> tuple[np.ndarray, np.ndarray]:
    """Integrate body-frame deltas onto (p0,R0) → per-step absolute (pos3, axis-angle3).

    Output is in the GR1 **body-eef** frame (the frame action57/state57 live in — what
    73_gr1_to_egoverse57.py built from ``robot0_base_to_*_eef_quat``). The OSC control
    frame (gripper site) is offset from this; apply ``_apply_ctrl_offset`` before sending
    to the env. (Kept offset-free so the round-trip unit test stays exact.)"""
    p = np.asarray(p0, float).copy()
    Rm = np.asarray(R0, float).copy()
    pos, aa = [], []
    for d in np.asarray(eef9, float):
        p = p + Rm @ d[:3]
        Rm = Rm @ _sixd_to_R(d[3:6], d[6:9])
        pos.append(p.copy())
        aa.append(_R.from_matrix(Rm).as_rotvec())
    return np.asarray(pos), np.asarray(aa)


# GR1 body-eef → OSC control (gripper site) rigid offset, per arm. Solved from the GR1 h5
# (robot0_base_to_*_eef vs the replayed 14-dim action) on near-static frames: env target =
# [p_body + R_body·p_off, axisangle(R_body·R_off)]. These are GR1-robot geometry constants
# (DemoTwoHand's differ — must use GR1's). recon on static frames: R 1.1°/0.4cm, L 2.1°/0.7cm.
_CTRL_OFF = {
    "R": (np.array([0.001236, 0.002752, -0.999984, -0.004839]), np.array([0.00104, 0.01354, 0.00726])),
    "L": (np.array([0.707104, 0.004116, 0.707094, 0.002261]), np.array([0.00012, -0.01272, 0.00862])),
}


def _apply_ctrl_offset(pos, aa, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Body-eef (pos, axis-angle) → OSC control-frame (pos, axis-angle) for ``arm`` in {L,R}."""
    roff_q, poff = _CTRL_OFF[arm]
    Roff = _R.from_quat(roff_q).as_matrix()
    out_p, out_a = [], []
    for p, a in zip(np.asarray(pos), np.asarray(aa)):
        Rb = _R.from_rotvec(a).as_matrix()
        out_p.append(p + Rb @ poff)
        out_a.append(_R.from_matrix(Rb @ Roff).as_rotvec())
    return np.asarray(out_p), np.asarray(out_a)


def _decode_image(img) -> np.ndarray:
    """{list HxWx3 uint8 | base64 png} -> (H,W,3) uint8."""
    if isinstance(img, str):
        raw = base64.b64decode(img.split(",")[-1])
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    return np.asarray(img, dtype=np.uint8)


def _query_obs_image(req: dict, n_cams: int) -> np.ndarray:
    """Build the query obs frame. With ``n_cams>1`` (concat_view) the client sends
    ``req["images"]`` = [center, sideleft, sideright] (cam order matching the reader's
    cams_override); they are vertically concat'd to match the trained video (e.g.
    384×128). Center-only falls back to the single ``req["image"]``."""
    if n_cams > 1:
        imgs = req.get("images")
        if imgs is None or len(imgs) < n_cams:
            raise ValueError(f"concat_view eval needs req['images'] with >= {n_cams} views (center,sideleft,sideright,...)")
        # The 2026-07-16+ client streams 4 views (center, sideleft, sideright, center_top)
        # for grid4 servers; 3-cam concat_view training uses the FIRST 3 (matching order).
        return np.concatenate([_decode_image(im) for im in imgs[:n_cams]], axis=0)  # vertical stack
    return _decode_image(req["image"])


class GR1Policy:
    """Loads the model + retriever + reader and serves one action chunk per call."""

    def __init__(self, args) -> None:
        self.model, self.config = load_model_from_checkpoint(
            experiment_name=args.experiment,
            checkpoint_path=args.checkpoint,
            parallelism_config={"data_parallel_shard_degree": 1, "data_parallel_replicate_degree": 1},
            experiment_opts=[f"model.config.tokenizer.vae_path={args.vae_path}"],
        )
        self.model = self.model.cuda().eval()
        tok = self.config.model.config.vlm_config.tokenizer
        # Match training: if the policy was trained with use_state, the demo clip and
        # the query BOTH carry a state0 conditioning row. Derive it from the loaded
        # experiment config so the server can't drift from the checkpoint.
        try:
            self.use_state = bool(
                self.config.dataloader_train.dataloader.datasets.egoverse.dataset.use_state
            )
        except Exception:  # noqa: BLE001 — config shape varies; default to no-state
            self.use_state = False
        # Match training target: "delta" (action57 frame diffs, integrate at decode) or
        # "abs" (v6 abs_action57 command rows, executed DIRECTLY — the cosmos_policy
        # `--pred-target abs_action57` contract). Derived from the checkpoint's config
        # so the server can't drift from what the model was trained to predict.
        try:
            self.action_source = str(
                self.config.dataloader_train.dataloader.datasets.egoverse.dataset.action_source
            )
        except Exception:  # noqa: BLE001 — older configs predate the knob
            self.action_source = "delta"
        # BC ("plain") checkpoints: query-only samples, retrieval BYPASSED entirely
        # (the cosmos_policy plain-eval contract — a retriever miss must not gate
        # the policy). Derived from the checkpoint's config.
        try:
            self.plain = bool(
                self.config.dataloader_train.dataloader.datasets.egoverse.dataset.plain
            )
        except Exception:  # noqa: BLE001 — paired (ICL) default
            self.plain = False
        # Match training cameras: the post-train config sets cams_override (3 agentviews
        # vertically concat'd) for the demo (mimicdroid) AND query (gr1). Read it from the
        # checkpoint's config so the eval video matches what the model was trained on.
        try:
            cams_ov = dict(
                self.config.dataloader_train.dataloader.datasets.egoverse.dataset.cams_override
            )
            cams_ov = {k: tuple(v) for k, v in cams_ov.items()}
        except Exception:  # noqa: BLE001 — center-only run
            cams_ov = None
        self.reader = EgoVerseWindowReader(
            data_root=args.data_root, pairs_dir=args.pairs_dir, normalization_path=_NORM,
            speed_stats_path=None, chunk_length=CHUNK, fps=20.0, use_state=self.use_state,
            action_source=self.action_source, cams_override=cams_ov,
        )
        self._n_cams = len((cams_ov or {}).get(QUERY_EMB, ("images.front_1",)))
        print(
            f"[GR1Policy] use_state={self.use_state} action_source={self.action_source} "
            f"n_cams={self._n_cams}", flush=True,
        )
        self.transform = EgoVersePairedTransform(
            resolution=str(args.resolution), max_action_dim=self.config.model.config.max_action_dim,
            share_temporal=True, keep_aspect_ratio=False, tokenizer_config=tok,
            plain=self.plain,
        )
        self.retriever = OnlineRetriever(args.pool, kind=args.pool_kind)
        # Optional t=0=taskdemo-opening → t>0=play switch (the contract). When --play-pool is
        # given, --pool must be the taskdemo pool (t=0) and this is the play pool (t>0).
        self.play_retriever = (
            OnlineRetriever(args.play_pool, kind=getattr(args, "play_pool_kind", "dth"))
            if getattr(args, "play_pool", None) else None
        )
        self._t0_taskdemo = args.pool_kind == "taskdemo"   # force the t=0 chunk to the taskdemo OPENING [0:32]
        self._chunk = 0
        # --- One-shot pinned-demo eval (cosmos_policy retrieval rules, 2026-07-16 인증) ---
        # /reset {"oneshot_demo": {ep, w0}} or {"oneshot_task": "<task[변형]>"} pins the
        # demo(s) for the whole episode; chunks >=1 re-match the robot's strict past
        # within the pinned episode(s) under the MONO band [last+floor, last+48]
        # (certified: mono, NO pin-lock; per-episode floor via /reset "mono_floor" —
        # FaucetL3 runs floor 16, everything else the default 8).
        self.pinned_advance = getattr(args, "pinned_advance", "mono")
        self.pin_lock_ep = bool(getattr(args, "pin_lock_ep", False))
        self._mono_floor_default = int(getattr(args, "mono_floor", 8))
        self.mono_floor = self._mono_floor_default
        self._mono_band = 48
        self._data_root = str(args.data_root)
        self._pinned = None
        self._pinned_windows = None
        self._pinned_eps = None
        self._pinned_last_w0 = 0
        self._pinned_lock_ep = None
        # GR1 action de-normalization stats (quantile inverse) for the returned chunk.
        # abs mode inverts with the SAME v6 pctl stats the training normalized with
        # (``gr1_abs`` = lo/hi of egoverse_pairs_gr1_v6/abs_action57_stats_pctl.json).
        _stats_key = f"{QUERY_EMB}_abs" if self.action_source == "abs" else QUERY_EMB
        st = load_action_stats(_NORM, stats_key=_stats_key)  # values are np.ndarray
        self._q01 = np.asarray(st["q01"], dtype=np.float32)
        self._q99 = np.asarray(st["q99"], dtype=np.float32)
        self.num_steps = args.num_steps
        self.guidance = args.guidance
        self.shift = args.shift
        self.seed = args.seed
        # Eval video logging: accumulate per-chunk [obs | generated | retrieved-demo]
        # panels across an episode → mp4 on reset. Off unless --save-video-dir is set.
        self._save_dir = getattr(args, "save_video_dir", None)
        if self._save_dir:
            os.makedirs(self._save_dir, exist_ok=True)
        self._vbuf: list[np.ndarray] = []
        self._cbuf: list[np.ndarray] = []   # per-chunk retrieved-demo condition clip [w0:w0+33]
        self._ep_idx = 0
        self.fps = args.fps

    def _flush_video(self) -> None:
        if self._save_dir and self._vbuf:
            frames = np.concatenate(self._vbuf, axis=0)  # (sum_T, h, 3W, 3) over the episode
            path = os.path.join(self._save_dir, f"ep_{self._ep_idx:03d}.mp4")
            imageio.mimwrite(path, frames, fps=self.fps, macro_block_size=1)
            print(f"[GR1Policy] saved eval video {path} ({len(frames)} frames)", flush=True)
        if self._save_dir and self._cbuf:
            cond = np.concatenate(self._cbuf, axis=0)  # (sum_T, H, W, 3): condition stream = retrieved [w0:w0+33] per chunk
            cpath = os.path.join(self._save_dir, f"cond_ep_{self._ep_idx:03d}.mp4")
            imageio.mimwrite(cpath, cond, fps=self.fps, macro_block_size=1)
            print(f"[GR1Policy] saved condition stream {cpath} ({len(cond)} frames)", flush=True)
        self._vbuf = []
        self._cbuf = []

    def reset(self, req: dict | None = None) -> None:
        self._flush_video()          # write the just-finished episode (if any)
        self._ep_idx += 1
        self._chunk = 0
        self.retriever.reset()
        if self.play_retriever is not None:
            self.play_retriever.reset()
        # Per-episode mono-floor override (task rule 2026-07-16: TurnOnFaucetL3 runs
        # floor 16, everything else the server default) — client sends it in /reset.
        self.mono_floor = int((req or {}).get("mono_floor", self._mono_floor_default))
        # EXPLICIT-DEMO one-shot mode — mirrors cosmos_policy eval_gr1_server.reset():
        # oneshot_demo pins ONE (ep, w0); oneshot_task pins ALL 3 taskdemos of the task
        # (within-episode matching over their COMBINED windows).
        self._pinned = None
        self._pinned_windows = None
        self._pinned_eps = None
        self._pinned_last_w0 = 0
        self._pinned_lock_ep = None
        od = (req or {}).get("oneshot_demo")
        ot = (req or {}).get("oneshot_task")
        hum_root = os.path.join(self._data_root, "egoverse")
        if ot and not od:
            eps3 = [f"taskdemo_{ot}_{ot}_D_demo_{i}" for i in (1, 2, 3)]
            eps3 = [e for e in eps3 if os.path.isdir(os.path.join(hum_root, e))]
            if not eps3:
                raise FileNotFoundError(f"oneshot_task has no taskdemos: {ot}")
            od = {"ep": eps3[0], "w0": 0, "_all": eps3}
        if od:
            ep, w0 = str(od["ep"]), int(od.get("w0", 0))
            if not os.path.isdir(os.path.join(hum_root, ep)):
                raise FileNotFoundError(f"oneshot_demo episode not found: {hum_root}/{ep}")
            self._pinned = (ep, w0)
            # WITHIN-EPISODE matching (not blind lockstep): precompute the pinned
            # episode(s)' past-window descriptors; chunk 0 conditions on the opening
            # (w0 as sent), chunks >=1 re-match the robot's strict past against these.
            _eps = od.get("_all", [ep]) if isinstance(od, dict) else [ep]
            _w0s, _ts, _eids = [], [], []
            for _e in _eps:
                st = np.asarray(
                    _open_array(os.path.join(hum_root, _e, "state57", "state")), dtype=np.float64
                )
                w0s_e, t_e = episode_windows(st)
                _w0s.append(np.asarray(w0s_e))
                _ts.append(np.asarray(t_e))
                _eids += [_e] * len(w0s_e)
            self._pinned_windows = (np.concatenate(_w0s), np.concatenate(_ts))
            self._pinned_eps = np.array(_eids)
            print(
                f"[GR1Policy] one-shot pinned: {len(_eps)} demo(s) "
                f"({len(self._pinned_eps)} windows for within-episode match, "
                f"advance={self.pinned_advance}, floor={self.mono_floor})", flush=True,
            )

    def _make_query(self, obs_img: np.ndarray, task: str, n_frames: int, proprio: dict) -> dict:
        """Current GR1 obs → a read_window-like query (frame 0 clean, rest generated).

        With ``use_state`` the action tensor is (CHUNK+1, dim): row 0 is the current
        env state (absolute state57, normalized with ``gr1_state``) — the clean
        conditioning the transform marks as ``condition_frame_indexes_action=[0]`` —
        and rows 1.. are the generated delta chunk."""
        vid = np.repeat(obs_img[None], n_frames, axis=0)                  # (T,H,W,3)
        video = torch.from_numpy(vid).permute(3, 0, 1, 2).contiguous()    # (C,T,H,W) uint8
        _act_key = f"{QUERY_EMB}_abs" if self.action_source == "abs" else QUERY_EMB
        dim = self.reader._stats[_act_key]["q01"].numel()
        action = torch.zeros(CHUNK + (1 if self.use_state else 0), dim, dtype=torch.float32)
        if self.use_state:
            st = torch.from_numpy(_state57_from_proprio(proprio)).float().unsqueeze(0)  # (1,57)
            action[0, :57] = self.reader._normalize(st, f"{QUERY_EMB}_state")[0]
        return {
            "video": video,
            "action": action,
            "domain_id": torch.tensor(get_domain_id("hand_pose"), dtype=torch.long),
            # Language condition MUST match training: read_window builds the GR1 query caption as
            # ``"gr1: <natural-language instruction>"`` (via _GR1_TASK_LANG), NOT the raw CamelCase
            # task tag. GR1 post-train drops speed, so no ". speed:" suffix.
            "emb": QUERY_EMB, "raw_action_dim": 57,
            "ai_caption": f"{QUERY_EMB}: {_GR1_TASK_LANG.get(task, task)}",
            # The paired/plain transform builds the TRAINING caption from ``task``
            # ("<instruction> demo <emb> target <emb>" / "<instruction> target <emb>");
            # without this key it would fall back to ai_caption and leak the "gr1: "
            # prefix that training never sees.
            "task": _GR1_TASK_LANG.get(task, task),
            "speed": None, "conditioning_fps": torch.tensor(20, dtype=torch.long),
        }

    def _abs_to_env_action(self, ab: np.ndarray) -> np.ndarray:
        """ABS COMMAND chunk (C,57; abs_action57, SITE frame) -> env actions DIRECTLY.
        No _CTRL_OFF: these ARE the recorded env-action targets (v6, row t = act[t]).
        Verbatim port of cosmos_policy eval_gr1_server._abs_to_env_action so the
        Cosmos3-Edge eval decodes IDENTICALLY to the 2.5 v6 eval."""
        ab = np.asarray(ab, float)

        def _arm(block):
            pos = block[:, :3]
            aa = np.stack([_R.from_matrix(_sixd_to_R(b[3:6], b[6:9])).as_rotvec() for b in block])
            return pos, aa

        Lpos, Laa = _arm(ab[:, 9:18])
        Rpos, Raa = _arm(ab[:, 33:42])
        gL = np.array([_grip_from_hand15(h) for h in ab[:, 18:33]])
        gR = np.array([_grip_from_hand15(h) for h in ab[:, 42:57]])
        return np.concatenate(
            [Rpos, Raa, Lpos, Laa, (2 * gL - 1)[:, None], (2 * gR - 1)[:, None]], axis=1
        ).astype(np.float32)

    def _to_env_action(self, act: np.ndarray, p: dict) -> np.ndarray:
        """(32,57) de-quantized action57 + current proprio → (32,14) GR1 env actions."""
        Lpos, Laa = _integrate_arm(p["pL"], _quat_mat(p["quatL"]), act[:, 9:18])    # left arm (body)
        Rpos, Raa = _integrate_arm(p["pR"], _quat_mat(p["quatR"]), act[:, 33:42])   # right arm (body)
        Lpos, Laa = _apply_ctrl_offset(Lpos, Laa, "L")    # body-eef → OSC control (gripper site)
        Rpos, Raa = _apply_ctrl_offset(Rpos, Raa, "R")
        gL = np.array([_grip_from_hand15(h) for h in act[:, 18:33]])
        gR = np.array([_grip_from_hand15(h) for h in act[:, 42:57]])
        # env layout: [Rpos3, Raa3, Lpos3, Laa3, grip12(=2gL-1), grip13(=2gR-1)]
        return np.concatenate(
            [Rpos, Raa, Lpos, Laa, (2 * gL - 1)[:, None], (2 * gR - 1)[:, None]], axis=1
        ).astype(np.float32)

    def _hold_action(self, p: dict) -> np.ndarray:
        """No-op chunk: hold the current absolute eef pose + current grip for CHUNK steps."""
        pL = np.asarray(p["pL"], float)[None]
        pR = np.asarray(p["pR"], float)[None]
        aaL = _R.from_matrix(_quat_mat(p["quatL"])).as_rotvec()[None]
        aaR = _R.from_matrix(_quat_mat(p["quatR"])).as_rotvec()[None]
        pL, aaL = _apply_ctrl_offset(pL, aaL, "L")        # same body→control offset as motion
        pR, aaR = _apply_ctrl_offset(pR, aaR, "R")
        one = np.concatenate([pR[0], aaR[0], pL[0], aaL[0],
                              [2 * p["gripL"] - 1, 2 * p["gripR"] - 1]]).astype(np.float32)
        return np.tile(one, (CHUNK, 1))

    def push(self, req: dict) -> dict:
        """Append proprio frame(s) to the retrieval buffer(s) WITHOUT generating. Accepts a single
        ``proprio`` or a batched ``proprios`` list. The client streams per-env-step proprio between
        chunks so the t>0 PAST window is a real 32-frame trajectory (else the buffer holds 1 coarse
        point/chunk and the play match never advances)."""
        props = req.get("proprios") or [req["proprio"]]
        for p in props:
            self.retriever.push(p["pL"], p["pR"], p["quatL"], p["quatR"], p["gripL"], p["gripR"])
            if self.play_retriever is not None:
                self.play_retriever.push(p["pL"], p["pR"], p["quatL"], p["quatR"], p["gripL"], p["gripR"])
        return {"ok": True}

    @torch.no_grad()
    def predict(self, req: dict) -> dict:
        # L3/L2 eval runs the SAME task in an unseen layout/style. Split the task usage:
        #   retr_task = FULL name  -> retrieval gates to that task's OWN TaskDemos. With an
        #     L3-inclusive pool this conditions L3 queries on the MATCHING unseen-layout demo;
        #     with a base-only pool it falls back to base (cross-layout).
        #   task (base) = language caption + play-pool subgoal gate (both keyed by base names;
        #     the instruction is layout-agnostic).
        retr_task = req["task"]
        task = _base_task(retr_task)
        p = req["proprio"]
        if not req.get("skip_push"):  # client streams via /push; single-frame callers push here
            self.retriever.push(p["pL"], p["pR"], p["quatL"], p["quatR"], p["gripL"], p["gripR"])
            if self.play_retriever is not None:
                self.play_retriever.push(p["pL"], p["pR"], p["quatL"], p["quatR"], p["gripL"], p["gripR"])
        if self.plain:
            # BC arm: no demo, no retrieval gate — query-only sample (33 frames,
            # frame-0 obs + state0 row clean, chunk generated), exactly the plain
            # training layout. Mirrors the cosmos_policy plain-eval bypass.
            self._chunk += 1
            demo = None
            demo_sample = None
            query_sample = self._make_query(_query_obs_image(req, self._n_cams), task, CHUNK + 1, p)
            sample = self.transform({"query": query_sample})
        elif self._pinned is not None:
            # PINNED one-shot demo: no pool retriever calls. Chunk 0 conditions on the
            # pinned opening; later chunks re-match the robot's strict past WITHIN the
            # pinned episode(s). Mirrors cosmos_policy eval_gr1_server (2026-07-16 인증:
            # mono band [last+floor, last+48], end-saturation clamp stays on the locked
            # episode, no pin-lock unless --pin-lock-ep).
            ep, w0_pin = self._pinned
            if self._chunk == 0 or self._pinned_windows is None:
                w0 = w0_pin                    # opening chunk: the demo's start (t=0 convention)
            else:
                w0s, t_arr = self._pinned_windows
                w0_max = int(w0s[-1]) if len(w0s) else w0_pin
                last = self._pinned_last_w0
                if self.pinned_advance == "lockstep":
                    w0 = min(last + getattr(self, "_pointer_step", CHUNK), w0_max)
                else:
                    src = self.play_retriever if self.play_retriever is not None else self.retriever
                    q = src.query()
                    eps_c = self._pinned_eps
                    w0s_c, t_c = w0s, t_arr
                    _lock = self._pinned_lock_ep
                    if self.pin_lock_ep and eps_c is not None and _lock is not None:
                        _mk = eps_c == _lock
                        if _mk.any():
                            w0s_c, t_c, eps_c = w0s_c[_mk], t_c[_mk], eps_c[_mk]
                    w0_max_c = int(w0s_c[-1]) if len(w0s_c) else w0_max
                    if q is None:
                        w0 = w0_pin
                    elif self.pinned_advance == "mono":
                        # Monotonic band + minimum advance — no phase-alias jumps and
                        # no stall self-loop. Floor default 8 (+16 for FaucetL3 via
                        # /reset mono_floor).
                        lo, hi = last + self.mono_floor, last + self._mono_band
                        m = (w0s_c >= lo) & (w0s_c <= hi)
                        if m.any():
                            _j = match_within_idx(q[0], w0s_c[m], t_c[m])
                            w0 = int(w0s_c[m][_j])
                            if eps_c is not None:
                                ep = str(eps_c[np.where(m)[0][_j]])
                        else:
                            # Band beyond the demo's last window (end saturation):
                            # clamp — and STAY on the locked episode (never flip the
                            # conditioning demo mid-episode).
                            w0 = min(lo, w0_max_c)
                            if _lock is not None:
                                ep = str(_lock)
                    else:                       # "match": unconstrained union re-match
                        _j = match_within_idx(q[0], w0s_c, t_c)
                        w0 = int(w0s_c[_j])
                        if eps_c is not None:
                            ep = str(eps_c[_j])
                    if self.pin_lock_ep and _lock is None:
                        self._pinned_lock_ep = ep   # first re-match fixes the episode
                w0 = int(min(w0, w0_max)) if self._pinned_windows is not None else int(w0)
            self._pinned_last_w0 = w0
            self._chunk += 1
            hits = [{"ep": ep, "w0": int(w0), "dist": 0.0}]
        else:
            if self.play_retriever is not None and self._chunk > 0:
                hits = self.play_retriever.retrieve(task, topk=1)        # t>0 → play PAST-window abs-traj match
            else:
                hits = self.retriever.retrieve(retr_task, topk=1)        # t=0 (or single-pool) → taskdemo (gated by FULL task)
                if hits and self._chunk == 0 and self._t0_taskdemo:
                    hits[0]["w0"] = 0                                    # t=0 → same-task taskdemo OPENING [0:32]
            self._chunk += 1
        if not self.plain:
            # Shared demo→sample assembly for the pinned AND pool paths.
            if not hits:  # not enough history yet → hold current pose
                return {"action57": np.zeros((CHUNK, 57), np.float32).tolist(),
                        "env_action": self._hold_action(p).tolist(), "demo": None}
            demo = hits[0]
            demo_sample = self.reader.read_clip(demo["ep"], DEMO_EMB, demo["w0"], span=SPAN)
            query_sample = self._make_query(_query_obs_image(req, self._n_cams), task, demo_sample["video"].shape[1], p)
            sample = self.transform({"query": query_sample, "condition": demo_sample})
        data_batch = {k: _to_cuda(v) for k, v in custom_collate_fn([sample]).items()}
        # Upstream inference extraction requires a per-sample processing record to
        # externalize (unpad) the generated action. Normalizer=None: the server
        # inverts normalization itself with the training stats (self._q01/_q99).
        data_batch["action_processing_record"] = [
            ActionProcessingRecord(raw_action_dim=57, action_normalizer=None)
        ]
        if isinstance(data_batch.get("image_size"), list):
            data_batch["image_size"] = [
                it for s in data_batch["image_size"] for it in (s if isinstance(s, list) else [s])
            ]
        result = self.model.generate_samples_from_batch(
            data_batch, guidance=self.guidance, seed=[self.seed], n_sample=1,
            num_steps=self.num_steps, shift=self.shift,
        )
        act_norm = result["action"][-1].detach().float().cpu().numpy()[:, :57]   # (33,57) use_state else (32,57)
        if self.use_state:
            act_norm = act_norm[1:]   # drop the state0 conditioning row → 32 generated deltas
        act = (act_norm + 1.0) / 2.0 * np.clip(self._q99 - self._q01, 1e-8, None) + self._q01  # de-quantile

        if self._save_dir:  # append this chunk's [obs | generated | retrieved-demo] panel
            gen_v = _to_thwc_uint8(self.model.decode(result["vision"][-1])[0])      # generated future
            obs_v = query_sample["video"].permute(1, 2, 3, 0)[0].numpy().astype(np.uint8)  # current obs frame
            if demo_sample is not None:
                demo_v = _to_thwc_uint8(demo_sample["video"])                       # retrieved demo clip
                self._vbuf.append(_panel(obs_v, gen_v, demo_v))
                self._cbuf.append(demo_v)   # raw condition clip this chunk (concat over episode = condition stream)
            else:  # plain: no demo panel column
                self._vbuf.append(_panel(obs_v, gen_v, np.zeros_like(gen_v)))
        if self.action_source == "abs":
            env_action = self._abs_to_env_action(act)   # (32,14) direct command execution (bias-free)
        else:
            env_action = self._to_env_action(act, p)   # (32,14) base-frame absolute eef + grip
        demo_frame = (
            _to_thwc_uint8(demo_sample["video"])[0, :128].tolist() if demo_sample is not None else None
        )   # retrieved demo center-cam frame0 (None for plain)
        return {"action57": act.astype(np.float32).tolist(),
                "env_action": env_action.tolist(),
                "demo_frame": demo_frame, "demo": demo}


def _to_cuda(v):
    if torch.is_tensor(v):
        return v.cuda()
    if isinstance(v, list):
        return [_to_cuda(x) for x in v]
    return v


def make_handler(policy: GR1Policy):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path.rstrip("/") == "/reset":
                    policy.reset(req)
                    self._send({"ok": True})
                elif self.path.rstrip("/") == "/push":
                    self._send(policy.push(req))
                elif self.path.rstrip("/") == "/predict":
                    self._send(policy.predict(req))
                else:
                    self._send({"error": f"unknown path {self.path}"}, 404)
            except Exception as e:  # noqa: BLE001 — return the error to the client, keep serving
                import traceback
                self._send({"error": str(e), "trace": traceback.format_exc()}, 500)

        def log_message(self, *a):  # quieter
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="action_policy_gr1_v6_icl_edge")
    ap.add_argument("--checkpoint", required=True, help="GR1 post-trained DCP checkpoint dir")
    # v6 pool defaults — IDENTICAL to cosmos_policy eval_gr1_server (taskdemo_pool_v6 at
    # t=0, dth_pool_1x_v6 for t>0), so the Edge eval retrieves the same demos as the 2.5 eval.
    ap.add_argument("--pool", default="/mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool_v6.npz",
                    help="retrieval pool npz: taskdemo_pool_v6.npz (per-task, default) or play/dth pools")
    ap.add_argument("--pool-kind", default="taskdemo", choices=["taskdemo", "play", "dth"],
                    help="taskdemo = per-task TaskDemos pool (eval); play/dth = subgoal-gated pools")
    ap.add_argument("--play-pool", default="/mnt/ddn/dataset/egoverse_pairs_v2/dth_pool_1x_v6.npz",
                    help="play/dth pool npz: the t=0 (first) chunk conditions on the "
                         "--pool taskdemo OPENING [0:32] and every later chunk uses this pool's PAST-window "
                         "abs-traj match (the contract's t=0=taskdemo → t>0=play switch). '' disables.")
    ap.add_argument("--play-pool-kind", default="dth", choices=["play", "dth"],
                    help="descriptor kind of --play-pool (v6 default: dth 20-d grip-augmented)")
    # One-shot pinned-demo advance rules (2026-07-16 인증: mono, NO pin-lock, floor 8;
    # FaucetL3 floor 16 via /reset payload mono_floor).
    ap.add_argument("--pinned-advance", default="mono", choices=["match", "lockstep", "mono"],
                    help="how the pinned one-shot demo window advances across chunks")
    ap.add_argument("--mono-floor", type=int, default=8,
                    help="mono band minimum advance per chunk (default 8; FaucetL3 uses 16 via /reset)")
    ap.add_argument("--pin-lock-ep", action="store_true",
                    help="lock within-episode re-matching to the first matched episode (training semantics)")
    ap.add_argument("--data-root", default="/mnt/ddn/dataset")
    ap.add_argument("--pairs-dir", default="/mnt/ddn/dataset/egoverse_pairs_gr1_v6")
    ap.add_argument("--vae-path", default="/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth")
    ap.add_argument("--resolution", default="256")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--save-video-dir", default=None,
                    help="if set, save per-episode eval mp4 [obs | generated | retrieved-demo] here")
    ap.add_argument("--fps", type=float, default=15.0, help="eval video fps")
    args = ap.parse_args()

    policy = GR1Policy(args)
    import atexit
    atexit.register(policy._flush_video)  # write the final episode's video on shutdown
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f"GR1 conditioning-policy server ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

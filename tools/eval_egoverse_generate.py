# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline generation test for the EgoVerse conditioning policy.

Loads a trained checkpoint, takes retrieval-paired samples (demo condition +
query frame-0 observation + caption), generates the query's future VIDEO + ACTION,
and saves them as videos for inspection:

  * ``<i>_demo.mp4``       — the demonstration clip (condition).
  * ``<i>_query_gt.mp4``   — the query's ground-truth future (frame0 + real rest).
  * ``<i>_query_gen.mp4``  — the GENERATED query video (frame0 obs + denoised rest).
  * ``<i>_sidebyside.mp4`` — demo | GT | generated, side by side.
  * ``<i>_action.png`` + ``<i>_action.npz`` — generated vs GT query action per dim.

Generation reuses the validated training data path (multi-vision/multi-action,
shared-grid PE) via ``model.generate_samples_from_batch`` (same batch the trainer
sees). Run after a real run produces a DCP checkpoint, e.g.::

    python tools/eval_egoverse_generate.py \\
        --experiment action_policy_egoverse_5s_nano \\
        --checkpoint /mnt/ddn/tmp/cosmos_egoverse_5s/.../checkpoints/iter_000050000 \\
        --chunk-length 148 --num-samples 4 --out-dir /mnt/ddn/tmp/egoverse_gen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn
from cosmos_framework.data.vfm.action.datasets.egoverse_zarr_dataset import (
    EgoVersePairedDataset,
    EgoVersePairedTransform,
    EgoVerseWindowReader,
)
from cosmos_framework.utils.vfm.model_loader import load_model_from_checkpoint

_D = "cosmos_framework/data/vfm/action/datasets"


def _to_cuda(x):
    """Move tensors (incl. nested lists) to CUDA; pass non-tensors through.

    The training dataloader transfers the batch to GPU; offline we collate on CPU,
    so move it ourselves (the model's is_preprocessed video path does not move).
    """
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, list):
        return [_to_cuda(e) for e in x]
    return x


def _to_uint8_thwc(video_cthw: torch.Tensor) -> np.ndarray:
    """(C,T,H,W) in [-1,1] → (T,H,W,3) uint8."""
    v = video_cthw.detach().float().clamp(-1, 1).cpu()
    v = ((v + 1.0) * 127.5).round().to(torch.uint8)  # (C,T,H,W)
    return v.permute(1, 2, 3, 0).numpy()  # (T,H,W,C)


def _save_mp4(frames_thwc: np.ndarray, path: Path, fps: float = 30.0) -> None:
    imageio.mimwrite(str(path), list(frames_thwc), fps=fps, macro_block_size=1)


def _save_action_plot(gen: np.ndarray, gt: np.ndarray, raw_dim: int, path: Path) -> None:
    """Overlay generated vs GT action for the first up-to-8 valid dims."""
    n = min(8, raw_dim)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for d in range(8):
        ax = axes[d // 4, d % 4]
        if d < n:
            ax.plot(gt[:, d], "b-", label="GT", lw=1.5)
            ax.plot(gen[:, d], "r--", label="gen", lw=1.5)
            ax.set_title(f"action dim {d}", fontsize=9)
            if d == 0:
                ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Generated vs GT query action ({raw_dim}D, normalized)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=90)
    plt.close(fig)


# Action57 layout (egoverse hand_pose domain): head9 | Leef9 | Lhand15 | Reef9 | Rhand15.
# Each eef9 block is [Δpos(3), 6D-rot]; dims 9:12 = left-EEF Δxyz, 33:36 = right-EEF Δxyz.
_LEEF, _REEF = slice(9, 12), slice(33, 36)


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """6-D rotation [col0(3), col1(3)] → 3x3 matrix (columns) via Gram-Schmidt."""
    c0, c1 = d6[..., :3], d6[..., 3:6]
    b0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1p = c1 - (b0 * c1).sum(-1, keepdims=True) * b0
    b1 = c1p / (np.linalg.norm(c1p, axis=-1, keepdims=True) + 1e-8)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=-1)                  # (...,3,3), columns = local axes


# Full action57 / state57 layout (hand_pose): head9 | Leef9 | Lhand15 | Reef9 | Rhand15.
# head9/eef9 = [pos(3), 6D-rot(3+3)] (6D = first two rotation-matrix columns); hand15 =
# 5 finger 3-D positions in the WRIST-local frame. _SEG[side] = (pos, rot6d, hand) slices.
_SEG = {
    "head": (slice(0, 3), slice(3, 9), None),
    "L": (slice(9, 12), slice(12, 18), slice(18, 33)),
    "R": (slice(33, 36), slice(36, 42), slice(42, 57)),
}


# Per-embodiment hand-frame correction, keyed (emb, side); default = identity. Was used
# to patch aria's right hand in the viz (its eef rotation came from ee_pose while hand15
# is in the wrist frame, off ~180°). That is now fixed AT THE DATA LEVEL — aria's eef9
# rotation rebuilt from wrist_pose (tools/fix_aria_eef_wrist.py) — so NO viz correction is
# needed; applying one now would RE-flip. Kept empty as a hook for any future rig.
_HAND_CORR: dict[tuple, np.ndarray] = {}


def _fk_skeleton(action: np.ndarray, q01: np.ndarray, q99: np.ndarray, state0: np.ndarray,
                 frame: str = "world", emb: str | None = None) -> dict:
    """De-normalize the full 57-D action and forward-integrate the skeleton, in WORLD (cm).

    The egomimic wrist pose + fingertips are stored in the **head frame** (egocentric),
    while ``state0`` head9 is the head's WORLD pose. So integrate each wrist in head-frame
    — start at its ``state0`` head-frame pose, apply the per-step body-frame eef DELTAS
    ``p_{t+1}=p_t+R_t·Δpos``, ``R_{t+1}=R_t·ΔR`` (validated to reconstruct GT state to
    <0.1°), place the wrist-local ``hand15`` fingertips ``f=p_t+R_t·hand15`` — then map the
    whole head-frame skeleton into WORLD by the head pose ``x_world = head_p + R_head·x``
    so head and hands share one frame (else the hand orientation looks wrong vs the head).
    Returns per-hand wrist path ``P`` ``(T+1,3)``, orientation ``R`` ``(T+1,3,3)``, fingers
    ``F`` ``(T,5,3)`` + head pose, all in world cm."""
    denom = np.clip(q99 - q01, 1e-8, None)
    a = (action[:, :57] + 1.0) / 2.0 * denom + q01          # de-quantile → native units
    T = a.shape[0]
    M = 100.0                                               # m → cm
    hps, hrs = _SEG["head"][0], _SEG["head"][1]             # INTEGRATE the head's WORLD pose over time
    HP = np.zeros((T + 1, 3)); HR = np.zeros((T + 1, 3, 3)) # (head9 deltas) — was pinned at state0 (BUG:
    HP[0] = state0[hps] * M; HR[0] = _rot6d_to_mat(state0[hrs])  # head looked static even when it moves).
    hd = a[:, hps.start : hps.start + 9]                    # head [Δpos(3), Δrot6d(6)]; IDENT9 for robots.
    for t in range(T):
        HP[t + 1] = HP[t] + HR[t] @ (hd[t, :3] * M)
        HR[t + 1] = HR[t] @ _rot6d_to_mat(hd[t, 3:9])
    world = frame == "world"
    # world: head moves along HP/HR (per-frame). head (egocentric): head is the reference → ORIGIN.
    out = {"head_p": HP if world else np.zeros((T + 1, 3)),
           "head_R": HR if world else np.broadcast_to(np.eye(3), (T + 1, 3, 3))}
    for side in ("L", "R"):
        ps, rs, hs = _SEG[side]
        P = np.zeros((T + 1, 3)); R = np.zeros((T + 1, 3, 3)); F = np.zeros((T, 5, 3))
        P[0] = state0[ps] * M                               # head-frame wrist start
        R[0] = _rot6d_to_mat(state0[rs])
        eef = a[:, ps.start : ps.start + 9]                 # [Δpos(3), Δrot6d(6)]
        hand = a[:, hs].reshape(T, 5, 3) * M                # wrist-local finger offsets (cm)
        corr = _HAND_CORR.get((emb, side), np.eye(3))       # per-emb wrist-frame fix
        for t in range(T):
            P[t + 1] = P[t] + R[t] @ (eef[t, :3] * M)
            R[t + 1] = R[t] @ _rot6d_to_mat(eef[t, 3:9])
            F[t] = P[t][None] + (R[t] @ (corr @ hand[t].T)).T
        Rd = R @ corr                                       # corrected orientation for DISPLAY
        # (the shown wrist axes), consistent with the corrected fingers; the trajectory P
        # keeps the raw-frame integration (positions are convention-independent).
        if world:                                           # head-frame → world, PER-FRAME head pose
            out[side + "_P"] = HP + np.einsum("tij,tj->ti", HR, P)
            out[side + "_R"] = np.einsum("tij,tjk->tik", HR, Rd)
            out[side + "_F"] = HP[:-1, None] + np.einsum("tij,tnj->tni", HR[:-1], F)
        else:                                               # egocentric: head-frame directly
            out[side + "_P"], out[side + "_R"], out[side + "_F"] = P, Rd, F
    return out


def _render_skeleton_frames(
    gen: np.ndarray, gt: np.ndarray, q01: np.ndarray, q99: np.ndarray, state0: np.ndarray,
    caption: str, axis_len: float = 4.0, frame: str = "world", emb: str | None = None,
) -> list[np.ndarray]:
    """Render the animated FULL-57-D skeleton (cm) to a LIST of RGB frames (no file write).
    TWO panels: left = GT, right = gen. Each panel shows BOTH hands in the same space:
    head pose + each wrist's pose with RGB orientation axes + 5 wrist-relative fingers +
    growing wrist trajectory. L = blue, R = orange, head = black. Returned so callers can
    either save a standalone mp4 (``_save_skeleton_video``) or stack it under the video row
    into a single combined report (``_save_report``)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3d projection

    Sg = _fk_skeleton(gen, q01, q99, state0, frame=frame, emb=emb)
    St = _fk_skeleton(gt, q01, q99, state0, frame=frame, emb=emb)
    panels = [("GT", St), ("gen", Sg)]
    T = St["L_P"].shape[0]
    step = max(1, (T - 1) // 60)                            # ≤~60 rendered frames (render cost)
    # Shared cube across everything in both panels.
    allp = np.concatenate(
        [S[k] for S in (St, Sg) for k in ("L_P", "R_P")]
        + [S[k].reshape(-1, 3) for S in (St, Sg) for k in ("L_F", "R_F")]
        + [St["head_p"], Sg["head_p"]],
        axis=0,
    )
    ctr = (allp.min(0) + allp.max(0)) / 2.0
    rad = max(float((allp.max(0) - allp.min(0)).max()) / 2.0, 3.0) * 1.1
    HAND = {"L": "tab:blue", "R": "tab:orange"}
    AX = ("r", "g", "b")

    def _draw_pose(ax, p, Rm, length):                      # RGB orientation axes at a joint
        for k in range(3):
            v = Rm[:, k] * length
            ax.plot([p[0], p[0] + v[0]], [p[1], p[1] + v[1]], [p[2], p[2] + v[2]], AX[k], lw=1.6)

    fig = plt.figure(figsize=(11, 5.5))
    axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    frames = []
    for f in range(1, T, step):
        for ax, (name, S) in zip(axes, panels):
            ax.clear()
            # head (MOVING pose + axes, per-frame integrated)
            hp, hR = S["head_p"][f], S["head_R"][f]
            ax.plot(*S["head_p"][: f + 1].T, "-", color="k", lw=1.0, alpha=0.6)  # head trajectory
            ax.scatter(*hp, color="k", s=30)
            _draw_pose(ax, hp, hR, axis_len)
            ax.text(*hp, "head", fontsize=6)
            for side in ("L", "R"):
                P, R, F = S[side + "_P"], S[side + "_R"], S[side + "_F"]
                c = HAND[side]
                ax.plot(*P[: f + 1].T, "-", color=c, lw=1.6, alpha=0.85)   # wrist trajectory
                ax.scatter(*P[f], color=c, s=40)                            # wrist
                _draw_pose(ax, P[f], R[f], axis_len)                        # wrist orientation axes
                ff = min(f, F.shape[0] - 1)
                for j in range(5):                                          # 5 fingers + bones
                    ax.plot([P[f, 0], F[ff, j, 0]], [P[f, 1], F[ff, j, 1]], [P[f, 2], F[ff, j, 2]],
                            "-", color=c, lw=0.8, alpha=0.6)
                ax.scatter(F[ff, :, 0], F[ff, :, 1], F[ff, :, 2], color=c, s=14)
            ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
            ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
            ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
            ax.set_title(name, fontsize=11)
            ax.tick_params(labelsize=6)
            ax.view_init(elev=18, azim=-60)                                # fixed camera (no spin)
        fig.suptitle(f"{caption} — full skeleton (cm, {frame}-frame)  head=black L=blue R=orange  "
                     f"axes=RGB orientation  t={f}/{T - 1}", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames


def _save_skeleton_video(
    gen: np.ndarray, gt: np.ndarray, q01: np.ndarray, q99: np.ndarray, state0: np.ndarray,
    mp4_path: Path, png_path: Path, caption: str, fps: int = 15, axis_len: float = 4.0,
    frame: str = "world", emb: str | None = None,
) -> None:
    """Standalone skeleton mp4 (GT | gen panels) + a final-frame png (back-compat)."""
    frames = _render_skeleton_frames(gen, gt, q01, q99, state0, caption, axis_len, frame, emb)
    imageio.mimwrite(str(mp4_path), frames, fps=fps, macro_block_size=1)
    imageio.imwrite(str(png_path), frames[-1])              # static final-frame reference


def _save_report(
    demo_thwc: np.ndarray, gt_thwc: np.ndarray, gen_thwc: np.ndarray,
    skel_frames: list[np.ndarray], mp4_path: Path, fps: float = 15.0,
) -> None:
    """ONE combined report mp4 (like the cosmos_policy eval): the [demo | GT | gen] video row
    STACKED ABOVE the [GT | gen] 3D-skeleton panels. Both streams are resampled to the same
    frame count (the skeleton's ~60) and width-matched, so a single file shows the generated
    video and the generated motion skeleton together."""
    import cv2

    n = len(skel_frames)
    if n == 0:
        return
    T = min(demo_thwc.shape[0], gt_thwc.shape[0], gen_thwc.shape[0])
    vid_idx = np.linspace(0, T - 1, n).round().astype(int)          # resample video row to n frames
    row = np.concatenate([demo_thwc[vid_idx], gt_thwc[vid_idx], gen_thwc[vid_idx]], axis=2)  # (n,H,3W,3)
    skel = np.stack(skel_frames, axis=0)                             # (n,Hs,Ws,3)
    Wt = max(row.shape[2], skel.shape[2])
    Wt -= Wt % 2                                                     # libx264/yuv420p needs even W
    def _wfit(a):
        h = int(a.shape[1] * Wt / a.shape[2]); h -= h % 2            # ... and even H
        out = np.zeros((a.shape[0], h, Wt, 3), a.dtype)
        for k in range(a.shape[0]):
            out[k] = cv2.resize(a[k], (Wt, h), interpolation=cv2.INTER_AREA)
        return out
    row_r, skel_r = _wfit(row), _wfit(skel)
    report = [np.concatenate([row_r[k], skel_r[k]], axis=0) for k in range(n)]  # stack vertically
    H = report[0].shape[0] - report[0].shape[0] % 2                  # even total H
    imageio.mimwrite(str(mp4_path), [f[:H] for f in report], fps=fps, macro_block_size=1)


def _win_task(reader, widx: int, _cache: dict = {}) -> str:
    """Episode ``task_description`` for a window index (cached per episode)."""
    from cosmos_framework.data.vfm.action.datasets.egoverse_zarr_dataset import EMB_REGISTRY, _episode_attrs
    ep = str(reader._ep[int(widx)])
    if ep not in _cache:
        spec = EMB_REGISTRY[str(reader._emb[int(widx)])]
        d = str(reader._data_root / spec.root / ep)
        _cache[ep] = str(dict(_episode_attrs(d)).get("task_description") or "")
    return _cache[ep]


def _build_specs(reader, ds, args) -> list[tuple]:
    """[(label, q_widx, c_widx|None, ds_idx|None)]. Default = spread path (dataset samples
    the condition). With --cross-emb / --query-emb / --demo-emb / --task-contains, build
    EXPLICIT (query, demo) pairs where demo & query are the chosen embodiments on the same
    matched task (the v8 matches are already same-task-class), capped per task for variety."""
    # Explicit pair ROWS (match cosmos_policy's dataset[K]: query=query_idx[K], demo=a match of K).
    # Lets us render the SAME query windows as the cosmos_policy viz for a fair 2B-vs-8B compare.
    # Item syntax "K" or "K:demoEmb": the optional demo embodiment picks the FIRST stored match
    # of that emb (2.5's dataset draws a RANDOM match at runtime, so pinning the emb reproduces
    # the demo embodiment the recorded 2.5 viz happened to use). Labels echo the 2.5 filename
    # convention: idx{K}_{demoEmb}2{queryEmb}.
    if getattr(args, "pairs_rows", None):
        specs = []
        for item in str(args.pairs_rows).split(","):
            if item == "":
                continue
            ks, _, demb = item.partition(":")
            K = int(ks)
            q = int(ds._query_idx[K % len(ds._query_idx)])
            ms = [int(m) for m in ds._match_idx[K % len(ds._match_idx)] if m >= 0]
            c = next((m for m in ms if demb and reader.emb_of(m) == demb),
                     ms[0] if ms else q)                 # pinned-emb match, else best match
            specs.append((f"idx{K}_{reader.emb_of(c)}2{reader.emb_of(q)}", q, c, None))
        return specs
    filtered = bool(args.cross_emb or args.query_emb or args.demo_emb or args.task_contains)
    if not filtered:
        if args.sample_idx is not None:
            idxs = [args.sample_idx + k for k in range(args.num_samples)]
        else:
            idxs = list(np.linspace(0, len(ds) - 1, args.num_samples).astype(int))
        return [(str(i), int(ds._query_idx[ds._order[int(i)]]), None, int(i)) for i in idxs]

    subs = [s.strip().lower() for s in args.task_contains.split(",")] if args.task_contains else None
    from collections import defaultdict

    cnt: dict = defaultdict(int)
    dcnt: dict = defaultdict(int)                            # per emb-direction (cemb2qemb)
    specs: list[tuple] = []
    for row in ds._order:
        row = int(row)
        q = int(ds._query_idx[row])
        qemb = str(reader._emb[q])
        if args.query_emb and qemb != args.query_emb:
            continue
        qtask = _win_task(reader, q)
        if subs and not any(s in qtask.lower() for s in subs):
            continue
        if cnt[qtask] >= args.per_task:                     # don't repeat one task too much
            continue
        for c in ds._match_idx[row]:
            c = int(c)
            if c < 0:
                continue
            cemb = str(reader._emb[c])
            if args.demo_emb and cemb != args.demo_emb:
                continue
            if args.cross_emb and cemb == qemb:
                continue
            if subs and not any(s in _win_task(reader, c).lower() for s in subs):
                continue
            direction = f"{cemb}2{qemb}"
            if dcnt[direction] >= args.per_dir:             # balance across emb directions
                continue
            specs.append((f"{direction}_{q}", q, c, None))  # demo_emb 2 query_emb
            cnt[qtask] += 1
            dcnt[direction] += 1
            break
        if len(specs) >= args.num_samples:
            break
    if not specs:
        print("WARNING: no (query, demo) pairs matched the filters")
    else:
        print(f"selected {len(specs)} pairs across {len(cnt)} tasks:")
        for lbl, q, _c, _ in specs:
            print(f"  {lbl}  task={_win_task(reader, q)[:60]!r}")
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="action_policy_egoverse_5s_nano")
    ap.add_argument("--checkpoint", required=True, help="Trained DCP checkpoint dir (e.g. .../iter_000050000)")
    ap.add_argument("--data-root", default="/mnt/ddn/dataset")
    ap.add_argument("--pairs-dir", default="/mnt/ddn/dataset/egoverse_pairs_v9_test")  # offline eval = held-out TEST split only
    ap.add_argument("--norm", default=f"{_D}/egoverse_normalization.json")
    ap.add_argument("--speed", default=f"{_D}/egoverse_speed_stats.json")
    ap.add_argument("--chunk-length", type=int, default=148, help="148 for 5s, 28 for 1s (match the trained config)")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--sample-idx", type=int, default=None, help="Fixed start index; default = spread across dataset")
    ap.add_argument("--pairs-rows", default=None,
                    help="comma-sep pair ROW indices 'K' or 'K:demoEmb' → render query_idx[K] + its "
                         "best (or pinned-emb) stored match (matches cosmos_policy's dataset[K]; "
                         "use to render the SAME query windows as the 2.5 viz)")
    ap.add_argument("--report-only", action="store_true",
                    help="save ONLY {label}_report.mp4 + {label}_action.npz (skip demo/query_gt/"
                         "query_gen/sidebyside/skeleton outputs) — mirrors the 2.5 eval-viz output set")
    # Cross-embodiment / task filters: select explicit (query, demo) pairs where the demo
    # and query are DIFFERENT embodiments doing the same (matched) task. Any of these
    # switches the run from the default spread path to the filtered-pair path.
    ap.add_argument("--cross-emb", action="store_true", help="only pairs where demo emb != query emb")
    ap.add_argument("--query-emb", default=None, help="pin the query (target) embodiment, e.g. aria")
    ap.add_argument("--demo-emb", default=None, help="pin the demo (condition) embodiment, e.g. scale")
    ap.add_argument("--task-contains", default=None,
                    help="comma-separated keywords; keep only tasks whose description contains ANY (e.g. 'cup,dish,plate')")
    ap.add_argument("--per-task", type=int, default=2, help="max pairs per distinct task (variety cap)")
    ap.add_argument("--per-dir", type=int, default=4,
                    help="max pairs per embodiment direction (e.g. scale2aria) — balances cross-emb directions")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=3.0)  # 256-res training shift
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/mnt/ddn/tmp/egoverse_gen")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frame", default="world", choices=["world", "head"],
                    help="skeleton frame: world (head at its world pose) or head (egocentric, head at origin)")
    ap.add_argument("--vae-path", default="/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load the trained model (single GPU, eval) ---------------------------
    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint,
        parallelism_config={"data_parallel_shard_degree": 1, "data_parallel_replicate_degree": 1},
        # The experiment's default VAE path is a checkpoint-db URI (the TOML normally
        # overrides it via $WAN_VAE_PATH); point it at the local file for offline eval.
        experiment_opts=[f"model.config.tokenizer.vae_path={args.vae_path}"],
    )
    model = model.cuda().eval()

    tokenizer_config = config.model.config.vlm_config.tokenizer  # same text tokenizer as training

    reader = EgoVerseWindowReader(
        data_root=args.data_root, pairs_dir=args.pairs_dir,
        normalization_path=args.norm, speed_stats_path=args.speed,
        chunk_length=args.chunk_length,
    )
    transform = EgoVersePairedTransform(
        resolution="256", max_action_dim=config.model.config.max_action_dim,
        share_temporal=True, keep_aspect_ratio=False, tokenizer_config=tokenizer_config,
    )
    ds = EgoVersePairedDataset(reader=reader, pairs_dir=args.pairs_dir, transform=transform)

    specs = _build_specs(reader, ds, args)

    for label, q_widx, c_widx, ds_idx in specs:
        if c_widx is None:                              # default: dataset samples the condition
            sample = ds[int(ds_idx)]
        else:                                           # filtered: explicit (query, demo) pair
            sample = transform({"query": reader.read_window(int(q_widx)),
                                "condition": reader.read_window(int(c_widx))})
        data_batch = {k: _to_cuda(v) for k, v in custom_collate_fn([sample]).items()}
        # custom_collate nests image_size as [[item0_size, item1_size]] per sample,
        # but the model indexes it per FLAT vision item (like x0_tokens_vision).
        # Flatten so it lines up (the PackingDataLoader does this in training).
        if isinstance(data_batch.get("image_size"), list):
            data_batch["image_size"] = [
                it for samp in data_batch["image_size"] for it in (samp if isinstance(samp, list) else [samp])
            ]

        with torch.no_grad():
            result = model.generate_samples_from_batch(
                data_batch,
                guidance=args.guidance,
                seed=[args.seed],
                n_sample=1,
                num_steps=args.num_steps,
                shift=args.shift,
            )

        # Last vision item / action segment = the QUERY (demo items come first).
        gen_query_latent = result["vision"][-1]
        gen_video = model.decode(gen_query_latent)  # [B,C,T,H,W] in [-1,1]
        gen_thwc = _to_uint8_thwc(gen_video[0])

        demo_thwc = _to_uint8_thwc(sample["video"][0])      # condition (already [-1,1])
        gt_thwc = _to_uint8_thwc(sample["video"][1])        # query GT

        if not args.report_only:
            _save_mp4(demo_thwc, out / f"{label}_demo.mp4", args.fps)
            _save_mp4(gt_thwc, out / f"{label}_query_gt.mp4", args.fps)
            _save_mp4(gen_thwc, out / f"{label}_query_gen.mp4", args.fps)
            # Side-by-side (pad to equal T by clipping to the shortest).
            T = min(demo_thwc.shape[0], gt_thwc.shape[0], gen_thwc.shape[0])
            side = np.concatenate([demo_thwc[:T], gt_thwc[:T], gen_thwc[:T]], axis=2)  # concat on W
            _save_mp4(side, out / f"{label}_sidebyside.mp4", args.fps)

        # Action: generated vs GT query action (normalized space).
        if result.get("action"):
            gen_action = result["action"][-1].detach().float().cpu().numpy()
            gt_action = sample["action"][1].detach().float().cpu().numpy()  # padded to max_action_dim
            raw_dim = int(sample["raw_action_dim"][1].item())
            if raw_dim == 57:
                # hand_pose rigs: full-57-D skeleton (head + both wrists w/ orientation
                # axes + 5 fingers each), GT panel | gen panel. state0 = real start pose.
                q_emb = reader.emb_of(int(q_widx))
                qs = reader._stats[q_emb]
                try:
                    state0 = reader.read_state0(int(q_widx))
                except Exception as e:  # noqa: BLE001 — no state{dim} zarr → anchor at origin
                    print(f"[{label}] read_state0 failed ({e}); anchoring skeleton at origin")
                    state0 = np.zeros(57, np.float32)
                np.savez(out / f"{label}_action.npz", gen=gen_action, gt=gt_action, raw_dim=raw_dim, state0=state0)
                # Render the skeleton frames ONCE, then write both the standalone skeleton
                # mp4/png AND the combined report (video row + skeleton, one file).
                skel_frames = _render_skeleton_frames(
                    gen_action, gt_action, qs["q01"].numpy(), qs["q99"].numpy(), state0,
                    caption=f"[{label}]", frame=args.frame, emb=q_emb,
                )
                # REAL-TIME playback: query-native fps (mimicdroid 20 Hz, others 30 Hz).
                # fps=15 made 3 s clips play ~5.9 s (recurring bug — keep native).
                _qfps = 20.0 if "mimicdroid" in str(q_emb) else 30.0
                if not args.report_only:
                    imageio.mimwrite(str(out / f"{label}_skeleton.mp4"), skel_frames, fps=_qfps, macro_block_size=1)
                    imageio.imwrite(str(out / f"{label}_skeleton.png"), skel_frames[-1])
                _save_report(demo_thwc, gt_thwc, gen_thwc, skel_frames,
                             out / f"{label}_report.mp4", fps=_qfps)
            else:
                np.savez(out / f"{label}_action.npz", gen=gen_action, gt=gt_action, raw_dim=raw_dim)
                _save_action_plot(gen_action, gt_action, raw_dim, out / f"{label}_action.png")

        print(f"[{label}] saved  (caption: {sample.get('ai_caption','')[:80]})")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()

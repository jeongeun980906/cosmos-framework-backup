# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Re-base aria's ``eef9`` ROTATION from ``ee_pose`` → ``wrist_pose`` (keep ee position).

Root cause (value-verified): aria's ``obs_ee_pose`` and ``obs_wrist_pose`` are DIFFERENT
rotation conventions (angle 93° left / 177° right), but our action57/state57 ``eef9``
rotation was built from ``ee_pose`` while ``hand15`` (fingertips) lives in the ``wrist``
frame — so the right hand renders/encodes flipped. MANO rigs (MECKA/scale) have
ee==wrist so they're fine; ALL rigs use the ee POSITION for eef9. Fix → make aria match:
keep the ee position, swap the eef rotation to ``wrist_pose``, and (since the action eef
Δpos is a BODY-frame delta ``R_t^T·Δp``) recompute Δpos in the new wrist-rotation frame.

  state57 eef:  pos = ee_pos (unchanged);  rot6d = [R_wrist[:,0], R_wrist[:,1]]
  action57 eef: Δpos = R_wrist_t^T (ee_pos_{t+1}-ee_pos_t);  Δrot6d = (R_wrist_t^T R_wrist_{t+1})[:, :2]
  head9 / hand15 / valid: unchanged

head9|Leef9|Lhand15|Reef9|Rhand15 → eef dims L=(9:18), R=(33:42); within eef9 = [Δpos(3),
6D(3+3)]. Idempotent (skips episodes already flagged ``aria_eef_wrist_fixed``). Reversible
(the raw ``*.obs_{ee,wrist}_pose_transformed`` keys stay in the zarr). Default = DRY-RUN
(compute + self-check reconstruction, write nothing); pass ``--apply`` to write back.

Usage::

    python tools/fix_aria_eef_wrist.py                 # dry-run all aria eps (verify)
    python tools/fix_aria_eef_wrist.py --limit 1       # one episode
    python tools/fix_aria_eef_wrist.py --apply         # write back
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

ROOT = "/mnt/ddn/dataset/egoverse"
# eef9 layout in the 57-vector: (pos slice, rot6d slice, raw-key prefix)
EEF = {"L": (slice(9, 12), slice(12, 18), "left"), "R": (slice(33, 36), slice(36, 42), "right")}


def _Rwxyz(p7: np.ndarray) -> np.ndarray:
    """(N,7)=pos+quat(wxyz) → (N,3,3) rotation matrices."""
    q = p7[:, 3:7][:, [1, 2, 3, 0]]  # wxyz → xyzw
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    return R.from_quat(q).as_matrix()


def _ang(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def _rot6_to_R(s6: np.ndarray) -> np.ndarray:
    c0 = s6[:3] / (np.linalg.norm(s6[:3]) + 1e-9)
    c1 = s6[3:6] - (c0 @ s6[3:6]) * c0
    c1 = c1 / (np.linalg.norm(c1) + 1e-9)
    return np.stack([c0, c1, np.cross(c0, c1)], 1)


def fix_episode(ep_dir: str, apply: bool) -> str:
    g = zarr.open_group(ep_dir, mode="r+" if apply else "r")
    emb = dict(g.attrs).get("embodiment") or ""
    if not emb.startswith("aria"):          # aria_bimanual / aria_right_arm (ee≠wrist convention)
        return "skip (not aria)"
    if dict(g.attrs).get("aria_eef_wrist_fixed"):
        return "skip (already fixed)"

    state = np.asarray(g["state57/state"][:], np.float64)   # (Ts, 57)
    action = np.asarray(g["action57/action"][:], np.float64)  # (Ta=Ts-1, 57)
    Ts = state.shape[0]
    new_state, new_action = state.copy(), action.copy()
    report = []
    for side, (psl, rsl, pre) in EEF.items():
        if f"{pre}.obs_wrist_pose_transformed" not in g:   # single-arm variant lacks this hand
            report.append(f"{side} skip(no wrist key)")
            continue
        wr = _Rwxyz(np.asarray(g[f"{pre}.obs_wrist_pose_transformed"][:Ts], np.float64))  # (Ts,3,3)
        ee_pos = np.asarray(g[f"{pre}.obs_ee_pose_transformed"][:Ts, :3], np.float64)      # (Ts,3) == state pos
        # state: keep pos, swap rot6d → wrist columns 0,1
        new_state[:, rsl] = np.concatenate([wr[:, :, 0], wr[:, :, 1]], axis=1)
        # action deltas (Ta = Ts-1) in the wrist body frame, of the ee position
        dpos = np.einsum("tji,tj->ti", wr[:-1], ee_pos[1:] - ee_pos[:-1])  # R_t^T Δp
        dRm = np.einsum("tji,tjk->tik", wr[:-1], wr[1:])                   # R_t^T R_{t+1}
        Ta = new_action.shape[0]
        new_action[:, psl] = dpos[:Ta]
        new_action[:, rsl] = np.concatenate([dRm[:, :, 0], dRm[:, :, 1]], axis=1)[:Ta]
        # self-check: integrate Δ from state0 → matches new_state rotation
        Rint = _rot6_to_R(new_state[0, rsl])
        errs = []
        for t in range(min(120, Ta)):
            Rint = Rint @ _rot6_to_R(new_action[t, rsl])
            errs.append(_ang(Rint, _rot6_to_R(new_state[t + 1, rsl])))
        report.append(f"{side} recon_rot_err={np.mean(errs):.3f}deg")

    if apply:
        g["state57/state"][:] = new_state.astype(g["state57/state"].dtype)
        g["action57/action"][:] = new_action.astype(g["action57/action"].dtype)
        g.attrs["aria_eef_wrist_fixed"] = True
        g.attrs["aria_eef_source"] = "wrist_rot+ee_pos"
    return ("APPLIED " if apply else "dry-run ") + " ".join(report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--apply", action="store_true", help="write back (default = dry-run verify only)")
    ap.add_argument("--limit", type=int, default=0, help="only N aria episodes (0=all)")
    args = ap.parse_args()

    eps = sorted(d for d in os.listdir(args.root) if os.path.isdir(f"{args.root}/{d}"))
    done = 0
    for ep in eps:
        d = f"{args.root}/{ep}"
        try:
            emb = dict(zarr.open_group(d, mode="r").attrs).get("embodiment") or ""
        except Exception:
            continue
        if not emb.startswith("aria"):
            continue
        msg = fix_episode(d, args.apply)
        print(f"{ep}: {msg}", flush=True)
        if not msg.startswith("skip"):
            done += 1
        if args.limit and done >= args.limit:
            break
    print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'} on {done} aria episode(s)")


if __name__ == "__main__":
    main()

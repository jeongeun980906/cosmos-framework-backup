# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Move EgoVerse's domain-3 head onto the channels a 7-D delta-EE arm actually uses.

Pointing RoboCasa at DomainAwareLinear row 3 reuses a head EgoVerse trained -- but on
the WRONG channels. From ``docs/action57_state57_convention.md`` the 57-D layout is::

    head9(0:9) | Leef9(9:18) | Lhand15(18:33) | Reef9(33:42) | Rhand15(42:57)
    eef9 = [ Dpos(3), DR[:,0](3), DR[:,1](3) ]        # body-frame delta
    robot head9 (action57) = constant IDENT9          # zero-motion identity delta
    robot hand15(g) = (1-g)*OPEN15 + g*CLOSED15       # gripper is a 1-D lerp

So channels 0..6 -- the ones a 7-D embodiment reads -- are HEAD pose, which for every
robot episode in EgoVerse was a CONSTANT. The weights that learned real end-effector
motion sit at 33..41 (right eef delta) and the gripper lives at 42..56 as a scalar
travelling along ``CLOSED15 - OPEN15``. RoboCasa's own 7-D action is
``[dpos(3), daxang(3), gripper]`` -- the same physical quantities as Reef9 + grip.

This rewrites row 3 so those channels land where RoboCasa reads them:

    out 0..2  <- 33,34,35   right-eef delta position
    out 3..5  <- 36,37,38   right-eef delta rotation, first column of DR
    out 6     <- projection of 42..56 onto the grip axis, rescaled to [-1,1]

``llm2action`` (H x A, output channels are columns) is permuted column-wise;
``action2llm`` (A x H, input channels are rows) row-wise, with the gripper row folded
through the same lerp so feeding a scalar reproduces what feeding hand15 would. The
constant blocks EgoVerse always saw (head9 = IDENT9, and the left arm, which RoboCasa
does not have) are folded into the biases so the layer's operating point is preserved
instead of being silently dropped.

Rotation caveat, stated rather than hidden: Reef9 stores ``DR[:,0]`` (a rotation-matrix
column) while RoboCasa sends an axis-angle delta. Both are 3-vectors describing the same
rotation and share scale and sign conventions near identity, but they are not the same
parameterisation -- channels 3..5 start closer than head pose did, not exact.

Usage::

    python tools/remap_domain3_head_dcp.py --src <EgoVerse DCP model dir> \\
        --out <new DCP model dir> [--domain 3] [--match-rms]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed.checkpoint as dcp

# docs/action57_state57_convention.md 3d / tools/convert_taskdemos.py
IDENT9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
OPEN15 = np.array([0.0005, 0.0958, -0.034, 0.0285, 0.0222, -0.1263, 0.0085, 0.0217, -0.1285,
                   -0.0116, 0.0222, -0.1282, -0.0314, 0.0229, -0.1259], dtype=np.float64)
CLOSED15 = np.array([0.0052, 0.0775, -0.0701, 0.0266, 0.044, -0.0916, 0.0085, 0.0441, -0.1016,
                     -0.0097, 0.044, -0.0937, -0.0278, 0.044, -0.0919], dtype=np.float64)
GRIP_AXIS = CLOSED15 - OPEN15
REEF = 33          # Reef9 start
RHAND = 42         # Rhand15 start


def _rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--domain", type=int, default=3)
    ap.add_argument("--action-dim", type=int, default=64)
    ap.add_argument("--raw-action-dim", type=int, default=7)
    ap.add_argument("--match-rms", action="store_true",
                    help="After remapping, rescale channels 0..6 to the RMS of the row they "
                         "replace, so the gradient norm starts where the untouched head's did.")
    args = ap.parse_args()

    A, R, d = args.action_dim, args.raw_action_dim, args.domain
    axis = torch.tensor(GRIP_AXIS, dtype=torch.float64)
    axis_n = axis / float(axis @ axis)                       # inverse-lerp direction
    open15 = torch.tensor(OPEN15, dtype=torch.float64)
    ident9 = torch.tensor(IDENT9, dtype=torch.float64)

    md = dcp.FileSystemReader(args.src).read_metadata().state_dict_metadata
    sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype)
          for k in md if hasattr(md[k], "size")}
    print(f"loading {len(sd)} tensors from {args.src} ...")
    dcp.load(sd, storage_reader=dcp.FileSystemReader(args.src), no_dist=True)

    for prefix in ("net.", "net_ema."):
        a2l_w, l2a_w = f"{prefix}action2llm.fc.weight", f"{prefix}llm2action.fc.weight"
        a2l_b, l2a_b = f"{prefix}action2llm.bias.weight", f"{prefix}llm2action.bias.weight"
        if a2l_w not in sd:
            continue
        H = sd[a2l_w].shape[1] // A
        print(f"\n=== {prefix}  hidden={H} ===")

        # ---- llm2action: (H, A); output channels are COLUMNS -------------------
        W = sd[l2a_w][d].double().view(H, A).clone()
        b = sd[l2a_b][d].double().clone()
        newW, newb = torch.zeros_like(W), torch.zeros_like(b)
        newW[:, 0:3] = W[:, REEF + 0:REEF + 3]
        newW[:, 3:6] = W[:, REEF + 3:REEF + 6]
        newb[0:6] = b[REEF:REEF + 6]
        # gripper: g = (out_hand15 - OPEN15) . axis / |axis|^2, then to [-1,1]
        newW[:, 6] = 2.0 * (W[:, RHAND:RHAND + 15] @ axis_n)
        newb[6] = 2.0 * float((b[RHAND:RHAND + 15] - open15) @ axis_n) - 1.0
        old_rms = _rms(W[:, :R])
        if args.match_rms:
            cur = _rms(newW[:, :R])
            if cur > 0:
                newW[:, :R] *= old_rms / cur
        sd[l2a_w][d] = newW.reshape(-1).to(sd[l2a_w].dtype)
        sd[l2a_b][d] = newb.to(sd[l2a_b].dtype)
        print(f"  llm2action cols 0..6  RMS {old_rms:.5f} -> {_rms(newW[:, :R]):.5f}"
              f"   (was head9, now Reef9+grip)")

        # ---- action2llm: (A, H); input channels are ROWS ------------------------
        V = sd[a2l_w][d].double().view(A, H).clone()
        vb = sd[a2l_b][d].double().clone()
        newV = torch.zeros_like(V)
        newV[0:3] = V[REEF + 0:REEF + 3]
        newV[3:6] = V[REEF + 3:REEF + 6]
        # hand15 = OPEN15 + g*axis with g = (a6+1)/2  ->  the a6-dependent part
        newV[6] = 0.5 * (axis @ V[RHAND:RHAND + 15])
        # Fold every block EgoVerse always fed but RoboCasa never will into the bias,
        # so the layer keeps the operating point it was trained at.
        const = (ident9 @ V[0:9]) + (open15 @ V[RHAND:RHAND + 15]) + 0.5 * (axis @ V[RHAND:RHAND + 15])
        old_rms_a = _rms(V[:R])
        if args.match_rms:
            cur = _rms(newV[:R])
            if cur > 0:
                newV[:R] *= old_rms_a / cur
        sd[a2l_w][d] = newV.reshape(-1).to(sd[a2l_w].dtype)
        sd[a2l_b][d] = (vb + const).to(sd[a2l_b].dtype)
        print(f"  action2llm rows 0..6  RMS {old_rms_a:.5f} -> {_rms(newV[:R]):.5f}")
        print(f"  folded constants into action2llm bias: |delta| mean {float(const.abs().mean()):.5f}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting to {args.out} ...")
    dcp.save(sd, storage_writer=dcp.FileSystemWriter(args.out), no_dist=True)
    cfg = args.src / "config.json"
    if cfg.is_file():
        shutil.copy2(cfg, args.out / "config.json")
    print("done")


if __name__ == "__main__":
    main()

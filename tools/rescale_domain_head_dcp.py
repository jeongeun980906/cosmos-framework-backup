# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rescale one domain's action-head rows so their magnitude matches another domain's.

Motivating case (RoboCasa c3edge ``dom3``): EgoVerse trained ``DomainAwareLinear``
row 3 (``hand_pose``), so pointing RoboCasa at row 3 starts from a head somebody
actually trained instead of row 5, which is untrained in the public Edge base
(``llm2action.bias[5]`` is exactly 0). Measured, that start IS better on the action
loss -- 0.9046 vs base's 1.1514 at 10 iters, -21%.

But row 3's output is 4.03x larger (std 1.948 vs 0.484 pushing h~N(0,1) through the
two rows), and that inflates the gradient: pre-clip global grad norm 89.0 vs base's
16.8, consistently 4.1-6.9x over 10 steps. With ``grad_clip.clip_norm = 1.0``
applied as ONE global scalar across every parameter (see callbacks/grad_clip.py --
"every mesh group is rescaled with that single global scalar"), the whole step
budget gets renormalised, and on the cluster dom3 fell behind base after ~1200
iters instead of converging.

This rescales row 3's weights to row 5's RMS. Direction -- the learned channel
structure that produced the lower loss -- is preserved exactly; only magnitude
changes. Both slices are scaled where they actually matter:

  * ``action2llm``  W = fc[d].view(A=64, H=2048): rows 0..6 are the only ones a
    7-dim embodiment multiplies (7..63 meet zero padding), so scale W[:7, :].
  * ``llm2action``  W = fc[d].view(H=2048, A=64): columns 0..6 are the only ones the
    loss reads (``flow_matching.py`` slices ``sqerr[:, :raw_action_dim]``), so scale
    W[:, :7]. Its bias is an offset in the same output space, so it scales with it.

Usage::

    python tools/rescale_domain_head_dcp.py \\
        --src <EgoVerse DCP model dir> --out <new DCP model dir> \\
        --domain 3 --target-domain 5 --raw-action-dim 7
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

A2L = "action2llm"
L2A = "llm2action"


def _rms(t: torch.Tensor) -> float:
    return t.float().pow(2).mean().sqrt().item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="DCP model dir to read.")
    ap.add_argument("--out", required=True, type=Path, help="Output DCP model dir (created).")
    ap.add_argument("--domain", type=int, default=3, help="Domain row to rescale.")
    ap.add_argument("--target-domain", type=int, default=5, help="Domain row whose RMS to match.")
    ap.add_argument("--target-offset", type=int, default=None,
                    help="Window start in the TARGET row (defaults to 0 — domain 5's used "
                         "channels are always the 0..R-1 prefix).")
    ap.add_argument("--raw-action-dim", type=int, default=7, help="Channels the embodiment actually uses.")
    ap.add_argument("--offset", type=int, default=0,
                    help="Start of the channel window to rescale. 0 is the historical "
                         "prefix layout; 33 is Reef9, the right end-effector delta block, "
                         "which is what a 7-D arm reads once the dataset places its "
                         "channels there (SW_OFFSET / action_channel_offset).")
    ap.add_argument("--action-dim", type=int, default=64, help="Padded action width (max_action_dim).")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="How far to move toward the target RMS, in log space. 1.0 lands "
                         "exactly on it; 0.5 goes halfway — a milder correction that keeps "
                         "more of the trained head's scale.")
    args = ap.parse_args()

    md = dcp.FileSystemReader(args.src).read_metadata().state_dict_metadata
    sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in md if hasattr(md[k], "size")}
    print(f"loading {len(sd)} tensors from {args.src} ...")
    dcp.load(sd, storage_reader=dcp.FileSystemReader(args.src), no_dist=True)

    d, t, R, A = args.domain, args.target_domain, args.raw_action_dim, args.action_dim
    # Window to rescale in the SOURCE row, and the window in the TARGET row whose RMS
    # is being matched. Offset 33 is Reef9 (right end-effector delta); the target row
    # (domain 5) always uses the 0..R-1 prefix.
    O = args.offset
    T = args.target_offset if args.target_offset is not None else 0
    touched = []
    for prefix in ("net.", "net_ema."):
        a2l_w, l2a_w = f"{prefix}{A2L}.fc.weight", f"{prefix}{L2A}.fc.weight"
        l2a_b = f"{prefix}{L2A}.bias.weight"
        if a2l_w not in sd:
            continue
        H = sd[a2l_w].shape[1] // A

        # action2llm: (A, H); rows 0..R are the ones a 7-dim action multiplies.
        w = sd[a2l_w][d].float().view(A, H)
        cur = _rms(w[O : O + R, :])
        tgt = _rms(sd[a2l_w][t].float().view(A, H)[T : T + R, :])
        s_a = (tgt / cur) ** args.strength
        w[O : O + R, :] *= s_a
        sd[a2l_w][d] = w.reshape(-1).to(sd[a2l_w].dtype)
        touched.append((a2l_w, cur, _rms(w[O : O + R, :]), s_a))

        # llm2action: (H, A); columns 0..R are the ones the loss reads.
        w = sd[l2a_w][d].float().view(H, A)
        cur = _rms(w[:, O : O + R])
        tgt = _rms(sd[l2a_w][t].float().view(H, A)[:, T : T + R])
        s_l = (tgt / cur) ** args.strength
        w[:, O : O + R] *= s_l
        sd[l2a_w][d] = w.reshape(-1).to(sd[l2a_w].dtype)
        touched.append((l2a_w, cur, _rms(w[:, O : O + R]), s_l))

        # The output bias lives in the same space as llm2action's output.
        if l2a_b in sd:
            before = sd[l2a_b][d][O : O + R].float().abs().mean().item()
            sd[l2a_b][d][O : O + R] = (sd[l2a_b][d][O : O + R].float() * s_l).to(sd[l2a_b].dtype)
            touched.append((l2a_b, before, sd[l2a_b][d][O : O + R].float().abs().mean().item(), s_l))

    if not touched:
        raise SystemExit(f"no {A2L}/{L2A} tensors found in {args.src}")
    print(f"\nrescaling domain {d} channels {O}..{O + R - 1} to match domain {t} channels {T}..{T + R - 1}:")
    for k, before, after, s in touched:
        print(f"  {k:<40} x{s:.4f}   {before:.5f} -> {after:.5f}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting to {args.out} ...")
    dcp.save(sd, storage_writer=dcp.FileSystemWriter(args.out), no_dist=True)
    src_cfg = args.src / "config.json"
    if src_cfg.is_file():
        shutil.copy2(src_cfg, args.out / "config.json")
        print(f"copied {src_cfg.name}")
    print("done")


if __name__ == "__main__":
    main()

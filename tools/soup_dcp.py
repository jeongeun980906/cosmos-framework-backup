# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Linearly interpolate two DCP checkpoints: ``out = alpha * A + (1 - alpha) * B``.

Built for the one question the RoboCasa arms keep running into: the EgoVerse warm start
("mid") begins post-train ABOVE the public base on action loss, and every intervention
that lowers its starting point has so far flattened its descent. Interpolating the two
checkpoints attacks the starting point directly -- ``alpha = 0`` is base exactly,
``alpha = 1`` is mid, and in between the run trades EgoVerse's representation against
the base's calibration. This is WiSE-FT's observation applied to a warm start rather
than to a fine-tune: the two endpoints sit in the same loss basin (mid IS base, trained
further), so the segment between them is well behaved rather than a random chord.

Evidence they are in the same basin, measured on domain 5's action-head row, which
neither run ever indexes: ``cos(base, mid) = 0.99999`` on both projections, with mid's
row a uniform 0.887x of base's -- weight decay acting on an untouched parameter. The
weights moved, but not in direction.

Tensors present in only one checkpoint, and non-floating-point tensors (step counters,
etc.), are copied from A unchanged.

Usage::

    python tools/soup_dcp.py --a <mid-dcp> --b <base-dcp> --alpha 0.5 --out <dcp>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def _load(path: Path) -> dict:
    md = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype)
          for k in md if hasattr(md[k], "size")}
    dcp.load(sd, storage_reader=dcp.FileSystemReader(path), no_dist=True)
    return sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path, help="Checkpoint weighted by alpha.")
    ap.add_argument("--b", required=True, type=Path, help="Checkpoint weighted by 1-alpha.")
    ap.add_argument("--alpha", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    al = args.alpha

    print(f"loading A {args.a} ...")
    A = _load(args.a)
    print(f"loading B {args.b} ...")
    B = _load(args.b)

    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    if only_a:
        print(f"  {len(only_a)} tensors only in A (kept as-is), e.g. {only_a[:3]}")
    if only_b:
        print(f"  {len(only_b)} tensors only in B (ignored), e.g. {only_b[:3]}")

    mixed = skipped = 0
    drift = []
    for k, ta in A.items():
        tb = B.get(k)
        if tb is None or tb.shape != ta.shape or not ta.is_floating_point():
            skipped += 1
            continue
        fa, fb = ta.double(), tb.double()
        if fa.numel() > 1:
            na, nb = fa.norm(), fb.norm()
            if na > 0 and nb > 0:
                drift.append(float(torch.dot(fa.flatten(), fb.flatten()) / (na * nb)))
        A[k] = (al * fa + (1 - al) * fb).to(ta.dtype)
        mixed += 1

    drift.sort()
    n = len(drift)
    print(f"\ninterpolated {mixed} tensors at alpha={al} ({skipped} copied from A unchanged)")
    if n:
        print(f"cos(A,B) per tensor:  min {drift[0]:.6f}   p05 {drift[n // 20]:.6f}   "
              f"median {drift[n // 2]:.6f}   max {drift[-1]:.6f}")
        print("(values near 1 mean the two checkpoints share a basin and the segment "
              "between them is meaningful)")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {args.out} ...")
    dcp.save(A, storage_writer=dcp.FileSystemWriter(args.out), no_dist=True)
    cfg = args.a / "config.json"
    if cfg.is_file():
        shutil.copy2(cfg, args.out / "config.json")
    print("done")


if __name__ == "__main__":
    main()

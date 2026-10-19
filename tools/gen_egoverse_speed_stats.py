# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-embodiment window end-effector-speed percentiles for EgoVerse.

Each EgoVerse episode stores a per-frame ``meta/eef_speed`` array of shape
``(T, 2)`` = ``[left_arm_speed, right_arm_speed]``. The absolute speed scale
differs ~5× across embodiments (the ``eva`` robot is much slower than the human
rigs), so to use speed as a normalized 0–1 conditioning control it must be scaled
**per embodiment**.

This tool samples windows per embodiment, computes each window's overall speed
(``median_t( mean(eef_speed[w0:w0+H], arms) )`` over the training span), and emits
robust ``p01``/``p99`` bounds per embodiment::

    {"<emb>": {"lo": <p01>, "hi": <p99>}}

The reader maps a window's overall speed to ``clip((s - lo) / (hi - lo), 0, 1)``
for the text/control conditioning. Same span ``H`` must be used at train and eval.

Usage::

    python tools/gen_egoverse_speed_stats.py \\
        --data-root /mnt/ddn/dataset --pairs-dir /mnt/ddn/dataset/egoverse_pairs_v5 \\
        --out cosmos_framework/data/vfm/action/datasets/egoverse_speed_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

EMB_ROOT = {"MECKA": "egoverse", "scale": "egoverse", "aria": "egoverse", "mimicdroid": "egoverse", "eva": "egoverse_eva"}


def window_speed(data_root: Path, root: str, ep: str, w0: int, span: int) -> float | None:
    """Overall window speed = median over frames of the per-arm-mean eef speed."""
    try:
        arr = np.asarray(zarr.open(f"{data_root/root/ep}/meta/eef_speed", mode="r")[w0 : w0 + span])
    except Exception:
        return None
    if arr.size == 0:
        return None
    return float(np.median(arr.mean(axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--pairs-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--span", type=int, default=29, help="Training frame span (chunk_length+1).")
    ap.add_argument("--sample-per-emb", type=int, default=8000, help="Windows sampled per embodiment.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w = np.load(args.pairs_dir / "windows.npz", allow_pickle=True)
    ep, emb, w0, keep = w["ep"], w["emb"], w["w0"].astype(np.int64), w["keep"]
    rng = np.random.default_rng(args.seed)

    payload: dict[str, dict] = {}
    for e, root in EMB_ROOT.items():
        idx = np.where((emb == e) & keep)[0]
        if idx.size > args.sample_per_emb:
            idx = rng.choice(idx, size=args.sample_per_emb, replace=False)
        speeds = [s for i in idx if (s := window_speed(args.data_root, root, str(ep[i]), int(w0[i]), args.span)) is not None]
        sp = np.asarray(speeds, dtype=np.float64)
        lo, hi = float(np.percentile(sp, 1)), float(np.percentile(sp, 99))
        payload[e] = {"lo": round(lo, 6), "hi": round(hi, 6)}
        print(f"{e}: n={sp.size} lo(p01)={lo:.4f} hi(p99)={hi:.4f} median={np.median(sp):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate the shared LIBERO action-normalization JSON from the dataset stats.

The ``nvidia/LIBERO_LeRobot_v3`` release ships one ``meta/stats.json`` per task
suite. We train a single 7D action policy on the four eval suites combined
(``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``), so the
normalization must span their union:

  * ``min`` / ``max``  → exact union (min-of-mins, max-of-maxes). These are the
    only stats min-max normalization (the recipe default) actually uses, so they
    are exact.
  * ``mean`` / ``std`` / ``q01`` / ``q99`` → frame-count-weighted aggregate.
    ``mean`` and ``q01``/``q99`` are exact / near-exact under weighting; ``std``
    is a slight approximation (ignores between-suite mean shift) and is only used
    by the optional ``meanstd`` method. Falls back to ``min``/``max`` when a
    suite omits quantiles.

The result is written as ``{"global": {"min": [...], "max": [...], ...}}`` —
the format consumed by both
:func:`cosmos_framework.data.vfm.action.action_normalization.load_action_stats`
(training) and ``action_policy_server_libero.py --action-stats-path`` (eval), so
train-time normalization and eval-time denormalization stay identical.

Usage::

    python tools/gen_libero_normalization.py \\
        --dataset-root /path/to/LIBERO_LeRobot_v3 \\
        --out cosmos_framework/data/vfm/action/datasets/libero_lerobot_normalization.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "cosmos_framework/data/vfm/action/datasets/libero_lerobot_normalization.json"
)


def _load_suite_action_stats(root: Path, suite: str) -> tuple[dict[str, np.ndarray], int]:
    info = json.loads((root / suite / "meta" / "info.json").read_text())
    stats = json.loads((root / suite / "meta" / "stats.json").read_text())["action"]
    out = {k: np.asarray(v, dtype=np.float64) for k, v in stats.items() if k in {"min", "max", "mean", "std"}}
    # Quantiles may be absent in some suite stats; fall back to min/max.
    out["q01"] = np.asarray(stats.get("q01", stats["min"]), dtype=np.float64)
    out["q99"] = np.asarray(stats.get("q99", stats["max"]), dtype=np.float64)
    return out, int(info["total_frames"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, type=Path, help="LIBERO_LeRobot_v3 root (contains suite dirs).")
    ap.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES), help="Suites to aggregate.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path.")
    args = ap.parse_args()

    per_suite = []
    weights = []
    for suite in args.suites:
        s, n = _load_suite_action_stats(args.dataset_root, suite)
        per_suite.append(s)
        weights.append(n)
        print(f"  {suite}: {n} frames")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    combined = {
        "min": np.min([s["min"] for s in per_suite], axis=0),
        "max": np.max([s["max"] for s in per_suite], axis=0),
        "mean": np.sum([wi * s["mean"] for wi, s in zip(w, per_suite)], axis=0),
        "std": np.sqrt(np.sum([wi * (s["std"] ** 2) for wi, s in zip(w, per_suite)], axis=0)),
        "q01": np.sum([wi * s["q01"] for wi, s in zip(w, per_suite)], axis=0),
        "q99": np.sum([wi * s["q99"] for wi, s in zip(w, per_suite)], axis=0),
    }
    payload = {"global": {k: [round(float(x), 6) for x in v] for k, v in combined.items()}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

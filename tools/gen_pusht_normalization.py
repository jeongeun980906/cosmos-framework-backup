# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export PushT action-normalization stats for the eval policy server.

The PushT training dataset (:class:`PushTLeRobotDataset`) computes its min/max
normalization **in-memory** from the action column. For closed-loop eval, the
policy server (``action_policy_server_libero.py``, which is domain-generic) must
denormalize predicted actions with the EXACT same stats. This script
instantiates the dataset, reads back its computed stats, and writes them in the
``{"global": {"min": [...], "max": [...], ...}}`` format the server consumes via
``--action-stats-path``.

Usage::

    python tools/gen_pusht_normalization.py \\
        --root /path/to/PushT-Cosmos-Policy-lerobot/pusht-baseline-top100 \\
        --out cosmos_framework/data/vfm/action/datasets/pusht_lerobot_normalization.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "cosmos_framework/data/vfm/action/datasets/pusht_lerobot_normalization.json"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="PushT LeRobot v3.0 dataset root (the trained split).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path.")
    args = ap.parse_args()

    from cosmos_framework.data.vfm.action.datasets.pusht_lerobot_dataset import PushTLeRobotDataset

    ds = PushTLeRobotDataset(root=str(args.root), action_normalization="minmax")
    stats = ds._load_norm_stats()  # {mean, std, min, max, q01, q99} tensors, shape [2]
    payload = {"global": {k: [round(float(x), 6) for x in v.tolist()] for k, v in stats.items()}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

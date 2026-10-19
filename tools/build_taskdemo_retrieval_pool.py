# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Precompute the TaskDemos ABS-TRAJ retrieval pool for GR1 online eval (t=0 conditioning).

The t=0 contract (see ``docs/gr1_inference_retrieval_spec.md`` §2 / ``77_gr1_t0_taskdemo.py``):
the first query conditions on a SAME-task TaskDemo opening. Pool = the 12 canonical taskdemos
``taskdemo_<task>_<task>_D_demo_{1,2,3}`` (3 per task × 4 GR1 tasks), read straight from the
egoverse ``state57``. Descriptor = the SAME abs both-hand 6D-pose trajectory as the play pool
/ ``78_future_pairs_abstraj`` (no z-score). Gated by **task** (no subgoal gate).

Build once (1x and 2x)::

    python tools/build_taskdemo_retrieval_pool.py \\
        --out /mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool.npz            # 1x
    python tools/build_taskdemo_retrieval_pool.py --speed 2x \\
        --out /mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool_2x.npz

Output keys: T (N,SEQ,18), Mpose (N,18), ep, w0(=boundary), task.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import zarr

from cosmos_framework.scripts.gr1_online_retrieval import FUT, SEQ, WIN, traj_from_state57

HUM = "/mnt/ddn/dataset/egoverse"
TASKS = ["TurnOnFaucet", "CloseLeftCabinetDoor", "PnPSinkToCabinet", "PnPSinkToRightCounterPlate"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", choices=["1x", "2x"], default="1x")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--tasks", nargs="+", default=TASKS,
                    help="task names to include; each is gated by its OWN name (append L3 names to add "
                         "unseen-layout TaskDemos, e.g. TurnOnFaucetL3, for L3 conditioning).")
    ap.add_argument("--out", type=Path, default=Path("/mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool.npz"))
    args = ap.parse_args()
    suf = "__2x" if args.speed == "2x" else ""
    t0 = time.time()

    T, MP, EP, W0, TASK = ([] for _ in range(5))
    for tk in args.tasks:
        for i in (1, 2, 3):
            ep = f"taskdemo_{tk}_{tk}_D_demo_{i}{suf}"
            zpath = f"{HUM}/{ep}/state57/state"
            try:
                s = np.asarray(zarr.open(zpath, mode="r")[:], np.float64)
            except Exception as ex:  # noqa: BLE001
                print(f"  SKIP {ep}: {ex}")
                continue
            n = len(s)
            for e in range(args.stride, n, args.stride):
                if e + FUT > n:
                    break
                tr = traj_from_state57(s, e)
                T.append(tr)
                MP.append(tr.mean(0))
                EP.append(ep)
                W0.append(e)
                TASK.append(tk)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        T=np.asarray(T, np.float32),
        Mpose=np.asarray(MP, np.float32),
        ep=np.array(EP),
        w0=np.asarray(W0, np.int64),
        task=np.array(TASK),
        WIN=WIN, SEQ=SEQ, stride=args.stride, speed=args.speed,
    )
    print(f"Wrote {args.out}: {len(EP)} windows ({time.time() - t0:.0f}s)")
    print("  per-task windows:", dict(collections.Counter(TASK)))


if __name__ == "__main__":
    main()

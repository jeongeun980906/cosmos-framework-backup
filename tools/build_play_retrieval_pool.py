# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Precompute the play-demo ABS-TRAJ descriptor pool for GR1 online retrieval.

Mirrors ``78_future_pairs_abstraj.py`` build_pool EXACTLY (per-subgoal PAST windows over
the egoverse ``state57``, abs both-hand 6D-pose trajectory, no z-score) and SAVES it so
the eval server's ``gr1_online_retrieval.OnlineRetriever`` loads it instead of rebuilding.
Unlike 78 (one gated pool per task), this stores EVERY subgoal's windows tagged with their
own (type, src, dst) so the online matcher reproduces 78's per-task pool via subgoal gating.
Build once per play set::

    python tools/build_play_retrieval_pool.py --play 1x \\
        --out /mnt/ddn/dataset/egoverse_pairs_v2/play_pool_1x.npz
    python tools/build_play_retrieval_pool.py --play 2x \\
        --out /mnt/ddn/dataset/egoverse_pairs_v2/play_pool_2x.npz

Output keys (row-aligned per window): T (N,SEQ,18) abs-traj, Mpose (N,18) mean pose,
ep, w0(=future-start boundary), sgtype, src, dst, prel (= boundary - subgoal start).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import zarr

from cosmos_framework.scripts.gr1_online_retrieval import SEQ, WIN, traj_from_state57

HUM = "/mnt/ddn/dataset/egoverse"


def _subgoals(g) -> list:
    sg = g.attrs.get("subgoals")
    if not sg:
        return []
    return json.loads(sg) if isinstance(sg, str) else sg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play", choices=["1x", "2x"], required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    two = args.play == "2x"
    t0 = time.time()

    T, MP, EP, W0, SGT, SRC, DST, PREL = ([] for _ in range(8))
    for d in sorted(glob.glob(f"{HUM}/*SinkEnvForPlay*demo*")):
        if two != d.endswith("__2x"):
            continue
        g = zarr.open_group(d, mode="r")
        sgl = _subgoals(g)
        if not sgl:
            continue
        s = np.asarray(g["state57/state"][:], np.float64)
        ep = os.path.basename(d)
        for sg in sgl:
            start = int(sg["start"])
            for e in range(max(start, WIN), min(int(sg["end"]), len(s)), args.stride):
                tr = traj_from_state57(s, e)
                T.append(tr)
                MP.append(tr.mean(0))
                EP.append(ep)
                W0.append(e)
                SGT.append(str(sg.get("type")))
                SRC.append(str(sg.get("src")))
                DST.append(str(sg.get("dst")))
                PREL.append(e - start)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        T=np.asarray(T, np.float32),
        Mpose=np.asarray(MP, np.float32),
        ep=np.array(EP),
        w0=np.asarray(W0, np.int64),
        sgtype=np.array(SGT),
        src=np.array(SRC),
        dst=np.array(DST),
        prel=np.asarray(PREL, np.int64),
        WIN=WIN, SEQ=SEQ, stride=args.stride, play=args.play,
    )
    print(f"Wrote {args.out}: {len(EP)} play({args.play}) windows ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

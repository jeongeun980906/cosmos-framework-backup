# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a windows.npz + pairs.npz pair set for GR1 post-training.

The GR1 retrieval is the FUTURE-PAIR set produced by ``75_future_pairs.py`` as two
match files (``gr1_future_pairs_1x.npz`` / ``gr1_future_pairs_2x.npz``) in the
q_ep/m_ep layout — GR1 (humanoid robot, query / current embodiment) conditioned on
SinkEnvForPlay ``play`` demos at 1x and 2x speed. CRITICAL: q_w0/m_w0 are the
future-start BOUNDARY (the match was done on up to WIN PAST frames BEFORE w0, so it
is causal and matches the online retrieval; early/first-frame GR1 windows are forced
to early-play windows so frame-0 maps to the demo's START — covers from frame 0).
The conditioning/target clip is the FUTURE ``action57[w0 : w0+FUT]`` read by the
reader as ``act[w0:w0+chunk]``. Both GR1 and the play demos are egoverse ``action57``
zarr (20 Hz); this tool converts the q_ep/m_ep matches into the window-index
``windows.npz`` (ep, w0, w1, isev, emb, cls, keep) + ``pairs.npz``
(query_idx, match_idx, match_dist) the dataloader consumes, MERGING the 1x and 2x
demo pools per GR1 query.

DO NOT use the OLD forward-window match files ``gr1_play{1x,2x}_match.npz`` (from
``67_gr1_desc``/``68_gr1_match``, WIN=100, descriptor over [w,w+WIN] = FUTURE): those
are NON-CAUSAL (matched on the robot's unseen future) and inconsistent with the
past-window online retrieval — first-frame queries map to arbitrary deep play
positions instead of the demo start. They are deprecated; 75 (past) is correct.

Embodiment tags: GR1 query → ``gr1`` (register in EMB_REGISTRY: egoverse root, dim
57, hand_pose domain, 20 fps); play demos → ``mimicdroid`` (already registered).

Usage::

    python tools/build_gr1_posttrain_pairs.py \\
        --src /mnt/ddn/dataset/egoverse_pairs_v2 \\
        --data-root /mnt/ddn/dataset \\
        --out /mnt/ddn/dataset/egoverse_pairs_gr1
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import zarr

WIN = 36  # window SPAN written to windows.npz (w1 = w0 + WIN) and the full-window validity
          # requirement (w0 + WIN <= action rows). Widened 32 -> 36 (gr1_data_eval_redesign §5):
          # the action chunk is still 32, but a 32-frame span floors the 4n+1 video read to 29
          # frames / 28 deltas AND leaves no room for the per-frame δ∈{0..3} lockstep offset.
          # Span 36 = chunk 32 + (STRIDE-1)=3 offset room + 1 → full 33-frame clip / 32 deltas
          # at every δ. Keep in sync with 78_future_pairs_abstraj --fut.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/mnt/ddn/dataset/egoverse_pairs_v2"))
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/ddn/dataset"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/ddn/dataset/egoverse_pairs_gr1"))
    # Which play-speed match pools to merge into the demo pool. Default = both (the
    # original merged set); pass --speeds 1x (or 2x) to train on a single speed only.
    ap.add_argument("--speeds", nargs="+", choices=["1x", "2x"], default=["1x", "2x"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _len: dict[str, int] = {}

    def eplen(ep: str) -> int:
        if ep not in _len:
            try:
                _len[ep] = int(zarr.open(f"{args.data_root}/egoverse/{ep}/action57/action", mode="r").shape[0])
            except Exception:
                _len[ep] = 0
        return _len[ep]

    win: dict[tuple, int] = {}
    W_ep: list[str] = []
    W_w0: list[int] = []
    W_emb: list[str] = []
    W_cls: list[str] = []

    def add_win(emb: str, ep: str, w0: int, cls: str) -> int | None:
        if ep == "" or w0 < 0:
            return None
        L = eplen(ep)
        if L == 0 or w0 + WIN > L:  # require a full WIN-frame window (span == WIN)
            return None
        key = (emb, ep, int(w0))
        if key not in win:
            win[key] = len(W_ep)
            W_ep.append(ep)
            W_w0.append(int(w0))
            W_emb.append(emb)
            W_cls.append(cls)
        return win[key]

    queries: "OrderedDict[tuple, dict]" = OrderedDict()  # (q_ep, q_w0) -> {idx, matches:[(midx,dist)]}

    def ingest(path: Path) -> None:
        d = np.load(path, allow_pickle=True)
        qep = d["q_ep"].astype(str)
        qw0 = d["q_w0"].astype(np.int64)
        qtask = d["q_task"].astype(str)
        mep = d["m_ep"].astype(str)
        mw0 = d["m_w0"].astype(np.int64)
        md = d["m_dist"].astype(np.float32)
        kept = 0
        for i in range(len(qep)):
            qi = add_win("gr1", qep[i], int(qw0[i]), qtask[i])
            if qi is None:
                continue
            kept += 1
            q = queries.setdefault((qep[i], int(qw0[i])), dict(idx=qi, matches=[]))
            for j in range(mep.shape[1]):
                mi = add_win("mimicdroid", mep[i, j], int(mw0[i, j]), "play")
                if mi is not None:
                    q["matches"].append((mi, float(md[i, j])))
        print(f"  {path.name}: {len(qep)} queries, {kept} with a valid GR1 window")

    for sp in args.speeds:
        ingest(args.src / f"gr1_future_pairs_{sp}.npz")  # t>0: 75_future_pairs (PAST-matched, causal)
        # t=0: FORCED map to same-task TaskDemo opening [0:32], SPEED-MATCHED (1x->taskdemo, 2x->taskdemo__2x).
        # 77_gr1_t0_taskdemo.py --speed {sp}. Gives every GR1 episode a w0=0 query that conditions on a
        # same-task demo's start at the matching play speed — what online retrieval needs at the first frame.
        _t0 = args.src / ("gr1_t0_taskdemo.npz" if sp == "1x" else f"gr1_t0_taskdemo_{sp}.npz")
        if _t0.exists():
            ingest(_t0)
        else:
            print(f"  WARNING: {_t0.name} missing — no t=0 taskdemo pairs for {sp} (run 77_gr1_t0_taskdemo.py --speed {sp})")

    # ---- windows.npz -----------------------------------------------------
    N = len(W_ep)
    windows = dict(
        ep=np.array(W_ep),
        w0=np.array(W_w0, np.int64),
        w1=np.array([w + WIN for w in W_w0], np.int64),
        isev=np.zeros(N, np.int64),
        emb=np.array(W_emb),
        cls=np.array(W_cls),
        keep=np.ones(N, bool),
    )

    # ---- pairs.npz (merged 1x + 2x demo pool per query, dedup matches) ----
    maxk = max((len(q["matches"]) for q in queries.values() if q["matches"]), default=1)
    qidx: list[int] = []
    midx: list[list[int]] = []
    mdist: list[list[float]] = []
    for q in queries.values():
        if not q["matches"]:
            continue
        seen: set[int] = set()
        row: list[int] = []
        dr: list[float] = []
        for m, dist in sorted(q["matches"], key=lambda x: x[1]):  # nearest first
            if m in seen:
                continue
            seen.add(m)
            row.append(m)
            dr.append(dist)
        midx.append(row + [-1] * (maxk - len(row)))
        mdist.append(dr + [0.0] * (maxk - len(dr)))
        qidx.append(q["idx"])

    pairs = dict(
        query_idx=np.array(qidx, np.int64),
        match_idx=np.array(midx, np.int64),
        match_dist=np.array(mdist, np.float32),
    )

    np.savez(args.out / "windows.npz", **windows)
    np.savez(args.out / "pairs.npz", **pairs)
    n_gr1 = int(sum(e == "gr1" for e in W_emb))
    n_md = int(sum(e == "mimicdroid" for e in W_emb))
    vcount = (pairs["match_idx"] >= 0).sum(1)
    print(
        f"Wrote {args.out}: windows={N} (gr1={n_gr1}, mimicdroid={n_md}), "
        f"queries={len(qidx)}, match width={maxk}, "
        f"matches/query min={int(vcount.min())} median={int(np.median(vcount))} max={int(vcount.max())}"
    )


if __name__ == "__main__":
    main()

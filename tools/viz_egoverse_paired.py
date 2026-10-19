# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Visualize EgoVerse conditioning-policy paired samples to sanity-check the
input/output objective (video-focused).

For a query, renders the demo (condition) clip and the query clip as a labeled
montage so the design is verifiable at a glance:
  * DEMO row  — all frames are CLEAN conditioning (the demonstration).
  * QUERY row — frame 0 is the CLEAN observation; frames 1..N are the GENERATION
    target (what the policy must produce), alongside the target action.

Saves PNGs to --out-dir. Picks a same-rig pair and a cross-embodiment (eva↔human)
pair so both regimes are visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmos_framework.data.vfm.action.datasets.egoverse_zarr_dataset import (
    EMB_REGISTRY,
    EgoVerseWindowReader,
    _read_frames,
)


def _decode_window_rgb(reader: EgoVerseWindowReader, widx: int, frame_idxs: list[int]) -> list[np.ndarray]:
    """Decode specific frames of a window (front cam) as RGB uint8."""
    emb = reader.emb_of(widx)
    spec = EMB_REGISTRY[emb]
    ep = str(reader._ep[widx])
    w0 = int(reader._w0[widx])
    d = str(reader._data_root / spec.root / ep)
    frames = _read_frames(d, spec.cams[0], w0, w0 + max(frame_idxs) + 1)  # (T,H,W,3)
    return [frames[k] for k in frame_idxs]


def _render_pair(reader, q_widx: int, c_widx: int, span: int, out_path: Path) -> None:
    # 6 sampled frames across the 1s window.
    idxs = [int(round(x)) for x in np.linspace(0, span - 1, 6)]
    demo = _decode_window_rgb(reader, c_widx, idxs)
    query = _decode_window_rgb(reader, q_widx, idxs)

    q_emb, c_emb = reader.emb_of(q_widx), reader.emb_of(c_widx)
    q_meta = reader.read_window(q_widx)
    c_meta = reader.read_window(c_widx)

    n = len(idxs)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.6))
    for col, (fi, img) in enumerate(zip(idxs, demo)):
        ax = axes[0, col]
        ax.imshow(img)
        ax.set_title(f"t={fi}", fontsize=9)
        ax.axis("off")
        # all demo frames are clean conditioning
        for s in ax.spines.values():
            s.set_visible(False)
    for col, (fi, img) in enumerate(zip(idxs, query)):
        ax = axes[1, col]
        ax.imshow(img)
        role = "OBS (clean)" if fi == 0 else "GENERATE"
        color = "tab:green" if fi == 0 else "tab:red"
        ax.set_title(f"t={fi}\n{role}", fontsize=9, color=color)
        ax.axis("off")

    cross = "CROSS-EMB" if q_emb != c_emb else "same-emb"
    axes[0, 0].set_ylabel("DEMO\n(condition, all clean)", fontsize=11, color="tab:blue")
    axes[1, 0].set_ylabel("QUERY\n(predict action+video)", fontsize=11)
    # matplotlib hides ylabel with axis off; re-add as text
    axes[0, 0].axis("on"); axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    axes[1, 0].axis("on"); axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    fig.suptitle(
        f"[{cross}]  demo={c_emb}(dom {c_meta['domain_id'].item()}, {c_meta['raw_action_dim']}D, "
        f"spd {c_meta['speed']:.2f})   →   query={q_emb}(dom {q_meta['domain_id'].item()}, "
        f"{q_meta['raw_action_dim']}D, spd {q_meta['speed']:.2f})\n"
        f"caption: {q_meta['ai_caption']}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    print(f"  wrote {out_path}  ({cross}: {c_emb}→{q_emb})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/ddn/dataset")
    ap.add_argument("--pairs-dir", default="/mnt/ddn/dataset/egoverse_pairs_v5")
    ap.add_argument("--norm", default="cosmos_framework/data/vfm/action/datasets/egoverse_normalization.json")
    ap.add_argument("--speed", default="cosmos_framework/data/vfm/action/datasets/egoverse_speed_stats.json")
    ap.add_argument("--out-dir", default="/tmp/egoverse_viz")
    ap.add_argument("--span", type=int, default=29)
    args = ap.parse_args()

    reader = EgoVerseWindowReader(
        data_root=args.data_root, pairs_dir=args.pairs_dir,
        normalization_path=args.norm, speed_stats_path=args.speed,
    )
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    p = np.load(Path(args.pairs_dir) / "pairs.npz", allow_pickle=True)
    qi, mi = p["query_idx"], p["match_idx"]
    emb = reader._emb
    rng = np.random.default_rng(0)

    # Find one same-emb and one cross-emb (prefer eva involved) example.
    same_done = cross_done = False
    order = rng.permutation(len(qi))
    for i in order:
        q = int(qi[i])
        cands = mi[i][mi[i] >= 0]
        if cands.size == 0:
            continue
        c = int(cands[0])
        is_cross = emb[q] != emb[c]
        eva_involved = "eva" in (emb[q], emb[c])
        if is_cross and eva_involved and not cross_done:
            _render_pair(reader, q, c, args.span, out / "pair_cross_emb_eva.png")
            cross_done = True
        elif (not is_cross) and not same_done:
            _render_pair(reader, q, c, args.span, out / "pair_same_emb.png")
            same_done = True
        if same_done and cross_done:
            break
    print("done.")


if __name__ == "__main__":
    main()

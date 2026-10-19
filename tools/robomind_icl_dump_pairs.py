# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dump RoboMIND ICL ``[demo | query]`` pairs as side-by-side mp4s, to eyeball the match.

Writes exactly what the model is fed -- the transformed, resized, normalized clips taken
straight out of ``RoboMINDPairedDataset`` -- so what you watch is what trains, not a
prettier re-render of the source video. Demo on the left (clean context), query on the
right (the generation target), with the task and episode indices burned into a header.

Usage::

    LEROBOT_VIDEO_BACKEND=pyav python tools/robomind_icl_dump_pairs.py \\
        --root /mnt/ddn/dataset/robomind_franka_3rgb \\
        --tasks 241021_insert_marker_1 241022_lamp_on_1 \\
        --num 4 --out /tmp/icl_pairs

``--tasks`` is worth passing: without it every task repo under ``root`` is opened and its
row index materialized, which is minutes of work for a look at four clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_robomind_icl_sft_dataset,
)

_HEADER_H = 28
_GAP = 8  # gutter between the two clips, so the seam is never mistaken for content


def _to_uint8(video: torch.Tensor) -> np.ndarray:
    """``[C,T,H,W]`` in [-1, 1] -> ``[T,H,W,C]`` uint8."""
    x = video.permute(1, 2, 3, 0).float()
    return x.add(1.0).mul(127.5).clamp(0, 255).to(torch.uint8).numpy()


def _label(width: int, text: str, *, align_left: bool = True) -> np.ndarray:
    strip = np.zeros((_HEADER_H, width, 3), dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = 8 if align_left else max(8, width - tw - 8)
    cv2.putText(strip, text, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def _side_by_side(demo: np.ndarray, query: np.ndarray, left: str, right: str) -> np.ndarray:
    t = min(demo.shape[0], query.shape[0])
    demo, query = demo[:t], query[:t]
    h = max(demo.shape[1], query.shape[1])
    gap = np.zeros((h, _GAP, 3), dtype=np.uint8)
    header = np.concatenate(
        [_label(demo.shape[2], left), np.zeros((_HEADER_H, _GAP, 3), np.uint8),
         _label(query.shape[2], right, align_left=False)],
        axis=1,
    )
    frames = [
        np.concatenate([header, np.concatenate([demo[i], gap, query[i]], axis=1)], axis=0)
        for i in range(t)
    ]
    return np.stack(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Parent dir holding the per-task LeRobot repos")
    ap.add_argument("--tasks", nargs="*", default=None, help="Task dirs to open (default: all under --root)")
    ap.add_argument("--embodiment-type", default="robomind-franka")
    ap.add_argument("--chunk-length", type=int, default=16)
    ap.add_argument("--resolution", default="256")
    ap.add_argument("--num", type=int, default=4, help="Pairs to dump")
    ap.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit sample indices")
    ap.add_argument("--seed", type=int, default=0, help="Pairing seed (matches the training config)")
    ap.add_argument("--demo-match", default="state", choices=("state", "random"),
                    help="state: demo picked at the query's phase; random: uniform in the episode")
    ap.add_argument("--fps", type=int, default=8, help="Playback fps (the clip is ~0.5 s of real time)")
    ap.add_argument("--repeat", type=int, default=3, help="Loop the clip this many times, to make it watchable")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    ds = get_action_robomind_icl_sft_dataset(
        root=args.root,
        tasks=args.tasks,
        embodiment_type=args.embodiment_type,
        chunk_length=args.chunk_length,
        resolution=args.resolution,
        tokenizer_config=None,  # inspection only: no captions to tokenize
        demo_match=args.demo_match,
        seed=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.indices:
        indices = args.indices
    else:
        # Spread across the whole index space rather than taking the first N, which would
        # all land in the same episode of the same task.
        indices = [round(k * (len(ds) - 1) / max(1, args.num - 1)) for k in range(args.num)]

    for i in indices:
        prov = ds.pair_provenance(i)
        sample = ds[i]
        demo, query = (_to_uint8(v) for v in sample["video"])
        frames = _side_by_side(
            demo,
            query,
            f"DEMO  ep{prov['demo_episode']} w{prov['demo_window']}  [{args.demo_match}]",
            f"QUERY ep{prov['query_episode']} w{prov['query_window']}",
        )
        frames = np.concatenate([frames] * max(1, args.repeat))
        name = f"{prov['task']}_i{i}_q-ep{prov['query_episode']}_demo-ep{prov['demo_episode']}.mp4"
        imageio.mimwrite(out / name, frames, fps=args.fps, macro_block_size=1, quality=8)
        print(
            f"[{i}] {prov['task']} ({prov['embodiment']}) "
            f"demo ep{prov['demo_episode']} w{prov['demo_window']} -> "
            f"query ep{prov['query_episode']} w{prov['query_window']}  "
            f"same_episode={prov['demo_episode'] == prov['query_episode']}  "
            f"match_dist={prov['match_distance'] if prov['match_distance'] is None else round(prov['match_distance'], 2)}\n"
            f"     {sample['ai_caption'][:110]}...\n"
            f"     -> {out / name}"
        )


if __name__ == "__main__":
    main()

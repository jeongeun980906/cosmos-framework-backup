# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Splice one checkpoint's action heads onto another checkpoint's backbone.

The checkpointer loads from a single source, so an arm that wants "backbone from A,
action heads from B" cannot be expressed with ``keys_to_skip_loading`` alone —
skipping a key leaves the module at its constructor init (Xavier/zeros), not at B's
value. This writes a merged DCP that can then be loaded normally.

Motivating case (RoboCasa c3edge ``midbb``): take the EgoVerse-trained Edge backbone
but keep the public Cosmos3-Edge action heads. Note ``action2llm``/``llm2action`` are
``DomainAwareLinear`` — an ``nn.Embedding`` of per-domain weight matrices — so a run
only trains the domain rows it actually indexes. EgoVerse (domain 3) leaves RoboCasa's
domain 5 row untouched apart from weight decay, whereas ``action_modality_embed`` is a
single tensor shared across domains and IS rewritten. Splicing all of them restores
the public head exactly.

Usage::

    python tools/merge_action_head_dcp.py \\
        --backbone <EgoVerse DCP model dir> \\
        --heads    <Cosmos3-Edge DCP model dir> \\
        --out      <new DCP model dir>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

# Every tensor that belongs to the action head, for both the live net and its EMA copy.
HEAD_SUBSTRINGS = ("action2llm", "llm2action", "action_modality_embed", "action_pos_embed")


def _metadata(path: Path):
    return dcp.FileSystemReader(path).read_metadata().state_dict_metadata


def _empty_like_meta(md, keys):
    return {
        k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype)
        for k in keys
        if hasattr(md[k], "size")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, type=Path, help="DCP model dir supplying everything else.")
    ap.add_argument("--heads", required=True, type=Path, help="DCP model dir supplying the action heads.")
    ap.add_argument("--out", required=True, type=Path, help="Output DCP model dir (created).")
    args = ap.parse_args()

    bb_md, hd_md = _metadata(args.backbone), _metadata(args.heads)
    head_keys = sorted(k for k in bb_md if any(t in k for t in HEAD_SUBSTRINGS))

    # A released base ships only ``net.*``; a training checkpoint also carries
    # ``net_ema.*``. Source each ema tensor from its own ``net.`` counterpart so the
    # EMA copy starts consistent with the net it tracks (EMA converges to net anyway,
    # but leaving the pair mismatched makes the first steps hard to reason about).
    src_of: dict[str, str] = {}
    for k in head_keys:
        if k in hd_md:
            src_of[k] = k
        elif k.startswith("net_ema.") and (alt := "net." + k[len("net_ema.") :]) in hd_md:
            src_of[k] = alt
        else:
            raise SystemExit(f"heads checkpoint has no source for {k}")
        if tuple(bb_md[k].size) != tuple(hd_md[src_of[k]].size):
            raise SystemExit(f"shape mismatch on {k}: {tuple(bb_md[k].size)} vs {tuple(hd_md[src_of[k]].size)}")
    print(f"action-head tensors to splice: {len(head_keys)}")
    for k in head_keys:
        via = "" if src_of[k] == k else f"   <- {src_of[k]}"
        print(f"  {k:<44} {tuple(bb_md[k].size)} {bb_md[k].properties.dtype}{via}")

    print(f"\nloading backbone ({len(bb_md)} keys) from {args.backbone} ...")
    sd = _empty_like_meta(bb_md, bb_md.keys())
    dcp.load(sd, storage_reader=dcp.FileSystemReader(args.backbone), no_dist=True)

    print(f"loading heads from {args.heads} ...")
    hd = _empty_like_meta(hd_md, sorted(set(src_of.values())))
    dcp.load(hd, storage_reader=dcp.FileSystemReader(args.heads), no_dist=True)

    for k in head_keys:
        before = sd[k].float()
        sd[k] = hd[src_of[k]].to(sd[k].dtype)
        d = (sd[k].float() - before).norm().item() / (before.norm().item() or 1.0)
        print(f"  spliced {k:<44} relative change {d:.2%}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting merged checkpoint to {args.out} ...")
    dcp.save(sd, storage_writer=dcp.FileSystemWriter(args.out), no_dist=True)

    # The loader reads config.json next to the shards; carry the backbone's over.
    src_cfg = args.backbone / "config.json"
    if src_cfg.is_file():
        shutil.copy2(src_cfg, args.out / "config.json")
        print(f"copied {src_cfg.name}")
    print("done")


if __name__ == "__main__":
    main()

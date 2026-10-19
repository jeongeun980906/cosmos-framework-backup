# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Present a single-clip post-train sample in the paired [demo, query] ICL layout.

Why this exists. Every EgoVerse warm start measured so far is WORSE on RoboCasa than the
public Edge base -- at backbone lr matched to base, at every band, routed through domain
5 or domain 8, with or without the RoboMIND BC replay -- and the deficit grows with
mid-train length (bcmix 2000 -> 4000 costs +3.4%, t=+4.33). The one thing every one of
those mid checkpoints shares is that ICL dominates its training: the model always saw a
retrieved demo in context, and the query-only pathway was never asked to carry a
prediction alone.

RoboCasa post-train is single-clip. So the warm start is evaluated in a layout its
mid-training never presented, and the penalty that costs would be exactly what we see:
per-step, not shrinking with post-train steps, and larger the longer mid-training ran.

This wrapper tests that directly and cheaply -- no retraining. It re-packs the RoboCasa
sample into the two-item layout the mid checkpoint was trained on, with the demo slot
present but EMPTY. If the deficit is a layout mismatch, restoring the layout recovers it;
if it is representational drift, this changes nothing. Either answer is decisive, and it
costs one 400-step probe instead of a 12-hour mid-train.

The demo slot is zeros in the already-normalized [-1, 1] space (mid-gray), one frame by
default: an empty demo carries the same nothing at any length, and the VAE encodes T=1 at
latent_T=1 without padding, so it costs 1 latent frame instead of the query's 9.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class DummyRetPairedWrap(Dataset):
    """Wrap a single-clip action dataset so each sample becomes ``[dummy_demo, query]``."""

    def __init__(self, dataset: Dataset, dummy_frames: int = 1) -> None:
        super().__init__()
        self._dataset = dataset
        self._dummy_frames = int(dummy_frames)
        # RankPartitionedDataLoader sets these on whatever object it is handed, so they
        # have to live here and be forwarded, or the wrapper silently un-shards the run
        # and every rank iterates the same samples.
        self.shard_world_size: int = 1
        self.shard_rank: int = 0

    def __len__(self) -> int:
        return len(self._dataset)

    def _sync_shard(self) -> None:
        inner = self._dataset
        if getattr(inner, "shard_world_size", None) != self.shard_world_size:
            inner.shard_world_size = self.shard_world_size
            inner.shard_rank = self.shard_rank

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._sync_shard()
        s = dict(self._dataset[idx])

        q_v = s["video"]
        if q_v.dtype == torch.uint8:
            # Both items of a multi-vision list must share a dtype: the model treats an
            # all-float list as already preprocessed and an all-uint8 list as raw, and a
            # mixed list takes the raw path and then asserts on the float item.
            q_v = q_v.float().div_(127.5).sub_(1.0)
        n = self._dummy_frames
        c_v = torch.zeros_like(q_v) if n <= 0 else torch.zeros_like(q_v[:, :1]).repeat(1, n, 1, 1)

        q_a = s["action"]
        c_a = torch.zeros_like(q_a)

        sp = s["sequence_plan"]
        # A shortened dummy has a different latent length than the query, so demo and
        # query cannot share one temporal grid.
        share = bool(getattr(sp, "share_vision_temporal_positions", False)) and n <= 0
        sp = type(sp)(
            has_text=sp.has_text,
            has_vision=sp.has_vision,
            condition_frame_indexes_vision=sp.condition_frame_indexes_vision,
            share_vision_temporal_positions=share,
            has_action=sp.has_action,
            condition_frame_indexes_action=sp.condition_frame_indexes_action,
            action_start_frame_offset=sp.action_start_frame_offset,
            has_sound=sp.has_sound,
            condition_frame_indexes_sound=sp.condition_frame_indexes_sound,
        )

        fps = s["conditioning_fps"]
        s.update(
            video=[c_v, q_v],
            image_size=[s["image_size"], s["image_size"]],
            action=[c_a, q_a],
            domain_id=[s["domain_id"], s["domain_id"]],
            raw_action_dim=[s["raw_action_dim"], s["raw_action_dim"]],
            conditioning_fps=torch.stack([fps, fps]),
            num_frames=int(q_v.shape[1]),
            sequence_plan=sp,
        )
        return s

    def get_shuffle_blocks(self):
        return self._dataset.get_shuffle_blocks()

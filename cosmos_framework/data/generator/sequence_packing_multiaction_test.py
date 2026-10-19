# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Regression + feature tests for multi-action-segment packing (conditioning policy).

Verifies that the additive ``num_action_items_per_sample`` path:
  1. is a no-op when unset — the single-action policy sample packs identically
     whether ``num_action_items_per_sample`` is ``None`` or ``[1]`` (gating is
     bit-identical, so existing LIBERO/DROID/PushT training is unchanged);
  2. packs a 2-segment sample (demo + query) with the right clean/noisy split —
     demo action fully clean (no MSE-supervised tokens), query action fully noisy;
  3. shares temporal mRoPE positions across the demo/query vision items AND the
     demo/query action segments when ``share_vision_temporal_positions=True``
     (the video-transfer-style aligned grid).
"""

from __future__ import annotations

import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

_SPECIAL = {"eos_token_id": 1, "start_of_generation": 2, "end_of_generation": 3}
_LAT_C, _LAT_T, _LAT_HW, _ACT_T, _ACT_D = 16, 8, 8, 28, 64


def _vision(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, _LAT_C, _LAT_T, _LAT_HW, _LAT_HW, generator=g)


def _action(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(_ACT_T, _ACT_D, generator=g)


def _pack(gen_data_clean: GenerationDataClean, plan: SequencePlan):
    return pack_input_sequence(
        sequence_plans=[plan],
        input_text_indexes=[[30] * 6],
        gen_data_clean=gen_data_clean,
        input_timesteps=torch.tensor([0.5]),
        special_tokens=_SPECIAL,
        position_embedding_type="unified_3d_mrope",
        temporal_compression_factor=4,
    )


def _single(num_action_items):
    """One policy sample: 1 vision item + 1 action segment."""
    gdc = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[_vision(0)],
        x0_tokens_action=[_action(0)],
        action_domain_id=[torch.tensor([3], dtype=torch.long)],
        num_action_items_per_sample=num_action_items,
    )
    plan = SequencePlan(
        has_text=True, has_vision=True, has_action=True,
        condition_frame_indexes_vision=[0], condition_frame_indexes_action=[],
        action_start_frame_offset=1,
    )
    return _pack(gdc, plan)


def _paired(share: bool):
    """One conditioning-policy sample: [demo_v, query_v] + [demo_a, query_a]."""
    gdc = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[_vision(1), _vision(2)],
        x0_tokens_action=[_action(1), _action(2)],
        action_domain_id=[torch.tensor([3], dtype=torch.long), torch.tensor([16], dtype=torch.long)],
        num_vision_items_per_sample=[2],
        num_action_items_per_sample=[2],
    )
    plan = SequencePlan(
        has_text=True, has_vision=True, has_action=True,
        condition_frame_indexes_vision=[0], condition_frame_indexes_action=[],
        share_vision_temporal_positions=share, action_start_frame_offset=1,
    )
    return _pack(gdc, plan)


def test_single_action_gating_is_noop():
    """num_action_items_per_sample None vs [1] must pack identically."""
    a = _single(None)
    b = _single([1])
    assert torch.equal(a.action.sequence_indexes, b.action.sequence_indexes)
    assert torch.equal(a.action.mse_loss_indexes, b.action.mse_loss_indexes)
    assert torch.equal(a.position_ids, b.position_ids)
    assert a.split_lens == b.split_lens
    assert len(a.action.condition_mask) == len(b.action.condition_mask) == 1
    for ca, cb in zip(a.action.condition_mask, b.action.condition_mask):
        assert torch.equal(ca, cb)


def test_paired_clean_noisy_split():
    """demo action fully clean (no MSE), query action fully noisy (all MSE)."""
    p = _paired(share=True)
    assert len(p.action.condition_mask) == 2, "two action segments expected"
    demo_mask, query_mask = p.action.condition_mask
    assert demo_mask.sum().item() == _ACT_T, "demo action must be all-clean (condition_mask=1)"
    assert query_mask.sum().item() == 0, "query action must be all-noisy (condition_mask=0)"
    # MSE-supervised action tokens == only the query segment's steps.
    assert p.action.mse_loss_indexes.numel() == _ACT_T


def _temporal_row(packed) -> torch.Tensor:
    pos = packed.position_ids
    return pos[0] if pos.ndim == 2 else pos  # mRoPE: (3, N) → temporal axis row 0


def test_shared_grid_aligns_demo_query():
    """share_vision_temporal_positions=True → demo & query vision items and action
    segments occupy the SAME temporal mRoPE positions (transfer-style)."""
    p = _paired(share=True)
    temporal = _temporal_row(p)

    # Vision: two items, each latent_T*HW*HW tokens, in packed order [demo_v, query_v].
    vis_idx = p.vision.sequence_indexes
    n_vis_tokens = _LAT_T * _LAT_HW * _LAT_HW
    demo_v_t = temporal[vis_idx[:n_vis_tokens]]
    query_v_t = temporal[vis_idx[n_vis_tokens : 2 * n_vis_tokens]]
    assert torch.equal(demo_v_t, query_v_t), "demo_v and query_v must share temporal positions"

    # Action: two segments of _ACT_T tokens each, in packed order [demo_a, query_a].
    act_idx = p.action.sequence_indexes
    demo_a_t = temporal[act_idx[:_ACT_T]]
    query_a_t = temporal[act_idx[_ACT_T : 2 * _ACT_T]]
    assert torch.equal(demo_a_t, query_a_t), "demo_a and query_a must share temporal positions"


def test_nonshared_grid_separates():
    """Without sharing, demo & query vision items occupy DIFFERENT temporal regions."""
    p = _paired(share=False)
    temporal = _temporal_row(p)
    vis_idx = p.vision.sequence_indexes
    n = _LAT_T * _LAT_HW * _LAT_HW
    demo_v_t = temporal[vis_idx[:n]]
    query_v_t = temporal[vis_idx[n : 2 * n]]
    assert not torch.equal(demo_v_t, query_v_t), "non-shared items must NOT align temporally"


if __name__ == "__main__":
    for fn in [
        test_single_action_gating_is_noop,
        test_paired_clean_noisy_split,
        test_shared_grid_aligns_demo_query,
        test_nonshared_grid_separates,
    ]:
        fn()
        print(f"PASS: {fn.__name__}")
    print("All multi-action packing tests passed.")

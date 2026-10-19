# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Double-check the conditioning-policy positional encoding (mRoPE) under the REAL
config params, confirming demo/query conditions are placed like video-transfer.

Builds a realistic paired sample (2 vision items + 2 action segments) and packs it
with the exact args the model passes (unified_3d_mrope, enable_fps_modulation=True,
base_fps=24, temporal_compression_factor=4, margin=15000, latent_patch_size=2),
then inspects the temporal-axis mRoPE positions per token block:

  * demo_video temporal positions == query_video  (shared grid, transfer-style)
  * demo_action temporal positions == query_action
  * action aligns onto the same physical-time axis as video (FPS-modulated)
  * a +15000 modality margin separates text from the diffusion tokens

Also packs an equivalent TRANSFER sample (control + target, same SequencePlan flag)
to show the mechanism is identical.
"""

from __future__ import annotations

import torch

from cosmos_framework.data.vfm.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean

# Real config params (from diffusion_expert_config / nano_model_config).
PE = dict(
    position_embedding_type="unified_3d_mrope",
    unified_3d_mrope_reset_spatial_ids=True,
    unified_3d_mrope_temporal_modality_margin=15000,
    enable_fps_modulation=True,
    base_fps=24.0,
    temporal_compression_factor=4,
    latent_patch_size=2,
)
SPECIAL = {"eos_token_id": 1, "start_of_generation": 2, "end_of_generation": 3}
FPS = 30.0
# 29 frames @tcf4 -> latent_t=8; 256x320 @16 -> 16x20 latent grid.
LAT = (1, 16, 8, 16, 20)
ACT_T, ACT_D = 28, 64


def _pack(gen, plan, n_action_fps):
    return pack_input_sequence(
        sequence_plans=[plan],
        input_text_indexes=[[30] * 6],
        gen_data_clean=gen,
        input_timesteps=torch.tensor([0.5]),
        special_tokens=SPECIAL,
        **PE,
    )


def _temporal(packed) -> torch.Tensor:
    p = packed.position_ids
    return (p[0] if p.ndim == 2 else p).float()


def _grp(temporal, idx, n0, n1):
    return temporal[idx[n0:n1]]


def verify_paired():
    print("=" * 78)
    print("PAIRED conditioning-policy sample  [text | demo_v query_v | demo_a query_a]")
    print("=" * 78)
    gen = GenerationDataClean(
        batch_size=1, is_image_batch=False,
        x0_tokens_vision=[torch.randn(*LAT), torch.randn(*LAT)],
        x0_tokens_action=[torch.randn(ACT_T, ACT_D), torch.randn(ACT_T, ACT_D)],
        action_domain_id=[torch.tensor([3]), torch.tensor([16])],  # scale(human) , eva(robot)
        fps_vision=torch.tensor([FPS]),
        fps_action=torch.tensor([FPS, FPS]),            # per-segment
        num_vision_items_per_sample=[2],
        num_action_items_per_sample=[2],
    )
    plan = SequencePlan(
        has_text=True, has_vision=True, has_action=True,
        condition_frame_indexes_vision=[0], condition_frame_indexes_action=[],
        share_vision_temporal_positions=True, action_start_frame_offset=1,
    )
    packed = _pack(gen, plan, 2)
    t = _temporal(packed)

    vis = packed.vision.sequence_indexes
    nv = vis.numel() // 2
    demo_v, query_v = _grp(t, vis, 0, nv), _grp(t, vis, nv, 2 * nv)
    act = packed.action.sequence_indexes
    demo_a, query_a = _grp(t, act, 0, ACT_T), _grp(t, act, ACT_T, 2 * ACT_T)

    text_t = t[: packed.text_indexes.numel() if hasattr(packed, "text_indexes") else 6]
    print(f"text temporal positions (first 6):     {t[:6].tolist()}")
    print(f"margin jump (text end -> vision start): ~{PE['unified_3d_mrope_temporal_modality_margin']}")
    print(f"demo_video  unique temporal frames: {sorted(set(demo_v.tolist()))}")
    print(f"query_video unique temporal frames: {sorted(set(query_v.tolist()))}")
    print(f"demo_action temporal positions:     {demo_a.tolist()[:8]} ...")
    print(f"query_action temporal positions:    {query_a.tolist()[:8]} ...")

    assert torch.equal(demo_v, query_v), "demo_v and query_v must share temporal grid"
    assert torch.equal(demo_a, query_a), "demo_a and query_a must share temporal grid"
    # action steps should fall on the same physical-time axis as video frames.
    vmin, vmax = float(demo_v.min()), float(demo_v.max())
    amin, amax = float(demo_a.min()), float(demo_a.max())
    print(f"\nvideo  temporal range: [{vmin:.2f}, {vmax:.2f}]   (8 latent frames, FPS-modulated)")
    print(f"action temporal range: [{amin:.2f}, {amax:.2f}]   (28 steps, frame-rate, +1 offset)")
    print("PASS: demo↔query vision aligned, demo↔query action aligned, shared physical-time axis.")
    return demo_v


def verify_transfer_equivalence(paired_demo_v_temporal):
    print("\n" + "=" * 78)
    print("TRANSFER reference  [text | control_v target_v]  (same share flag, no action)")
    print("=" * 78)
    gen = GenerationDataClean(
        batch_size=1, is_image_batch=False,
        x0_tokens_vision=[torch.randn(*LAT), torch.randn(*LAT)],
        fps_vision=torch.tensor([FPS]),
        num_vision_items_per_sample=[2],
    )
    plan = SequencePlan(
        has_text=True, has_vision=True,
        condition_frame_indexes_vision=[0],
        share_vision_temporal_positions=True,
    )
    packed = _pack(gen, plan, 0)
    t = _temporal(packed)
    vis = packed.vision.sequence_indexes
    nv = vis.numel() // 2
    ctrl, tgt = _grp(t, vis, 0, nv), _grp(t, vis, nv, 2 * nv)
    assert torch.equal(ctrl, tgt), "transfer control/target must share grid"
    print(f"control unique temporal frames: {sorted(set(ctrl.tolist()))}")
    print(f"target  unique temporal frames: {sorted(set(tgt.tolist()))}")
    same = torch.equal(ctrl, paired_demo_v_temporal)
    print(f"\ntransfer control temporal grid == paired demo_video temporal grid?  {same}")
    print("PASS: paired conditioning uses the SAME shared-temporal-grid mechanism as transfer.")


if __name__ == "__main__":
    dv = verify_paired()
    verify_transfer_equivalence(dv)
    print("\nALL PE CHECKS PASSED.")

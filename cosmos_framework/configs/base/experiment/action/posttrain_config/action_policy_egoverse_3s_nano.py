# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_egoverse_3s_nano`` — 3-second (v10) variant of ``action_policy_egoverse_nano``.

Mirrors ``action_policy_egoverse_5s_nano`` but for the 3s pairing (``egoverse_pairs_v10``,
human WIN_eg=90 @30Hz, MD WIN_md=60 @20Hz). Each transition spans the full 3s retrieval
window. ``_frames_from_span`` reads each rig at its native fps for the reference duration
``(chunk_length+1)/30`` then floors to a VAE-legal 4n+1 length capped by the window span:
  * chunk_length=88 → reference duration 89/30 = 2.97s.
  * human (30fps, span 90) → 89 frames (= 4*22+1); all 90 native frames minus the single
    4n+1 boundary frame — no meaningful subsampling (same pattern as 5s: 149 of 150).
  * mimicdroid (20fps, span 60) → 57 frames (= 4*14+1), handled per-emb like 5s MD (97).
Set the frame params HERE (not fragile CLI overrides) so the 3s run can't fall back to 1s.
Data dir is env EGOVERSE_PAIRS_DIR (point at egoverse_pairs_v10); normalization + speed
stats are window-independent and reused from the base recipe.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_egoverse_nano import (
    action_policy_egoverse_nano,
)

cs = ConfigStore.instance()

action_policy_egoverse_3s_nano = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_nano)),
    flags={"allow_objects": True},
)

# 3s @30fps reference = 89 frames (89 = 1 + 4*22). chunk_length = 88 deltas.
action_policy_egoverse_3s_nano["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"][
    "chunk_length"
] = 88
# 3s latent_t = (89-1)/4+1 = 23 (vs 5s 38) → ~1.6x lighter than 5s; 5s ran batch 5.
# Start at 8 for 3s and adjust to peak_gpu_mem_reserved_gb (lower if it OOMs).
action_policy_egoverse_3s_nano["dataloader_train"]["max_samples_per_batch"] = 8
action_policy_egoverse_3s_nano["job"]["name"] = "action_policy_egoverse_3s_nano"

# VAE must encode exactly 89 frames for the human reference path (like 5s pins [149];
# per-emb MD counts fall out of _frames_from_span the same way).
action_policy_egoverse_3s_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [89]


for _item in [action_policy_egoverse_3s_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)

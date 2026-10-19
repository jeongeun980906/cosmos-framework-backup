# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_egoverse_5s_nano`` — 5-second variant of ``action_policy_egoverse_nano``.

Identical recipe, but each transition spans the full **5s** retrieval window
(149 frames, chunk_length 148, ≡1 mod 4) instead of the leading 1s (29 frames).
5s has ~4.75x the vision latent frames per item (latent_t 38 vs 8), so it is much
heavier — the batch is dropped accordingly and one sample already fills most of
the packing-token budget. Frame params live here (not as fragile CLI overrides)
so the 5s run can't silently fall back to 1s.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_egoverse_nano import (
    action_policy_egoverse_nano,
)

cs = ConfigStore.instance()

action_policy_egoverse_5s_nano = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_nano)),
    flags={"allow_objects": True},
)

# 5s @30fps = 149 frames (149 = 1 + 4*37). chunk_length = 148 deltas.
action_policy_egoverse_5s_nano["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"][
    "chunk_length"
] = 148
# 5s sample ≈ 6k packed tokens (latent_t 38 x patch 8x10 x 2 items), so the
# 45056 token budget fits ~7 — batch is count-limited, not token-limited. At
# HSDP shard=8 batch=2 used ~40GB/80 (~50%); 4 ≈ ~59GB (~74%). Bump to 5-6 if
# memory holds (watch peak_gpu_mem_reserved_gb).
action_policy_egoverse_5s_nano["dataloader_train"]["max_samples_per_batch"] = 5
action_policy_egoverse_5s_nano["job"]["name"] = "action_policy_egoverse_5s_nano"

# VAE must encode exactly 149 frames (set after the LazyDict, like the 1s recipe).
action_policy_egoverse_5s_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [149]


for _item in [action_policy_egoverse_5s_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)

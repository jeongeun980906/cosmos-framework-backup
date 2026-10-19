# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_gr1_posttrain_1x_nano`` — GR1 post-training on **1x-speed play demos only**.

Identical to ``action_policy_gr1_posttrain_5s_nano`` except the demo pool is the
``1x``-speed ``SinkEnvForPlay`` matches only (``egoverse_pairs_gr1_1x``, built by
``tools/build_gr1_posttrain_pairs.py --speeds 1x``). Paired with
``action_policy_gr1_posttrain_2x_nano`` so the two play speeds can be trained as
two SEPARATE runs and compared. Everything else (chunk 32, fps 20, use_state,
warm-start from the EgoVerse 10k checkpoint) is inherited unchanged.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_gr1_posttrain_5s_nano import (
    action_policy_gr1_posttrain_5s_nano,
)

cs = ConfigStore.instance()

action_policy_gr1_posttrain_1x_nano = LazyDict(
    copy.deepcopy(dict(action_policy_gr1_posttrain_5s_nano)),
    flags={"allow_objects": True},
)

_cfg = action_policy_gr1_posttrain_1x_nano
_cfg["job"]["name"] = "action_policy_gr1_posttrain_1x_nano"
# 1x-only demo pool (env override still honored; distinct default).
_cfg["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]["pairs_dir"] = (
    "${oc.env:GR1_PAIRS_DIR,/mnt/ddn/dataset/egoverse_pairs_gr1_1x}"
)


for _item in [action_policy_gr1_posttrain_1x_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)

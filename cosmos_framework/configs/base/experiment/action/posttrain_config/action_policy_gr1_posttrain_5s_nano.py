# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_gr1_posttrain_5s_nano`` — GR1 post-training stage.

Second-stage fine-tuning that **continues from the EgoVerse-trained 5s checkpoint**
(not the base) and adapts the policy to the **GR1 humanoid robot**: each sample is
a GR1 query (the current embodiment) conditioned on a ``SinkEnvForPlay`` *play*
demo (1x or 2x speed), built by ``tools/build_gr1_posttrain_pairs.py`` into
``egoverse_pairs_gr1``. GR1 and the play demos are both 20 Hz / ``action57`` /
``hand_pose`` domain, so a 5 s window is **97 frames** (vs 149 for the 30 Hz human
rigs) — the reader derives this per-embodiment from ``EgoVerseEmbSpec.fps``.

Differences vs ``action_policy_egoverse_5s_nano``:
  * ``pairs_dir`` → ``egoverse_pairs_gr1`` (GR1 query → play demo).
  * ``encode_exact_durations = [97]`` (20 Hz 5 s clip; all items are 97 frames).
  * checkpoint ``keys_to_skip_loading = []`` — LOAD the trained generation + action
    heads from the EgoVerse checkpoint (the base recipe skips them to init fresh;
    here we warm-start from the already-trained policy). ``load_path`` is supplied
    via TOML/env and must point at an EgoVerse 5s checkpoint dir (e.g.
    ``.../action_policy_egoverse_5s_repro/checkpoints/iter_000050000``), NOT the
    Cosmos3-Nano base.

Usage (4 node x 8 GPU)::

    GR1_PAIRS_DIR=/mnt/ddn/dataset/egoverse_pairs_gr1 \\
    BASE_CHECKPOINT_PATH=<EgoVerse 5s checkpoint dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun ... -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_gr1_posttrain_5s_repro.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_egoverse_5s_nano import (
    action_policy_egoverse_5s_nano,
)

cs = ConfigStore.instance()

action_policy_gr1_posttrain_5s_nano = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_5s_nano)),
    flags={"allow_objects": True},
)

_cfg = action_policy_gr1_posttrain_5s_nano
_cfg["job"]["name"] = "action_policy_gr1_posttrain_5s_nano"

_ds = _cfg["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
# GR1 query → play demo pairs (own env knob so it can't collide with the EgoVerse run).
_ds["pairs_dir"] = "${oc.env:GR1_PAIRS_DIR,/mnt/ddn/dataset/egoverse_pairs_gr1}"
# Post-training drops the speed (meta) conditioning — caption is the instruction +
# embodiments only ("<instruction> demo <emb> target <emb>"). (GR1 also has no speed
# stats yet.) The EgoVerse run keeps speed; this just turns it off for GR1.
_ds["speed_stats_path"] = None

# Action chunk = 32 steps. GR1 + play are BOTH 20 Hz, so set the reference fps to 20
# (the EgoVerse run uses 30 because the human rigs are 30 Hz); then chunk_length is
# literal — 32 deltas → 33 frames (= 1 + 4*8, VAE-legal) ≈ 1.65 s @20 Hz for every
# item. (NOTE: this makes the clip ~1.65 s, not 5 s — the "5s" in the name is now
# nominal; the chunk size, not the window, is what changed.)
_ds["chunk_length"] = 32
_ds["fps"] = 20.0
_cfg["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]

# GR1 post-training conditions on the robot's STATE: prepend the absolute state0 as
# the conditioning ``action[0]`` (normalized with the per-emb ``<emb>_state`` stats).
# Action tensor becomes (chunk+1, dim) = (33, 57) per clip — state0 + 32 deltas — for
# BOTH the demo and the query segment. (The EgoVerse pre-train run used use_state=False;
# state was dropped in v1. This anchors the delta trajectory with the start pose.)
_ds["use_state"] = True

# concat_view: post-training reads 3 agentviews (center + sideleft + sideright, vertically
# concat'd by the reader) for BOTH the play demo (mimicdroid) and the GR1 query — added as
# images.sideleft/sideright by tools/add_concat_view_3cam.py. The egoverse PRE-train leaves
# this unset, so it reads only images.front_1 (center) from the same zarrs.
_THREE_CAMS = ("images.front_1", "images.sideleft", "images.sideright")
_ds["cams_override"] = {"mimicdroid": _THREE_CAMS, "gr1": _THREE_CAMS}

# Per-frame coverage: the GR1 query windows are built on a STRIDE-frame grid (w0 =
# 0, 4, 8, … — see tools/build_gr1_posttrain_pairs.py / 75_future_pairs.py, FUT=32),
# so by default only every 4th frame is ever a training query. Setting per_frame_stride
# = the query-boundary stride makes the loader draw a sub-stride offset δ ∈ {0..STRIDE-1}
# per sample and advance BOTH the query target and its retrieved demo by the SAME δ
# (lockstep), so every frame becomes a reachable example across epochs. Only this GR1
# path sets it; everywhere else it stays 1 (δ≡0 ⇒ unchanged behavior).
_GR1_QUERY_STRIDE = 4  # = w0 boundary spacing of the GR1 future-pairs (75_future_pairs STRIDE)
_ds["per_frame_stride"] = _GR1_QUERY_STRIDE

# 33-frame clips are ~4x lighter than the 5s run (latent_t 9 vs 38), so pack more
# samples per batch (the 5s recipe was count-limited at 5 for the heavy 149f clips).
_cfg["dataloader_train"]["max_samples_per_batch"] = 16

# Continue from the EgoVerse-trained policy: load the generation + action heads too
# (the base recipe skips them to start fresh; post-training warm-starts from them).
_cfg["checkpoint"]["keys_to_skip_loading"] = []
_cfg["checkpoint"]["load_training_state"] = False  # fresh optimizer/scheduler for the new stage


for _item in [action_policy_gr1_posttrain_5s_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)

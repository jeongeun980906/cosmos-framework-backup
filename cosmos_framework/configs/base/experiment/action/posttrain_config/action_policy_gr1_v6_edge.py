# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GR1 **v6** post-training on **Cosmos3-Edge** — ICL vs BC comparison pair.

Two experiments over the SAME v6 data (``egoverse_pairs_gr1_v6``: 61,686 GR1
query windows on a stride-1 frame grid, K=6 retrieved demos each, all-DTW
pairing, 4-cam gen episodes under ``dataset/egoverse``):

  * ``action_policy_gr1_v6_icl_edge`` — **ICL** (in-context / retrieval arm).
    Each sample is ``[demo, query]``: the retrieved play/taskdemo clip (video +
    action, fully clean) conditions the GR1 query (frame-0 observation + state0
    row clean, 32-step action chunk + video generated). Multi-segment shared-grid
    packing — the same machinery as ``action_policy_gr1_posttrain_*_nano``.
  * ``action_policy_gr1_v6_bc_edge`` — **BC** (plain arm). The demo item is
    dropped (``plain=True``): query-only single-clip samples, standard packing
    path. The demo-sampling RNG and the per-frame δ draw still run identically,
    so BC sees the byte-identical query stream — the ONLY difference vs ICL is
    the demo conditioning. (Following the cosmos_policy ablation doctrine:
    "plain = no retrieval, NOT no video" — the query's future-video generation
    block is kept in both arms.)

Recipe deltas vs the nano GR1 chain:
  * ``model.config`` = ``EDGE_MODEL_CONFIG`` (Nemotron-2B-Dense-VL backbone,
    Cosmos3-Edge loss scales, 480 native res). Edge's renewed (2026-07-16+)
    checkpoints ship trained action heads, so the action head warm-starts from
    the Edge base exactly like the nano recipe does from Cosmos3-Nano
    (``keys_to_skip_loading=["net_ema."]``).
  * ``pairs_dir`` → ``egoverse_pairs_gr1_v6`` (env ``GR1_V6_PAIRS_DIR``).
  * ``per_frame_stride = 1`` — v6 windows are ALREADY per-frame (w0 diffs = 1);
    the stride-4 sub-offset draw of the older gr1 manifests must be off.
  * Both arms keep chunk 32 @20 Hz (33 frames, VAE-legal), ``use_state=True``
    (state0 conditioning row), 3-cam concat view, per-emb quantile norm from
    ``egoverse_normalization.json`` — the proven framework GR1 recipe.

Usage (1 node, N GPU)::

    GR1_V6_PAIRS_DIR=/mnt/ddn/dataset/egoverse_pairs_gr1_v6 \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Edge DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_gr1_v6_icl_edge.toml
    # BC arm: ..._bc_edge.toml (same env knobs)
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_gr1_posttrain_5s_nano import (
    action_policy_gr1_posttrain_5s_nano,
)

cs = ConfigStore.instance()


def _make_gr1_v6_edge(name: str, *, plain: bool) -> LazyDict:
    cfg = LazyDict(
        copy.deepcopy(dict(action_policy_gr1_posttrain_5s_nano)),
        flags={"allow_objects": True},
    )
    cfg["job"]["name"] = name

    # ---- Edge backbone -----------------------------------------------------
    # Swap the whole model.config for the Edge baseline (Nemotron-2B-Dense-VL,
    # Edge loss scales, action_gen=True default), then re-pin the recipe's VAE
    # encode duration (33 frames = 1 + 4*8, chunk 32 @20 Hz).
    cfg["model"]["config"] = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]

    # ---- v6 data -----------------------------------------------------------
    ds = cfg["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
    ds["pairs_dir"] = "${oc.env:GR1_V6_PAIRS_DIR,/mnt/ddn/dataset/egoverse_pairs_gr1_v6}"
    # v6 windows sit on a stride-1 frame grid — every frame is already a query
    # row, so the sub-stride δ draw must be OFF (the older gr1 manifests were
    # stride-4 and needed it).
    ds["per_frame_stride"] = 1
    # v6 target = ABSOLUTE command chunk (abs_action57 rows, act[t] → s[t+1]),
    # NOT measured-state deltas — the state-chunk prediction pivot. Normalized
    # with the v6 p1/p99 stats (``<emb>_abs`` blocks in egoverse_normalization
    # .json, sourced from egoverse_pairs_gr1_v6/abs_action57_stats_pctl.json;
    # "quantile" over q01=lo/q99=hi reproduces the v6 pctl-minmax exactly).
    ds["action_source"] = "abs"
    # BC (plain) arm: query-only samples; ICL keeps the paired default.
    ds["plain"] = bool(plain)
    # tokenizer_config interpolation follows the swapped model config
    # ("${model.config.vlm_config.tokenizer}" → Edge processor) automatically.

    # ---- schedule: match the cosmos_policy v6 post-train ------------------
    # max_iter 8K = cycle 8K (no LR cliff), warmup 500, peak lr 1e-4 with
    # linear decay to 0. (LambdaLinear multiplies optimizer.lr by the lambda,
    # so peak = lr × f_max = 1e-4.) Action-head 5x multipliers inherited from
    # the DROID-proven framework recipe stay (head-specific, orthogonal).
    cfg["optimizer"]["lr"] = 1.0e-04
    cfg["scheduler"]["cycle_lengths"] = [8000]
    cfg["scheduler"]["warm_up_steps"] = [500]
    cfg["scheduler"]["f_max"] = [1.0]
    cfg["scheduler"]["f_min"] = [0.0]
    cfg["scheduler"]["f_start"] = [0.0]
    cfg["trainer"]["max_iter"] = 8000
    cfg["checkpoint"]["save_iter"] = 1000

    # ---- warm-start from the Edge BASE (not a mid checkpoint) --------------
    # The nano GR1 chain loads a trained EgoVerse mid checkpoint
    # (keys_to_skip_loading=[]); here both arms start from the public
    # Cosmos3-Edge base, whose renewed checkpoints carry trained action heads —
    # warm-start them (skip only EMA), like the egoverse_nano base recipe.
    cfg["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
    cfg["checkpoint"]["load_path"] = "???"  # Cosmos3-Edge DCP dir; supply via TOML/env

    # BC samples carry ~half the tokens of an ICL pair; keep max_samples_per_batch
    # IDENTICAL anyway so the two arms see the same number of queries per step
    # (fair comparison by data, not by tokens).
    return cfg


action_policy_gr1_v6_icl_edge = _make_gr1_v6_edge("action_policy_gr1_v6_icl_edge", plain=False)
action_policy_gr1_v6_bc_edge = _make_gr1_v6_edge("action_policy_gr1_v6_bc_edge", plain=True)


for _item in [action_policy_gr1_v6_icl_edge, action_policy_gr1_v6_bc_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)

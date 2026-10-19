# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_egoverse_v10_icl_edge`` — Cosmos3-**Edge** EgoVerse v10 (3s) ICL MID-TRAIN.

The missing first stage of the Edge GR1 pipeline: cross-embodiment retrieval
("ICL") mid-training on ``egoverse_pairs_v10`` (1.09M windows / 1.04M queries,
K=20; human rigs aria/scale/MECKA @30Hz + mimicdroid @20Hz), before the GR1 v6
post-train. Motivation (2026-08-03 eval verdict): the no-mid Edge ICL arm sits
at the 2.5 no-mid floor on Plate-L1 (20–30% vs mid-driven 83% in 2.5) — mid is
the proven lever, not longer post-training.

Recipe = ``action_policy_egoverse_3s_nano`` layout (chunk 88 → 89f human / 57f
MD, paired shared-grid multi-segment packing, speed conditioning ON), TARGET =
**절대 next-state57 chunk** (2.5 SMID; ``action_source="state"`` + ``use_state``
s[t0] 조건행). 2026-08-06 delta로 잠시 전환했다가 재전환: delta run에서 action
성분 하강을 확인했고, state 타깃의 1차 정체/스파이크 원인이던 corrupt 타깃은
reader 3중 방어(퇴화-stats 가드 + droplist + 정규화 ±10 클램프)로 봉쇄됨.
Demo 세그먼트는 2.5 계약대로 항상 video + delta action만 읽음 (state는 쿼리 전용).

Schedule: max_iter 10K = cycle 10K (no LR cliff), warmup 500, peak lr 1e-4
linear→0, DCP save 1K.

Usage (3 node x 8 GPU)::

    EGOVERSE_DATA_ROOT=/mnt/ddn/dataset \\
    EGOVERSE_PAIRS_DIR=/mnt/ddn/dataset/egoverse_pairs_v10 \\
    EDGE_LOAD_PATH=/mnt/ddn/ckpt/cosmos3_edge_dcp \\
    WAN_VAE_PATH=/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth \\
    torchrun ... -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_egoverse_v10_icl_edge.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_robomind_sft_dataset,
    get_action_robomind_icl_sft_dataset,
)

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_egoverse_3s_nano import (
    action_policy_egoverse_3s_nano,
)

cs = ConfigStore.instance()

action_policy_egoverse_v10_icl_edge = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_3s_nano)),
    flags={"allow_objects": True},
)

_cfg = action_policy_egoverse_v10_icl_edge
_cfg["job"]["name"] = "action_policy_egoverse_v10_icl_edge"

# ---- Edge backbone (same swap as the GR1 v6 Edge configs) -------------------
_cfg["model"]["config"] = copy.deepcopy(EDGE_MODEL_CONFIG)
_cfg["model"]["config"]["tokenizer"]["encode_exact_durations"] = [89]  # 3s human reference

# ---- v10 pairs (default; env still wins) ------------------------------------
_ds = _cfg["dataloader_train"]["dataloader"]["datasets"]["egoverse"]["dataset"]
_ds["pairs_dir"] = "${oc.env:EGOVERSE_PAIRS_DIR,/mnt/ddn/dataset/egoverse_pairs_v10}"
# STATE target (2026-08-06 재전환, 유저 결정): 2.5 SMID 그대로 — 절대 next-state57
# chunk + s[t0] 조건행. 1차 state 런의 1.0 정체/스파이크 원인이던 corrupt 타깃은
# 이제 reader의 3중 방어(퇴화-stats 가드 + droplist + **정규화 ±10 클램프**)로
# 봉쇄됨 (최악 배치 loss ~1e3 상한, 폭탄 불가). delta run은 action 성분 하강
# 확인 후 전환. corrupt state 13.2% eps는 클램프된 노이즈로 남음(v10 raw 유지 결정).
_ds["action_source"] = "state"
_ds["use_state"] = True
# zero-head9 MECKA 결함 에피소드 31개 제외 (쿼리+데모; 2026-08-06 loss 스파이크 규명).
_ds["droplist_path"] = "cosmos_framework/data/generator/action/datasets/egoverse_v10_droplist.json"

# ---- schedule: mid 10K, same shape as the v6 post-train (no LR cliff) -------
_cfg["optimizer"]["lr"] = 1.0e-04
_cfg["scheduler"]["cycle_lengths"] = [10000]
_cfg["scheduler"]["warm_up_steps"] = [500]
_cfg["scheduler"]["f_max"] = [1.0]
_cfg["scheduler"]["f_min"] = [0.0]
_cfg["scheduler"]["f_start"] = [0.0]
_cfg["trainer"]["max_iter"] = 10000
_cfg["checkpoint"]["save_iter"] = 1000
# Warm-start the action heads from the Edge base (renewed ckpts ship them).
_cfg["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
_cfg["checkpoint"]["load_path"] = "???"  # Cosmos3-Edge DCP dir; supply via TOML/env


# ---- GR1 v6 post-train FROM-MID variants ------------------------------------
# Identical to action_policy_gr1_v6_{icl,bc}_edge except the checkpoint loads a
# TRAINED Edge mid (generation + action heads INCLUDED: keys_to_skip_loading=[]),
# mirroring the nano GR1 chain's mid warm-start semantics.
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_gr1_v6_edge import (  # noqa: E402
    action_policy_gr1_v6_icl_edge,
    action_policy_gr1_v6_bc_edge,
)

action_policy_gr1_v6_icl_edge_frommid = LazyDict(
    copy.deepcopy(dict(action_policy_gr1_v6_icl_edge)), flags={"allow_objects": True}
)
action_policy_gr1_v6_icl_edge_frommid["job"]["name"] = "action_policy_gr1_v6_icl_edge_frommid"
action_policy_gr1_v6_icl_edge_frommid["checkpoint"]["keys_to_skip_loading"] = []

action_policy_gr1_v6_bc_edge_frommid = LazyDict(
    copy.deepcopy(dict(action_policy_gr1_v6_bc_edge)), flags={"allow_objects": True}
)
action_policy_gr1_v6_bc_edge_frommid["job"]["name"] = "action_policy_gr1_v6_bc_edge_frommid"
action_policy_gr1_v6_bc_edge_frommid["checkpoint"]["keys_to_skip_loading"] = []



# =====================================================================================
# BCMIX — the same ICL mid-train, plus two RoboMIND BC streams.
#
# The problem this addresses, measured: the v10 pool that actually trains is 98.4% human
# hands (MECKA/scale/aria, 1,042,181 windows) and 1.6% robot (mimicdroid, 16,485); eva is
# indexed but carries keep=0 in every pair set, so it has never trained. A backbone that
# barely sees a parallel gripper is then asked for one at RoboCasa post-train, and the
# 20K cluster curves show it losing to the public base at every step (0.0035 vs 0.0034 at
# 15-20K, paired t=+9.5) with the deficit growing monotonically in EgoVerse iters
# (2000 -> 5000 -> 8000 gives 0.9920 -> 1.0074 -> 1.0341 on the 400-step proxy).
#
# So: keep two of the rows the PUBLIC BASE already trained alive during mid-training,
# rather than letting them decay. Domain 8 (robomind-franka, 10-D) and domain 12
# (robomind-agilex-dual, 20-D) both carry non-zero bias in the base at exactly their
# widths. This is replay against forgetting, not a new row to fill.
#
# Sizing, by frames against the EgoVerse pool's 32,436,920:
#   franka   80 tasks / 11,480 eps / 1,610,770 frames = 4.97%
#   agilex   32 tasks /  2,785 eps / 1,606,647 frames = 4.95%
#   total                            3,217,417 frames = 9.92%
#
# The ratios are 14/1/1, not 90/5/5, because RankPartitionedDataLoader hands out WHOLE
# RANKS rather than mixing per sample: it takes ideal = ratio/sum * world_size, floors
# it (min 1), then distributes the leftover ranks by largest remainder. One rank of 32
# is 3.125%, so a 5% share is not representable and 90/5/5 resolves to [29, 2, 1] --
# 6.25% franka against 3.125% agilex, lopsided even though the two datasets are the same
# size on disk. 14/1/1 divides exactly: [28, 2, 2] = 87.5 / 6.25 / 6.25, so the two BC
# streams stay equal, which is the property worth having. The cost is that the BC share
# is 12.5% rather than 9.4%.
#
# mode="wam" on both: first frame in, video and actions jointly denoised out — the same
# contract RoboCasa post-train uses. The point is to preserve that input/output shape for
# a gripper embodiment, so sampling forward/inverse dynamics here would dilute it.
#
# view20_prob puts a slice of mimicdroid through the 57->20 projection onto domain 12 as
# well, so the 20-D row gets EgoVerse motion too and not only AgileX. It stays a slice:
# mimicdroid is the only robot data domain 3 has, and moving all of it out would leave
# the 57-D signal 100% human hands.
# =====================================================================================
action_policy_egoverse_v10_icl_edge_bcmix = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_v10_icl_edge)), flags={"allow_objects": True}
)
_bc = action_policy_egoverse_v10_icl_edge_bcmix
_bc["job"]["name"] = "action_policy_egoverse_v10_icl_edge_bcmix"

_bc_sets = _bc["dataloader_train"]["dataloader"]["datasets"]
_bc_sets["egoverse"]["ratio"] = 14
_bc_sets["egoverse"]["dataset"]["view20_embs"] = ("mimicdroid",)
_bc_sets["egoverse"]["dataset"]["view20_domain"] = "robomind-agilex-dual"   # domain 12
_bc_sets["egoverse"]["dataset"]["view20_prob"] = 0.25

_ROBOMIND_CHUNK = 16   # RoboMIND is 30 fps; ~0.53 s of action per sample

# (dataloader entry name, dataset dir under ROBOMIND_DATA_ROOT, head row, rank ratio)
_ROBOMIND_STREAMS = (
    ("robomind_franka", "robomind_franka_3rgb", "robomind-franka", 1),
    ("robomind_agilex", "robomind_agilex_3rgb", "robomind-agilex-dual", 1),
)


def _robomind_entry(factory, data_dir: str, emb: str, ratio: int, **extra):
    """One RoboMIND dataloader entry. ``factory`` picks BC or ICL; nothing else moves,
    so the two arms differ in the sample LAYOUT and in nothing else."""
    return dict(
        ratio=ratio,
        dataset=L(factory)(
            root="${oc.env:ROBOMIND_DATA_ROOT,/mnt/ddn/dataset}/" + data_dir,
            embodiment_type=emb,
            fps=30.0,
            chunk_length=_ROBOMIND_CHUNK,
            mode="wam",
            viewpoint="concat_view",
            action_normalization="quantile",
            resolution="256",
            max_action_dim="${model.config.max_action_dim}",
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            cfg_dropout_rate=0.1,
            **extra,
        ),
    )


for _name, _dir, _emb, _ratio in _ROBOMIND_STREAMS:
    _bc_sets[_name] = _robomind_entry(get_action_robomind_sft_dataset, _dir, _emb, _ratio)



# Two-stream variant, for smoke-testing on a box with fewer GPUs than streams.
# RankPartitionedDataLoader assigns whole ranks to datasets, so it asserts
# world_size >= len(datasets); the 3-stream config therefore needs >= 3 ranks (fine at
# 3 nodes x 8 GPU = 24, where 90/5/5 lands as roughly 22/1/1 ranks). This drops the
# agilex stream so the multi-stream packing path can still be exercised on 2 GPUs.
action_policy_egoverse_v10_icl_edge_bcmix2 = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_v10_icl_edge_bcmix)), flags={"allow_objects": True}
)
action_policy_egoverse_v10_icl_edge_bcmix2["job"]["name"] = "action_policy_egoverse_v10_icl_edge_bcmix2"
del action_policy_egoverse_v10_icl_edge_bcmix2["dataloader_train"]["dataloader"]["datasets"]["robomind_agilex"]


# =====================================================================================
# ICLMIX — bcmix with the two RoboMIND streams switched from BC to ICL.
#
# What bcmix actually does, restated: 87.5% of the ranks train ICL (demo clean in context,
# query generated) and 12.5% train BC (one clip, no demo). So the replay that exists to
# keep domains 8 and 12 from decaying refreshes those rows under a contract the rest of the
# run never uses. Two things follow, and they pull in opposite directions:
#
#   * The gripper rows are kept alive, which is the point, and bcmix does shrink the
#     RoboCasa deficit (13.3% -> 3.7% over steps 100+).
#   * But the query-only pathway those BC ranks exercise is the pathway the dummyret arm
#     was built to test separately, so bcmix confounds "replay the gripper rows" with
#     "train the no-demo pathway". Whatever it measures is the sum of the two.
#
# This arm separates them. Same windows, same tasks, same normalization, same 14/1/1 rank
# split, same query ORDER (the ICL dataset keeps the BC stream's flat index space and
# strided shard, verified sample-for-sample) — only the demo slot is added. iclmix vs bcmix
# is therefore a single-variable read on the layout, and iclmix vs plain v10 is a clean read
# on the replay alone, with the layout held fixed across the whole run.
#
# The demo is retrieved through a per-task descriptor rather than a learned index: the
# republication is one LeRobot repo per task, so "same task" is "same repo", and the
# descriptor decides the two things left — which EPISODE and WHEN inside it.
#
# The episode is always a different one: a window from the query's own episode overlaps the
# query in time and is close enough to its answer that the model would copy it instead of
# reading it as a demonstration.
#
# The WHEN is matched on robot state, and it has to be. Drawing the window uniformly puts
# the demo at a uniformly random phase of the task — measured on insert_marker, the median
# demo sat 0.34 of an episode away from the query's phase and moved in the OPPOSITE
# direction half the time, i.e. a 0.57 s clip of the release while the query is still
# approaching. Matching on t0 EE pose + gripper + net displacement (``demo_match="state"``,
# the default), over 8 candidate episodes, leaves 4.4% opposite on franka and 1.7% on
# agilex, with 64%/82% of demos inside 0.1 of the query's phase.
#
# Two things that were tried and are OFF, recorded so they are not re-tried blind:
#   * More key channels (wrist rotation, a mid-window path point) — flat to slightly worse
#     on every metric, both tasks. The key is not short of information.
#   * A task-progress channel (gripper closures so far), ``progress_weight`` — at weight 2
#     it moved 6 of 45 sampled pairs, buying cycle agreement and costing scene similarity.
#     The candidate-episode search fixed the cycle mismatch it was aimed at anyway.
#
# Absolute coordinates need no canonicalization here — demo and query are the same robot in
# the same repo. ``demo_match="random"`` restores the unmatched draw for the ablation.
#
# Cost: an ICL sample carries two 17-frame vision items instead of one, i.e. 10 latent
# frames against the BC sample's 5. That is still far under the EgoVerse pair's 46, so
# max_samples_per_batch=8 stays as-is.
# =====================================================================================
action_policy_egoverse_v10_icl_edge_iclmix = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_v10_icl_edge_bcmix)), flags={"allow_objects": True}
)
action_policy_egoverse_v10_icl_edge_iclmix["job"]["name"] = "action_policy_egoverse_v10_icl_edge_iclmix"
_icl_sets = action_policy_egoverse_v10_icl_edge_iclmix["dataloader_train"]["dataloader"]["datasets"]
for _name, _dir, _emb, _ratio in _ROBOMIND_STREAMS:
    _icl_sets[_name] = _robomind_entry(get_action_robomind_icl_sft_dataset, _dir, _emb, _ratio)

# Two-stream copy for smoke-testing on a 2-GPU box (see bcmix2).
action_policy_egoverse_v10_icl_edge_iclmix2 = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_v10_icl_edge_iclmix)), flags={"allow_objects": True}
)
action_policy_egoverse_v10_icl_edge_iclmix2["job"]["name"] = "action_policy_egoverse_v10_icl_edge_iclmix2"
del action_policy_egoverse_v10_icl_edge_iclmix2["dataloader_train"]["dataloader"]["datasets"]["robomind_agilex"]


# =====================================================================================
# EMPTY RETRIEVAL SLOT.
#
# What every arm measured so far has in common: the mid-trained backbone is WORSE on
# RoboCasa than the public base, at bb_lr matched to base, at every band, whether the
# post-train routes through domain 5 or domain 8, with or without the BC replay. The
# replay shrinks the deficit (13.3% -> 3.7% over steps 100+) but never removes it, and
# the deficit grows with mid-train length (bcmix 2000 -> 4000 costs +3.4%, t=+4.33).
# So the damage is done BY mid-training, and more of it is worse.
#
# The candidate mechanism this arm tests: ICL mid-training never shows the model the
# situation post-train actually presents. Every training sample carries an informative
# demo, so the model may route prediction through the demo and leave the query-only
# pathway underdeveloped. RoboCasa post-train is single-clip -- no demo at all -- and
# has to rebuild that pathway, which is exactly a per-step penalty that does not shrink
# with post-train steps and does grow with mid-train length. Both are what we measure.
#
# The intervention empties the demo slot on HALF the samples: the two-item layout, the
# caption grammar, and the packing path stay identical, and only the information in the
# slot is removed. Half rather than all, because the ICL objective is the point of the
# mid-train -- an all-dummy run is the `plain` BC arm, which already exists.
#
# ``dummy_ret_frames=1`` shortens the emptied clip to one frame: an empty demo carries
# the same nothing at any length, and the VAE encodes T=1 at latent_T=1 with no padding,
# so the sample costs 1 latent frame instead of 23. That roughly halves the vision tokens
# on dummy samples, which is also why the shared temporal grid is off for them.
# =====================================================================================
action_policy_egoverse_v10_icl_edge_dummyret = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_v10_icl_edge_bcmix)), flags={"allow_objects": True}
)
action_policy_egoverse_v10_icl_edge_dummyret["job"]["name"] = "action_policy_egoverse_v10_icl_edge_dummyret"
_dr = action_policy_egoverse_v10_icl_edge_dummyret["dataloader_train"]["dataloader"]["datasets"]
_dr["egoverse"]["dataset"]["dummy_ret_prob"] = 0.5
_dr["egoverse"]["dataset"]["dummy_ret_frames"] = 1

# Two-stream copy for smoke-testing on a 2-GPU box.
action_policy_egoverse_v10_icl_edge_dummyret2 = LazyDict(
    copy.deepcopy(dict(action_policy_egoverse_v10_icl_edge_dummyret)), flags={"allow_objects": True}
)
action_policy_egoverse_v10_icl_edge_dummyret2["job"]["name"] = "action_policy_egoverse_v10_icl_edge_dummyret2"
del action_policy_egoverse_v10_icl_edge_dummyret2["dataloader_train"]["dataloader"]["datasets"]["robomind_agilex"]


for _item in [
    action_policy_egoverse_v10_icl_edge,
    action_policy_egoverse_v10_icl_edge_bcmix,
    action_policy_egoverse_v10_icl_edge_bcmix2,
    action_policy_egoverse_v10_icl_edge_iclmix,
    action_policy_egoverse_v10_icl_edge_iclmix2,
    action_policy_egoverse_v10_icl_edge_dummyret,
    action_policy_egoverse_v10_icl_edge_dummyret2,
    action_policy_gr1_v6_icl_edge_frommid,
    action_policy_gr1_v6_bc_edge_frommid,
]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)

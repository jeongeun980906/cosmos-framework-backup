# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa 24-task post-training on **Cosmos3-Edge** — the Cosmos-Policy comparison arm.

Trains a Cosmos3-Edge action policy on the SAME data the Cosmos-Policy RoboCasa
runs use (``nvidia/RoboCasa-Cosmos-Policy`` ``success_only``: 24 kitchen tasks,
1199 demos, ~317k frames), read in place from HDF5 by
:class:`~cosmos_framework.data.generator.action.datasets.robocasa_hdf5_dataset.RoboCasaHDF5Dataset`.
The action target is min-max normalized with the release's own
``dataset_statistics.json`` min/max, so the two stacks optimize the same scale and
the eval numbers are directly comparable.

Recipe (mirrors ``action_policy_gr1_v6_bc_edge``, the proven Edge action recipe):

  * ``model.config`` = ``EDGE_MODEL_CONFIG`` (Nemotron-2B-Dense-VL backbone, Edge
    loss scales, 480 native res), with ``encode_exact_durations=[33]`` — chunk 32
    @20 Hz is 33 frames, which is VAE-legal (33 = 1 + 4*8).
  * **3 cameras** (left agent-view / right agent-view / wrist) vertically
    concatenated into one 672x224 frame — the Cosmos3 ``concat_view`` pattern, the
    same 3-high stack the GR1 Edge post-train uses. At ``resolution="480"`` that
    snaps to a 832x480 canvas, giving each view ~277 px (the GR1 recipe runs the
    same 3:1 stack at the 256 tier, which would cut each view to ~106 px — below
    the 224 px Cosmos-Policy feeds each camera, so this recipe pays the extra
    tokens to keep the comparison honest).
  * ``use_state=True`` — the absolute proprio ``[ee_pos(3), ee_ori(3), gripper]``
    is prepended as ``action[0]`` and the transform marks it clean conditioning
    (``condition_frame_indexes_action=[0]``). This is the framework's proprio path
    (DROID / PushT / GR1 all use it); Cosmos-Policy carries proprio in a dedicated
    latent slot instead, so this is the closest available equivalent.
  * Warm-start from the public **Cosmos3-Edge base**, whose renewed checkpoints
    ship trained action heads — skip only ``net_ema.`` (as the GR1/EgoVerse Edge
    recipes do), so the action head warm-starts instead of initializing fresh.

Architectural differences vs the Cosmos-Policy recipe — read any comparison with
these in mind. NOT reproduced here: the value-function head, the future-proprio
auxiliary target, the 50/50 rollout mixing from ``all_episodes``, and T5 text
conditioning (Edge conditions on the VLM prompt directly, so no
``t5_embeddings.pkl`` is needed). Going the other way, the future-VIDEO target is
RICHER here: ``wam`` conditions on frame 0 only, so all 32 remaining frames of the
3-camera stack are generated and supervised, where the Cosmos-Policy layout predicts
one future frame per camera at ``t + chunk_size``.

Usage (1 node, N GPU)::

    ROBOCASA_ROOT=/mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only \\
    EDGE_LOAD_PATH=/mnt/ddn/ckpt/cosmos3_edge_dcp \\      # DCP parent; loader adds /model
    WAN_VAE_PATH=/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth \\
    IMAGINAIRE_OUTPUT_ROOT=/mnt/ddn/tmp/robocasa24_edge \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_robocasa_edge.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_aloha_tabletop_sft_dataset,
    get_action_robocasa_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

# chunk 32 @20 Hz -> 33 observation frames. Must be ≡1 mod 4 for the 4x
# temporal-compression tokenizer: 33 = 1 + 4*8.
_CHUNK_LENGTH = 32
_ENCODE_DURATION = _CHUNK_LENGTH + 1


def _robocasa_edge_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["tokenizer"]["encode_exact_durations"] = [_ENCODE_DURATION]
    return cfg


action_policy_robocasa_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action",
            group="action_sft",
            name="action_policy_robocasa_edge",
            wandb_mode="online",
        ),
        model=dict(
            config=_robocasa_edge_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-04,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[20000],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[500],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=50,
            max_iter=20000,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # The renewed Cosmos3-Edge base ships trained action heads, so warm-start
            # them (skip only EMA, which re-seeds from net) — same as the GR1/EgoVerse
            # Edge recipes.
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Edge DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=1000,
            strict_resume=False,  # base init: tolerate key set differences
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_robocasa",
            # Token budget (measured): the 672x224 3-cam stack snaps to a 832x480
            # canvas at the 480 tier -> 9 latent frames x 390 tokens = 3510 video
            # tokens per sample, so 16 samples = 56.2k tokens per rank. That is the
            # same per-rank sample count the GR1 Edge recipe uses, but ~6.5x the
            # sequence length (its 3-cam stack runs at the 256 tier, 540 tok/sample),
            # so this is a MEMORY bound, not a config one: ``max_num_tokens_after_packing``
            # is never read at runtime here (it only feeds ``max_sequence_length`` in
            # recipes that enable packing, and packing is off below), so the real
            # bound is memory. Measured on 2xH100 with shard 2 (a strict upper bound on
            # per-rank memory vs the 8-way production shard): 16 -> 5.0 s/iter, peak
            # 38.8 GB of 80; 25 -> 7.9 s/iter, 43.1 GB. Both fit; 16 is the setting.
            # Budget note: at 3 nodes this is global batch 384 -> 24.3 epochs over the
            # 315,796-window dataset, vs 37.9 for the Cosmos-Policy arm (20k iters at
            # global batch 600). Raising this to 25 would make all three of iter count,
            # global batch and epochs match, at ~44 h instead of ~28 h.
            max_samples_per_batch=16,
            max_sequence_length=None,  # None disables token packing
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                # Each sample reads ~15 MB of raw uint8 out of HDF5; more workers keep
                # the GPU fed, and the iterable shuffle keeps those reads sequential.
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    robocasa=dict(
                        ratio=1,
                        dataset=L(get_action_robocasa_sft_dataset)(
                            root="${oc.env:ROBOCASA_ROOT,/mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only}",
                            fps=20.0,
                            chunk_length=_CHUNK_LENGTH,
                            mode="wam",
                            use_state=True,
                            action_normalization="minmax",
                            task_split=None,  # all 24 tasks
                            pad_tail=True,
                            resolution="480",  # Edge native res
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            format_prompt_as_json=False,
                            # Prompt = CAPTION_PREFIX + task, nothing else, so the eval
                            # server reproduces it byte-for-byte. The optional appenders
                            # are all off on purpose:
                            #   viewpoint / duration-fps / resolution — CONSTANT for this
                            #     recipe (always 33 frames @20 Hz on an 832x480 canvas from
                            #     the same 3-cam view), so they carry no information, and
                            #     the server's resolution sentence measures the PRE-pad
                            #     frame while training measures the padded canvas.
                            #   idle_frames — computed from the ground-truth action chunk
                            #     (compute_idle_frames), which eval cannot know. Training on
                            #     it teaches the model a cue that vanishes at test time.
                            append_viewpoint_info=False,
                            append_duration_fps_timestamps=False,
                            append_resolution_info=False,
                            append_idle_frames=False,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            # The episode scan is ~3 s (metadata only), so no cache by
                            # default — writing an index file into the read-only dataset
                            # tree is not worth it. Set ROBOCASA_INDEX_CACHE to a writable
                            # path to enable it.
                            index_cache_path="${oc.env:ROBOCASA_INDEX_CACHE,null}",
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# Single-task smoke arm: PnPStoveToCounter only, 20 iters, tiny batch. Same code
# path as the full run so a config/dataloader break shows up in ~2 minutes.
action_policy_robocasa_edge_smoke = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge)),
    flags={"allow_objects": True},
)
action_policy_robocasa_edge_smoke["job"]["name"] = "action_policy_robocasa_edge_smoke"
action_policy_robocasa_edge_smoke["job"]["wandb_mode"] = "disabled"
action_policy_robocasa_edge_smoke["trainer"]["max_iter"] = 20
action_policy_robocasa_edge_smoke["trainer"]["logging_iter"] = 1
action_policy_robocasa_edge_smoke["checkpoint"]["save_iter"] = 1000000
action_policy_robocasa_edge_smoke["scheduler"]["cycle_lengths"] = [20]
action_policy_robocasa_edge_smoke["scheduler"]["warm_up_steps"] = [0]
action_policy_robocasa_edge_smoke["dataloader_train"]["max_samples_per_batch"] = 1
action_policy_robocasa_edge_smoke["dataloader_train"]["dataloader"]["num_workers"] = 2
action_policy_robocasa_edge_smoke["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"]["task_split"] = [
    "PnPStoveToCounter"
]
action_policy_robocasa_edge_smoke["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"][
    "index_cache_path"
] = None


# ---- FROM-MID variant: warm-start from OUR EgoVerse-trained Edge model ---------
# Identical to ``action_policy_robocasa_edge`` except the checkpoint loads a TRAINED
# Edge mid instead of the public Cosmos3-Edge base — generation AND action heads
# included (``keys_to_skip_loading=[]``), mirroring the GR1 v6 from-mid variants in
# ``action_policy_egoverse_v10_edge.py``. Point ``EGOVERSE_EDGE_LOAD_PATH`` at the
# mid DCP dir, e.g. the v10 ICL 8K:
#   /mnt/ddn/tmp/egoverse_v10_icl_edge_4node/cosmos3_action/action_sft/
#       action_policy_egoverse_v10_icl_edge/checkpoints/iter_000008000
#
# Caveat worth stating: unlike the GR1 from-mid pair (57-dim hand_pose -> 57-dim
# hand_pose), this crosses ACTION SPACES — EgoVerse trained the head on 57-dim
# hand_pose (domain_id 3), RoboCasa uses the 7-dim delta-EE command (domain_id 5).
# Shapes still line up (both pad to max_action_dim=64), so the head starts from
# hand_pose-tuned weights and re-adapts under its 5x lr multiplier; the transfer
# that this arm is really testing is the BACKBONE's. The embodiment embedding row
# for domain 5 is untouched by EgoVerse training (only domain 3 got gradients), so
# it still carries the Edge base value.
# ``load_training_state`` stays False (inherited), so this does NOT resume EgoVerse's
# optimizer/scheduler/iteration — it is a fresh 20K schedule off those weights.
# To instead reset the action heads, set
#   checkpoint.keys_to_skip_loading=["net_ema.","action2llm","llm2action","action_modality_embed","action_pos_embed"]
# — note that with a mid checkpoint "skip" means RANDOM init, not the Edge base's
# trained heads, so it is the weaker option unless you are isolating backbone transfer.
action_policy_robocasa_edge_frommid = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_frommid["job"]["name"] = "action_policy_robocasa_edge_frommid"
action_policy_robocasa_edge_frommid["checkpoint"]["keys_to_skip_loading"] = []


# ---- FROM-MID with the action heads RESET ------------------------------------
# Third arm, to explain why `..._frommid` starts ~1.4 higher than the base arm over
# the first 100 iters. Total loss is 10x(vision) + 10x(action), so that gap has two
# candidate causes and the two-arm pair cannot separate them:
#   (a) the EgoVerse action head is actively WRONG here — it was trained to spend 57
#       of its 64 channels on human hand_pose (domain 3); RoboCasa uses 7 delta-EE
#       channels (domain 5), so it starts further off than the public base's generic
#       head, and action_loss_weight=10 amplifies that.
#   (b) the EgoVerse BACKBONE simply does not transfer to RoboCasa.
# This arm keeps the EgoVerse backbone but drops the action heads, so they init
# fresh. If it converges to base-or-better, (a) was the story and EgoVerse transfer
# is real; if it stays worse than base, (b) is, and the mid arm is not worth pursuing.
#
# Expect the HIGHEST initial loss of the three — the heads start from random, not
# from the Edge base's trained ones. Judge this arm on final success rate, not on
# where its loss curve starts.
action_policy_robocasa_edge_frommid_headreset = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_frommid_headreset["job"]["name"] = "action_policy_robocasa_edge_frommid_headreset"
action_policy_robocasa_edge_frommid_headreset["checkpoint"]["keys_to_skip_loading"] = [
    "net_ema.",
    "action2llm",
    "llm2action",
    "action_modality_embed",
    "action_pos_embed",
]


# ---- BACKBONE-ONLY transfer: EgoVerse backbone + the PUBLIC Edge action heads ----
# Measured, not assumed: ``action2llm``/``llm2action`` are ``DomainAwareLinear``, an
# ``nn.Embedding`` of per-domain weight matrices, so a run only trains the domain rows
# it indexes. Diffing the EgoVerse 8K checkpoint against the Edge base row by row:
#   domain 3 (hand_pose, what EgoVerse trained)  action2llm 28.6%  llm2action 41.5%
#   domain 5 (libero/robocasa, what WE use)                 11.3%             11.3%
#   domain 0 (no_action, never indexed)                     11.3%             11.3%
# domain 5 moved exactly as much as a never-used row — i.e. weight decay only. So the
# mid arm's RoboCasa head is already the public head (scaled ~0.887), and the +1.4
# early-loss gap vs base canNOT be the action head. What EgoVerse DID rewrite is
# ``action_modality_embed`` (49.4%) — a single tensor shared across all domains.
#
# This arm therefore restores BOTH the per-domain heads and that shared embed to their
# public-Edge values while keeping the EgoVerse backbone, isolating backbone transfer.
# The loader reads one source, so ``EGOVERSE_EDGEHEAD_LOAD_PATH`` must point at a DCP
# built by ``tools/merge_action_head_dcp.py`` (verified: heads 0.00% vs Edge base,
# backbone 0.00% vs EgoVerse).
action_policy_robocasa_edge_frommid_backboneonly = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_frommid_backboneonly["job"]["name"] = (
    "action_policy_robocasa_edge_frommid_backboneonly"
)
action_policy_robocasa_edge_frommid_backboneonly["checkpoint"]["load_path"] = (
    "${oc.env:EGOVERSE_EDGEHEAD_LOAD_PATH}"
)

# MEASURED OUTCOME (10-iter local, 2 GPUs, same seed/data order, paired per iter):
#   base 12.926 | backboneonly 13.356 (+0.43, 10/10) | mid 13.399 (+0.47, 10/10)
#   | headreset 13.610 (+0.68, 10/10)
# backboneonly lands on TOP of mid (-0.04, 5/10 = noise), so ``action_modality_embed``
# is NOT the cause after all — the whole early-loss gap rides with the backbone.
# Two further facts from the same audit, both worth knowing before reading any of
# these arms:
#   * domain 5 is UNTRAINED in the public Edge base. ``llm2action.bias[5]`` is exactly
#     0.00000 (``nn.init.zeros_`` untouched — one gradient step makes that impossible)
#     and ``fc[5]`` sits at RMS 0.01213, inside the 0.01210-0.01217 band shared by the
#     18 never-indexed slots (9,10,11,14,16-31). The 10 trained rows are all >=0.0157
#     with nonzero bias. So EVERY arm here starts RoboCasa from an untrained head.
#   * that makes headreset nearly a no-op relative to mid on the heads (it only drops
#     them from the checkpoint's RMS 0.0121 to fresh Xavier ~0.0039), which matches its
#     measured +0.21 vs mid.


# ---- DOMAIN-3 head: the only arm that starts from a head someone actually trained ----
# Same full EgoVerse warm start as ``frommid``; the ONE change is that the dataset
# indexes domain 3 (``hand_pose``, the row EgoVerse trained) instead of domain 5.
# Nothing else moves: ``raw_action_dim`` stays 7 (it comes from the dataset's action
# spec, not the domain), the loss still slices ``sqerr[:, :7]``, the 57 padded channels
# are still zeros, and the normalization block is still ``robocasa``.
#
# The trade is explicit. Gain: a trained head instead of an untrained row. Costs:
#   * channel semantics differ. EgoVerse's 57D layout is
#     ``head9|Labs9|Lhand15|Rabs9|Rhand15`` with ``abs9=[pos(3),R0(3),R1(3)]``, so its
#     ch0-2 are head delta-position (loosely like our delta-EE position), ch3-5 are a
#     rotation-matrix column (vs our delta axis-angle), and ch6 is R1[0] -- against our
#     GRIPPER, the single channel that decides task success.
#   * output scale is 4.03x too large. Pushing h~N(0,1) through the two rows gives
#     output std 1.948 (domain 3) vs 0.484 (domain 5), against a target v=eps-x0 of
#     std ~1, so initial action MSE runs ~3-4x higher.
# Expect a WORSE starting loss than mid. Judge on final success rate; the heads carry a
# 5x lr multiplier and re-adapt quickly.
#
# EVAL: the checkpoint is only valid when queried on the same row, so the client MUST
# be run with ``--domain-name hand_pose``. The default (``robocasa`` -> 5) would read an
# untrained row and return garbage.
action_policy_robocasa_edge_frommid_domain3 = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_frommid_domain3["job"]["name"] = "action_policy_robocasa_edge_frommid_domain3"
action_policy_robocasa_edge_frommid_domain3["dataloader_train"]["dataloader"]["datasets"]["robocasa"][
    "dataset"
]["domain_name"] = "hand_pose"


# =====================================================================================
# DOMAIN-8 arm: same warm start as ``..._frommid``, but the action heads index
# DomainAwareLinear row 8 (robomind-franka / droid) instead of row 5 (robocasa).
#
# Why row 8. The 400-step RoboCasa proxy was run through every candidate row, and 8 is
# the only one that beats the default at BOTH ends of the curve:
#
#   step 300-400 (base = 1.000)      step 2-50
#     d8_base       0.917 t=-2.19      0.240   <- public base, row 8, NO EgoVerse
#     d8_bcmix 1K   0.913 t=-2.40      0.327
#     d8_bcmix 4K   0.957 t=-1.06      0.356
#     d8_bcmix 8K   0.965 t=-1.00      0.347
#
# Read that carefully before attributing anything to EgoVerse: the BEST dom8 arm is
# ``d8_base``, which never saw EgoVerse at all. The gain is pure routing. Row 5 is
# untrained in the public Edge base (llm2action.bias[5] is exactly 0.0, still at its
# ``zeros_`` init), while rows 8 is trained by robomind-franka/droid, so pointing
# RoboCasa at 8 starts from weights somebody actually fit. Layering the EgoVerse warm
# start on top makes it WORSE, consistently: d8_bcmix4000 - d8_base = +0.00856,
# t=+6.32 over steps 100+, and the early-band ratio degrades 0.240 -> 0.356.
#
# This arm exists to carry that 8K start into a real 20K run anyway, so the routing
# effect can be judged on success rate rather than on a 400-step proxy. If the goal is
# to isolate the routing win itself, run the same config off the PUBLIC base instead
# (set EGOVERSE_EDGE_LOAD_PATH to the base checkpoint) — the proxy says that arm is
# the stronger one.
#
# The action tensor stays 7-wide. domain_utils maps robomind-franka to dim 10, but the
# dataset's own ``raw_action_dim`` governs the tensor; the domain id only selects which
# embedding row the heads read. Same arrangement as the dom3 arm (57D map, 7D tensor).
#
# *** EVAL MUST PASS --domain-name robomind-franka ***
# The client defaults to "robocasa" -> row 5, which this run never trains. Querying an
# untrained row returns plausible-looking garbage rather than an error.
# =====================================================================================
action_policy_robocasa_edge_frommid_domain8 = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_frommid_domain8["job"]["name"] = "action_policy_robocasa_edge_frommid_domain8"
action_policy_robocasa_edge_frommid_domain8["dataloader_train"]["dataloader"]["datasets"]["robocasa"][
    "dataset"
]["domain_name"] = "robomind-franka"


# ---- DOMAIN-3 with the head rescaled to domain 5's magnitude ----
# ``dom3`` measured better at the start and WORSE later, and the cause is a scale
# mismatch, not the semantics. Row 3's output is 4.03x larger than row 5's, which
# inflates the gradient: pre-clip global grad norm 89.0 vs base's 16.8, consistently
# 4.1-6.9x across all 10 logged steps. ``grad_clip.clip_norm = 1.0`` is applied as ONE
# global scalar over every parameter (callbacks/grad_clip.py: "every mesh group is
# rescaled with that single global scalar"), so that inflation renormalises the whole
# step budget rather than just the head's share. On the cluster dom3 tracked base to
# +1.3% by iter 900-1200 and then fell back to +2.7% at 1200-1500 instead of closing.
#
# This arm keeps row 3's learned DIRECTION and rescales only its MAGNITUDE to row 5's
# RMS (action2llm rows 0..6 x0.569, llm2action cols 0..6 and its bias x0.249), via a
# one-time checkpoint edit in tools/rescale_domain_head_dcp.py. If the grad norm drops
# to base's range and the curve tracks, the -21% action-loss head start was real and
# was simply unusable under global clipping.
#
# EVAL still needs ``--domain-name hand_pose`` — the rescale does not change the row.
action_policy_robocasa_edge_frommid_domain3_rescaled = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid_domain3)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_frommid_domain3_rescaled["job"]["name"] = (
    "action_policy_robocasa_edge_frommid_domain3_rescaled"
)
action_policy_robocasa_edge_frommid_domain3_rescaled["checkpoint"]["load_path"] = (
    "${oc.env:EGOVERSE_DOM3SCALED_LOAD_PATH}"
)


# =====================================================================================
# Making the EgoVerse warm start actually WIN on action loss, start to finish.
#
# The problem, measured on the cluster (both arms ran the full 20K, gbs 384):
#   step    100     200     300     500    1000   2-5K   15-20K
#   base  0.4622  0.0460  0.0372  0.0318  0.0239  0.0123  0.0035
#   mid   0.5625  0.0569  0.0431  0.0350  0.0248  0.0126  0.0035
#   Δ     +21.7%  +23.9%  +15.8%  +10.0%   +4.1%   +2.6%   +2.2%
# mid is above base at EVERY logged step. dom3s (trained head, rescaled) is the only
# arm that ever starts lower (-14.1% at step 100) but is already +11.9% by step 200 --
# it descends less, not more. Both EgoVerse arms share one thing: the backbone.
#
# Where the damage happens: the first ~2K steps. Global grad norm there is 2.7-2.8,
# well above grad_clip.clip_norm=1.0, so the step IS renormalised and the loss weights
# decide how the unit budget splits between fixing vision and fixing action. By 5K the
# norm has fallen to 0.87 and clipping stops firing entirely, which is also where the
# gap stops shrinking (+2% and flat from 5K to 20K). So the early budget split is the
# lever, and it is only a lever early.
#
# Three independent knobs, one arm each so the effect of each is readable:
#   _aw  loss weights   — action 30 / vision 5 (6:1, from 1:1). Spends the clipped
#                         early budget on the action head instead of on re-learning
#                         kitchen video the EgoVerse backbone has not seen.
#   _lr  lr multipliers — head 10x (from 5x), backbone (moe_gen) 0.3x. Adapts the small
#                         head fast while keeping the pretrained backbone from being
#                         knocked out of its basin by a fresh 1e-4 with reset Adam
#                         moments (load_training_state=False resets them).
#   _wu  warm-up        — 2000 steps (from 500). A converged checkpoint re-entered with
#                         zero Adam state takes its worst steps early; a longer ramp
#                         lets the moment estimates settle before the steps get big.
# =====================================================================================


def _mid_variant(name: str) -> LazyDict:
    """A copy of the mid arm; caller mutates the one knob under test."""
    d = LazyDict(copy.deepcopy(dict(action_policy_robocasa_edge_frommid)), flags={"allow_objects": True})
    d["job"]["name"] = name
    return d


# ---- _aw: spend the clipped early budget on action, not video ----
action_policy_robocasa_edge_frommid_aw = _mid_variant("action_policy_robocasa_edge_frommid_aw")
action_policy_robocasa_edge_frommid_aw["model"]["config"]["rectified_flow_training_config"][
    "action_loss_weight"
] = 30.0
action_policy_robocasa_edge_frommid_aw["model"]["config"]["rectified_flow_training_config"][
    "loss_scale"
] = 5.0

# ---- _lr: fast head, slow backbone ----
action_policy_robocasa_edge_frommid_lr = _mid_variant("action_policy_robocasa_edge_frommid_lr")
action_policy_robocasa_edge_frommid_lr["optimizer"]["lr_multipliers"] = {
    "action2llm": 10.0,
    "llm2action": 10.0,
    "action_modality_embed": 10.0,
    "moe_gen": 0.3,
}

# ---- _wu: longer warm-up so reset Adam moments settle first ----
action_policy_robocasa_edge_frommid_wu = _mid_variant("action_policy_robocasa_edge_frommid_wu")
action_policy_robocasa_edge_frommid_wu["scheduler"]["warm_up_steps"] = [2000]


# ---- dom3 + the _aw weights: keep the lowest start there is, then make it descend ----
# dom3 (UNSCALED domain-3 head) is the only arm that starts below base -- 0.3971 vs
# 0.4622 at step 100, -14.1%. Its problem was never the start; it was that the descent
# stalls. Rescaling (dom3s) fixed its grad norm (89 -> 17 locally, 0.93-1.14x base on
# the cluster) but also gave back part of the low start. This arm keeps dom3's original
# head AND shifts the early budget toward action, so the low start has something to
# descend with.
action_policy_robocasa_edge_frommid_domain3_aw = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid_domain3)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_frommid_domain3_aw["job"]["name"] = (
    "action_policy_robocasa_edge_frommid_domain3_aw"
)
action_policy_robocasa_edge_frommid_domain3_aw["model"]["config"]["rectified_flow_training_config"][
    "action_loss_weight"
] = 30.0
action_policy_robocasa_edge_frommid_domain3_aw["model"]["config"]["rectified_flow_training_config"][
    "loss_scale"
] = 5.0

# ---- same, on the rescaled head ----
action_policy_robocasa_edge_frommid_domain3s_aw = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid_domain3_rescaled)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_frommid_domain3s_aw["job"]["name"] = (
    "action_policy_robocasa_edge_frommid_domain3s_aw"
)
action_policy_robocasa_edge_frommid_domain3s_aw["model"]["config"]["rectified_flow_training_config"][
    "action_loss_weight"
] = 30.0
action_policy_robocasa_edge_frommid_domain3s_aw["model"]["config"]["rectified_flow_training_config"][
    "loss_scale"
] = 5.0


# =====================================================================================
# SWEEP arm — every knob under test reads from an env var, so an outer search loop can
# try configurations without editing or re-registering anything.
#
# Goal it exists to serve: get an EgoVerse-warm-started run BELOW base on action loss
# at every logged step. Nothing has managed it yet. Measured on the cluster (gbs 384):
#   step        100     200     300     500    1000   15-20K
#   base     0.4622  0.0460  0.0372  0.0318  0.0239   0.0035
#   mid      0.5625  0.0569  0.0431  0.0350  0.0248   0.0035   (+21.7% .. +2.2%)
#   dom3s    0.3971  0.0514  0.0407       -       -        -   (-14.1%, then +11.9%)
# dom3 starts lowest and stalls; mid never leads. Both share the backbone.
#
# Knobs (env var, default = the mid recipe):
#   SW_LOAD       checkpoint dir                    (no default - must be set)
#   SW_DOMAIN     robocasa (row 5) | hand_pose (3)  robocasa
#   SW_AW         action_loss_weight                10.0
#   SW_VW         loss_scale (vision)               10.0
#   SW_HEAD_LR    lr multiplier on the action head  5.0
#   SW_BB_LR      lr multiplier on moe_gen backbone 1.0
#   SW_WARMUP     scheduler warm_up_steps           500
# =====================================================================================
action_policy_robocasa_edge_sweep = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_sweep["job"]["name"] = "action_policy_robocasa_edge_sweep"
action_policy_robocasa_edge_sweep["checkpoint"]["load_path"] = "${oc.env:SW_LOAD}"
action_policy_robocasa_edge_sweep["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"][
    "domain_name"
] = "${oc.env:SW_DOMAIN,robocasa}"
action_policy_robocasa_edge_sweep["model"]["config"]["rectified_flow_training_config"][
    "action_loss_weight"
] = "${oc.decode:${oc.env:SW_AW,10.0}}"
action_policy_robocasa_edge_sweep["model"]["config"]["rectified_flow_training_config"][
    "loss_scale"
] = "${oc.decode:${oc.env:SW_VW,10.0}}"
action_policy_robocasa_edge_sweep["optimizer"]["lr_multipliers"] = {
    "action2llm": "${oc.decode:${oc.env:SW_HEAD_LR,5.0}}",
    "llm2action": "${oc.decode:${oc.env:SW_HEAD_LR,5.0}}",
    "action_modality_embed": "${oc.decode:${oc.env:SW_HEAD_LR,5.0}}",
    "moe_gen": "${oc.decode:${oc.env:SW_BB_LR,1.0}}",
}
action_policy_robocasa_edge_sweep["scheduler"]["warm_up_steps"] = ["${oc.decode:${oc.env:SW_WARMUP,500}}"]
# Optimizer knobs. These matter more than the head initialisation: measured on the
# cluster, lowering the START of the action loss does not survive 100 steps --
#   step        100      200     fall
#   base     0.4622   0.0460   90.1%
#   mid      0.5625   0.0569   89.9%
#   dom3s    0.3971   0.0514   87.0%
#   dom3aw   0.2711   0.0536   80.2%   <- lowest start, WORST fall, ends above base
# All four land in 0.046-0.057 whatever they start from, and the lower the start the
# shallower the fall. So the arrival point is set by how fast the run descends, not by
# where it begins -- which makes lr, weight decay and beta2 the levers that matter.
# weight_decay=0.05 in particular pulls a pretrained backbone toward zero every step.
action_policy_robocasa_edge_sweep["optimizer"]["lr"] = "${oc.decode:${oc.env:SW_LR,0.0001}}"
action_policy_robocasa_edge_sweep["optimizer"]["weight_decay"] = "${oc.decode:${oc.env:SW_WD,0.05}}"
action_policy_robocasa_edge_sweep["optimizer"]["betas"] = [0.9, "${oc.decode:${oc.env:SW_BETA2,0.99}}"]
# What the checkpoint loader is allowed to read. The base arm skips EMA
# (``["net_ema."]``) because the public Edge ships one it should not inherit; the mid
# arm loads everything (``[]``) so it also inherits EgoVerse's EMA. That makes the two
# arms ASYMMETRIC in a way cosmos_policy's pair is not -- there the mid arm changes
# ``checkpoint.load_path`` and nothing else, both arms reading their checkpoint whole.
# Since eval loads ``net_ema`` (``use_ema_weights`` defaults True) and the damage we are
# chasing is in the first few thousand steps, restoring the symmetry is worth one trial:
# SW_SKIP=ema loads EgoVerse's ``net.*`` only and lets EMA start fresh like base's does.
_SKIP = {"ema": ["net_ema."], "none": []}
action_policy_robocasa_edge_sweep["checkpoint"]["keys_to_skip_loading"] = (
    "${oc.decode:${oc.env:SW_SKIP,'[]'}}"
)
# Per-rank batch. The search box is shared: another job took ~49 GB per GPU mid-run and
# every trial started OOMing at 16 samples/rank (~29 GB). Shrinking our own footprint is
# the only correct response -- other users' processes are never ours to kill. Ratios stay
# valid as long as the base reference is measured at the SAME value, which the driver
# enforces by re-measuring base whenever this changes.
action_policy_robocasa_edge_sweep["dataloader_train"]["max_samples_per_batch"] = (
    "${oc.decode:${oc.env:SW_BATCH,16}}"
)
# Where the 7 delta-EE channels sit inside the 64-wide action tensor. EgoVerse trained
# DomainAwareLinear row 3 on head9|Leef9|Lhand15|Reef9|Rhand15, so channels 0..6 -- what a
# 7-D arm reads by default -- are the HEAD block, constant in every robot episode. Offset
# 33 puts our channels on Reef9, the right end-effector delta, which is the same physical
# quantity RoboCasa sends. The loss still covers exactly 7 dims (action_window slices
# [33:40]); only WHICH trained weights get reused changes. 0 = historical layout.
action_policy_robocasa_edge_sweep["dataloader_train"]["dataloader"]["datasets"]["robocasa"][
    "dataset"
]["action_channel_offset"] = "${oc.decode:${oc.env:SW_OFFSET,0}}"
# SW_DUMMYRET: frames in an EMPTY demo slot prepended to every RoboCasa sample, turning
# the single-clip post-train into the paired [demo, query] layout the ICL mid checkpoints
# were trained on. Unset (null) keeps the single-clip layout. See
# datasets/dummy_ret_wrap.py -- this separates "the warm start drifted" from "the warm
# start is being asked for a layout it never saw", which every arm so far has confounded.
action_policy_robocasa_edge_sweep["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"][
    "dummy_ret_frames"
] = "${oc.decode:${oc.env:SW_DUMMYRET,null}}"


# =====================================================================================
# The same sweep, aimed at a SECOND benchmark: Tabletop-Sim ALOHA, 20D bimanual.
#
# Every warm-start conclusion so far rests on one 7-DOF benchmark through one untrained
# head row. If the EgoVerse backbone is worse for RoboCasa specifically, that is a
# recipe problem; if it is worse for a 20D bimanual sim task too, it is a property of
# the mid-training. This arm is the difference between those two readings.
#
# Row 16 plays the part row 5 plays for RoboCasa: ``llm2action.bias[16]`` is identically
# 0 in the public Edge base, the ICL mid, and the BC-replay mid, so all three start from
# the same random head and only the backbone differs.
#
# Everything else -- optimizer, loss weights, warmup, batch, the SW_* knobs -- is
# inherited unchanged, so a number here is directly comparable to the RoboCasa sweep's.
# =====================================================================================
action_policy_aloha_edge_sweep = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_sweep)), flags={"allow_objects": True}
)
action_policy_aloha_edge_sweep["job"]["name"] = "action_policy_aloha_edge_sweep"
_al = action_policy_aloha_edge_sweep["dataloader_train"]["dataloader"]["datasets"]
del _al["robocasa"]
_al["aloha"] = dict(
    ratio=1,
    dataset=L(get_action_aloha_tabletop_sft_dataset)(
        root="${oc.env:ALOHA_ROOT,/mnt/ddn/dataset/tabletop-simulation-hdf5}",
        fps=20.0,
        chunk_length=_CHUNK_LENGTH,
        mode="wam",
        action_normalization="minmax",
        task_split=None,
        pad_tail=True,
        resolution="${oc.env:SW_RES,480}",
        max_action_dim="${model.config.max_action_dim}",
        action_channel_offset="${oc.decode:${oc.env:SW_OFFSET,0}}",
        cfg_dropout_rate=0.1,
        format_prompt_as_json=False,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        iterable_shuffle=True,
        episode_shuffle_seed=42,
        domain_name="${oc.env:SW_DOMAIN,aloha-tabletop}",
        use_state=True,   # RoboCasa/DROID 와 같은 proprio 경로 (action[0] 조건행)
        tokenizer_config="${model.config.vlm_config.tokenizer}",
    ),
)


# =====================================================================================
# Tabletop-Sim ALOHA, 20D bimanual — the full post-train arms (not the 400-step sweep).
#
# Same recipe as the RoboCasa arms, aimed at a second benchmark with a different action
# width and a different head row, so a warm-start conclusion can be read as a property of
# the mid-training rather than of one 7-DOF task.
#
# Row 16 plays the part row 5 plays for RoboCasa: ``llm2action.bias[16]`` is identically 0
# in the public Edge base, in the ICL mid, and in the BC-replay mid, so every arm starts
# post-training from the same random head and only the backbone differs.
#
# Two arms, differing ONLY in load_path -- the same invariant the RoboCasa pair keeps:
#   action_policy_aloha_edge           public Edge base       (EDGE_LOAD_PATH)
#   action_policy_aloha_edge_frommid   an EgoVerse mid ckpt   (EGOVERSE_EDGE_LOAD_PATH)
# The frommid arm sets keys_to_skip_loading=[] so generation AND action heads come from
# the mid checkpoint; load_training_state stays False, so this is a fresh schedule off
# those weights rather than a resume.
# =====================================================================================
action_policy_aloha_edge = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge)), flags={"allow_objects": True}
)
action_policy_aloha_edge["job"]["name"] = "action_policy_aloha_edge"
_al_full = action_policy_aloha_edge["dataloader_train"]["dataloader"]["datasets"]
del _al_full["robocasa"]
_al_full["aloha"] = dict(
    ratio=1,
    dataset=L(get_action_aloha_tabletop_sft_dataset)(
        root="${oc.env:ALOHA_ROOT,/mnt/ddn/dataset/tabletop-simulation-hdf5}",
        fps=20.0,
        chunk_length=_CHUNK_LENGTH,
        mode="wam",
        action_normalization="minmax",
        task_split=None,
        pad_tail=True,
        resolution="480",
        max_action_dim="${model.config.max_action_dim}",
        action_channel_offset=0,
        cfg_dropout_rate=0.1,
        format_prompt_as_json=False,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        iterable_shuffle=True,
        episode_shuffle_seed=42,
        domain_name="aloha-tabletop",
        use_state=True,   # RoboCasa/DROID 와 같은 proprio 경로 (action[0] 조건행)
        tokenizer_config="${model.config.vlm_config.tokenizer}",
    ),
)

action_policy_aloha_edge_frommid = LazyDict(
    copy.deepcopy(dict(action_policy_aloha_edge)), flags={"allow_objects": True}
)
action_policy_aloha_edge_frommid["job"]["name"] = "action_policy_aloha_edge_frommid"
action_policy_aloha_edge_frommid["checkpoint"]["keys_to_skip_loading"] = []
action_policy_aloha_edge_frommid["checkpoint"]["load_path"] = "${oc.env:EGOVERSE_EDGE_LOAD_PATH}"


# =====================================================================================
# ARM33 — the recipe the overnight local search converged on.
#
# Two changes to the mid arm, both measured over 9 x 400-step local trials against a
# base reference run under identical conditions (ratios are trial/base action loss):
#
#   1. domain 3 with the action channels moved to Reef9 (offset 33). EgoVerse trained
#      DomainAwareLinear row 3 on head9|Leef9|Lhand15|Reef9|Rhand15; channels 0..6 --
#      what a 7-D arm reads by default -- are the HEAD block, a CONSTANT in every robot
#      episode. Reef9 is the right end-effector body-frame delta, the same physical
#      quantity RoboCasa sends. The loss still covers exactly 7 dims.
#   2. backbone (moe_gen) lr multiplier 3.0.
#
#   band          arm33+bb3   reef9half   arm_ch9   d3_ref   base
#   2-50             0.1932      0.1915    0.1862   0.4180    1.0
#   51-200           0.7932      0.8139    0.7731   0.9838    1.0
#   201-300          1.0377      1.0706    1.0022   1.0796    1.0
#   301-400          1.1134      1.0521    1.2635   1.0571    1.0
#   mean             0.8594      0.8598    0.8796   0.9553    1.0
#
# Read the late bands honestly: arm33 wins the run and wins it by a real margin (11% on
# the mean over d3_ref, five times the 2.2% noise floor), and its early advantage is
# enormous -- 19% of base's action loss over steps 2-50. But it does NOT end below base,
# and neither does anything else in the table: every trial in the search is above 1.0
# from step ~200 on. Halving the run's remaining goal, "start lower", is met; "end lower"
# is not, by any arm tried so far.
#
# (An earlier version of this block claimed 201-400 = 0.9469 and "below base in all three
# bands". That was read off the trial while it was still running, when the 201-400 band
# held 15 of its eventual 100 points. The numbers above are the completed run.)
#
# reef9half ties arm33 within the noise floor (0.8598 vs 0.8594) at a third of the grad
# norm (40 vs 114). At the 3-node global batch of 384 that is the safer of the two, and
# it is the arm to fall back to if arm33 destabilises late.
#
# What did NOT work, so it is not here: moving the head WEIGHTS to 0..6 instead of the
# data to 33 (1.4025 -- slicing 7 of 57 channels destroys what action2llm feeds the
# backbone); re-splitting the clipped step budget via loss weights (grad norm did not
# move and the descent got worse); and shrinking the Reef9 output weights to cut the
# grad norm (114 -> 40 changed nothing, so grad magnitude was never the mechanism).
# backbone lr 5.0 is clearly worse (0.9578 vs 0.8594). 2.0 vs 3.0 is NOT separated:
# 0.7808 vs 0.7817 at batch 8, well inside the 2.2% noise floor measured above; 3.0 is
# kept because the 400-step batch-16 trials were run there, not because it won.
#
# *** EVAL MUST PASS --domain-name hand_pose AND --action-channel-offset 33 ***
# Either one alone reads a window the run never trained and returns plausible garbage.
# =====================================================================================
action_policy_robocasa_edge_arm33 = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_frommid)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_arm33["job"]["name"] = "action_policy_robocasa_edge_arm33"
action_policy_robocasa_edge_arm33["dataloader_train"]["dataloader"]["datasets"]["robocasa"][
    "dataset"
]["domain_name"] = "hand_pose"
action_policy_robocasa_edge_arm33["dataloader_train"]["dataloader"]["datasets"]["robocasa"][
    "dataset"
]["action_channel_offset"] = 33
action_policy_robocasa_edge_arm33["optimizer"]["lr_multipliers"] = {
    "action2llm": 5.0,
    "llm2action": 5.0,
    "action_modality_embed": 5.0,
    "moe_gen": 3.0,
}
# 3. Skip EMA on load, matching the base arm (the mid arm sets this to [] to warm-start
#    the action heads). This is a NO-OP in both directions, kept only so the two arms
#    differ in nothing that is not deliberate: PowerEMA's first update runs at
#    `ema_beta(0) = 0.0`, i.e. `net_ema = net_ema*0 + net*1`, so whatever was loaded
#    into `net_ema` is completely overwritten by `net` on the very first optimizer
#    step. (Even reading the first call as iteration 1 leaves only 0.41% of it.) It
#    cannot move the training loss either -- `net_ema.requires_grad_(False)` and
#    `ema_scope` restores `net` on exit.
#
#    This matters for reading the search log: `r2_ema_fresh+bbfast3` (0.7647) and its
#    control `r2_backbone_fast3.0` (0.7817) differ ONLY by this flag, so their gap is
#    not an effect -- it is a direct measurement of run-to-run noise on this metric:
#    2.2% on the mean ratio and 6.2% on the late band, at 250 steps / batch 8.
action_policy_robocasa_edge_arm33["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]


# =====================================================================================
# REEF9HALF — arm33's twin, differing only in the checkpoint it warm-starts from.
#
# Same recipe as arm33 (domain 3, action channels at Reef9 via offset 33, moe_gen lr
# 3.0, EMA skipped on load). The load path points at a copy of the EgoVerse mid whose
# llm2action columns for the Reef9 window (33..39) have been scaled by 0.5234 --
# tools/rescale_domain_head_dcp.py, direction untouched, cosine 1.000000. That window
# sits at RMS 0.03909 in mid; domain 5's untrained row, which the base arm starts from,
# sits at 0.01071. 0.5234 lands halfway between the two (0.02046); scaling all the way
# down to domain 5's own value is the reef9full arm, and it was worse (0.9111).
#
# Why run it as well as arm33, measured over 400 local steps at batch 16:
#
#   band          arm33    reef9half
#   2-50         0.1932       0.1915     tied
#   51-200       0.7932       0.8139     tied
#   301-400      1.1134       1.0521     reef9half
#   350-400      1.0609       1.0118     reef9half -- the ONLY arm in the search that
#                                        ends inside the noise floor of base
#   mean         0.8594       0.8598     tied
#   grad max        114           40     reef9half, and this one is not noise-limited
#
# The 350-400 gap (4.9%) is inside the measured 6.2% late-band noise floor, so it is a
# direction, not a proven win. The grad norm is the reason to spend a job on it anyway:
# clip_norm is 1.0, so a step whose pre-clip norm is 114 is being renormalised by 114x
# and only its direction survives. Going from batch 16 to the 3-node global batch of 384
# cuts gradient noise ~5x, which is exactly the regime where a 3x lower pre-clip norm
# stops being cosmetic. 400 local steps cannot see that; 20K at batch 384 can.
#
# *** EVAL MUST PASS --domain-name hand_pose AND --action-channel-offset 33, same as
#     arm33 -- the rescale does not move the window, it only changes its gain. ***
# =====================================================================================
action_policy_robocasa_edge_reef9half = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge_arm33)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_reef9half["job"]["name"] = "action_policy_robocasa_edge_reef9half"


# =====================================================================================
# ROUTING arms — the finding that outlived the warm-start question.
#
# action2llm/llm2action are nn.Embedding tables of per-domain weight matrices, so a run
# only trains the rows it indexes. Measured against the public Edge base, the row each
# benchmark DEFAULTS to is untrained:
#
#   row       a2llm   l2a    bias   상태
#   0  (미사용)  11.3%  11.3%  0.0%   weight decay 만 (기준선)
#   5  (robocasa) 11.3%  11.3%  0.0%   <- RoboCasa 기본. 미학습.
#   16 (aloha)   11.3%  11.3%  0.0%   <- ALOHA 기본. 미학습.
#   8  (r-franka) 16.0%  40.8%  94.2%  학습됨
#   12 (r-dual)   20.7%  43.5%  37.6%  학습됨, 20D
#
# 400-step 프록시 (step>100 paired t, 음수 = base 보다 좋음):
#   RoboCasa  d8_base   t=-2.19    ALOHA  base(row 12)   t=-11.73
#             d8_bcmix  t=-1.00           bcmix8K(row12) t= -7.91
# 두 벤치마크 모두 EgoVerse 를 얹지 않은 공개 base 라우팅이 가장 좋다. ALOHA 는
# 101-200 구간에서 base(row16) 0.0679 -> base(row12) 0.0427 로 37% 낮다.
#
# 아래 두 arm 은 그 라우팅을 20K 실측으로 확인하기 위한 것이다. EgoVerse 는 쓰지 않는다.
# EVAL 은 반드시 해당 domain 이름으로 띄울 것 — 기본값 row 는 이 런들이 학습하지 않는다.
# =====================================================================================

# RoboCasa, 공개 base 를 row 8 로. 지금 도는 dom8(EgoVerse+row8)의 대조군.
# *** EVAL: --domain-name robomind-franka ***
action_policy_robocasa_edge_domain8 = LazyDict(
    copy.deepcopy(dict(action_policy_robocasa_edge)), flags={"allow_objects": True}
)
action_policy_robocasa_edge_domain8["job"]["name"] = "action_policy_robocasa_edge_domain8"
action_policy_robocasa_edge_domain8["dataloader_train"]["dataloader"]["datasets"]["robocasa"][
    "dataset"
]["domain_name"] = "robomind-franka"

# ALOHA, 공개 base 를 row 12 로. 20D -> 20D, dual-arm -> dual-arm 로 차원도 의미도 맞는다.
# *** EVAL: --domain-name robomind-agilex-dual ***
action_policy_aloha_edge_domain12 = LazyDict(
    copy.deepcopy(dict(action_policy_aloha_edge)), flags={"allow_objects": True}
)
action_policy_aloha_edge_domain12["job"]["name"] = "action_policy_aloha_edge_domain12"
action_policy_aloha_edge_domain12["dataloader_train"]["dataloader"]["datasets"]["aloha"][
    "dataset"
]["domain_name"] = "robomind-agilex-dual"


# ALOHA, EgoVerse ICL warm start 를 row 12 로. base+row12 / ICL+row16 과 짝을 이뤄
# "warm start"와 "라우팅"의 기여를 20K 실측으로 분리하기 위한 2x2 의 네 번째 칸.
# *** EVAL: --domain-name robomind-agilex-dual ***
action_policy_aloha_edge_frommid_domain12 = LazyDict(
    copy.deepcopy(dict(action_policy_aloha_edge_frommid)), flags={"allow_objects": True}
)
action_policy_aloha_edge_frommid_domain12["job"]["name"] = (
    "action_policy_aloha_edge_frommid_domain12"
)
action_policy_aloha_edge_frommid_domain12["dataloader_train"]["dataloader"]["datasets"]["aloha"][
    "dataset"
]["domain_name"] = "robomind-agilex-dual"


for _item in [
    action_policy_aloha_edge_frommid_domain12,
    action_policy_robocasa_edge_domain8,
    action_policy_aloha_edge_domain12,
    action_policy_aloha_edge_sweep,
    action_policy_aloha_edge,
    action_policy_aloha_edge_frommid,
    action_policy_robocasa_edge,
    action_policy_robocasa_edge_smoke,
    action_policy_robocasa_edge_frommid,
    action_policy_robocasa_edge_frommid_headreset,
    action_policy_robocasa_edge_frommid_backboneonly,
    action_policy_robocasa_edge_frommid_domain3,
    action_policy_robocasa_edge_frommid_domain8,
    action_policy_robocasa_edge_frommid_domain3_rescaled,
    action_policy_robocasa_edge_frommid_aw,
    action_policy_robocasa_edge_frommid_lr,
    action_policy_robocasa_edge_frommid_wu,
    action_policy_robocasa_edge_frommid_domain3_aw,
    action_policy_robocasa_edge_frommid_domain3s_aw,
    action_policy_robocasa_edge_sweep,
    action_policy_robocasa_edge_arm33,
    action_policy_robocasa_edge_reef9half,
]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)

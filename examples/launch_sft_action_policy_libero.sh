#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for LIBERO 7D action-policy SFT on Cosmos3-Nano (8B MoT).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_libero_repro.toml (selects the
# registered `action_policy_libero_nano` experiment; res256, single agent-view,
# 7D delta-EE [dpos(3), daxang(3), gripper] action, four eval suites mixed,
# trains the generation + action heads).
#
# Env vars (override for your filesystem):
#   DATASET_PATH          LIBERO_LeRobot_v3 parent dir (holds libero_spatial/
#                         libero_object/libero_goal/libero_10). Bridged to
#                         LIBERO_ROOT for the experiment.
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY         for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#
# torchcodec needs FFmpeg libs on LD_LIBRARY_PATH to decode the av1 videos, e.g.:
#   export LD_LIBRARY_PATH=/path/to/ffmpeg/lib:$LD_LIBRARY_PATH
#
# Two-GPU smoke (config/data/checkpoint sanity, 10 iters; Nano-8B needs >=2-way
# FSDP shard, and we drop the batch + EMA to fit memory):
#   NPROC_PER_NODE=2 \
#   TOML_FILE="examples/toml/sft_config/action_policy_libero_smoke.toml" \
#   TAIL_OVERRIDES=(dataloader_train.max_samples_per_batch=1 \
#                   model.config.ema.enabled=false) \
#   bash examples/launch_sft_action_policy_libero.sh
# ============================================================================

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_repro.toml}"
: "${DATASET_PATH:=examples/data/lerobot_v30/LIBERO_LeRobot_v3}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# The experiment reads ${oc.env:LIBERO_ROOT}; bridge the launcher's DATASET_PATH to it.
export LIBERO_ROOT="${LIBERO_ROOT:-$DATASET_PATH}"

EXTRA_DATASET_CHECK='for s in libero_spatial libero_object libero_goal libero_10; do [[ -f "$LIBERO_ROOT/$s/meta/info.json" ]] || { echo "ERROR: missing $LIBERO_ROOT/$s/meta/info.json (download nvidia/LIBERO_LeRobot_v3)" >&2; exit 1; }; done'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

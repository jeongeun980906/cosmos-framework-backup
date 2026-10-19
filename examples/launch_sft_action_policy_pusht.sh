#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for PushT 2D action-policy SFT on Cosmos3-Nano (8B MoT).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_pusht_repro.toml (selects the
# registered `action_policy_pusht_nano` experiment; res256, single view, 2D
# [pos_x, pos_y] action, trains the generation + action heads).
#
# Env vars (override for your filesystem):
#   DATASET_PATH          PushT LeRobot v3.0 split (…/pusht_lerobot/success)
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY         for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#
# Single-GPU smoke (config/data sanity, a few iters):
#   NPROC_PER_NODE=1 \
#   TAIL_OVERRIDES=(trainer.max_iter=10 checkpoint.save_iter=10 \
#                   model.parallelism.data_parallel_shard_degree=1 \
#                   dataloader_train.max_samples_per_batch=8) \
#   bash examples/launch_sft_action_policy_pusht.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_pusht_repro.toml"
: "${DATASET_PATH:=examples/data/lerobot_v30/pusht_lerobot/success}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# The experiment reads ${oc.env:PUSHT_ROOT}; bridge the launcher's DATASET_PATH to it.
export PUSHT_ROOT="${PUSHT_ROOT:-$DATASET_PATH}"

EXTRA_DATASET_CHECK='[[ -f "$PUSHT_ROOT/meta/info.json" ]] || { echo "ERROR: missing $PUSHT_ROOT/meta/info.json (prepare PushT LeRobot v3.0)" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

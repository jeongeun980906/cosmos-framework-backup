#!/bin/bash
# Multi-node WiLoR alignment + video-quality eval for the Cosmos3 EgoVerse v10 3s policy.
# Same two stages as tools/wilor_eval_cosmos3.sh (single-node), sharded across NNODES pods
# exactly like cosmos_policy's tools/wilor_eval_all.sh:
#   stage 1 (every node): decode_dump_egoverse.py on local GPUs 0..N_GPUS-1 as global
#           shards NODE_RANK*N_GPUS .. +N_GPUS-1 of NNODES*N_GPUS -> npz dumps in OUT_DIR.
#   stage 2 (node 0 only, after the barrier): /mnt/ddn/tmp/wilor_metric.py (UNMODIFIED)
#           in a POD-LOCAL wilor_env (pods are HF_HUB_OFFLINE; setup_wilor_env.sh copies
#           the pretrained WiLoR models from the shared env).
# NNODES=1 also works (8 shards + stage-2 on the same node) — used by the 1-node yaml.
#
# env: NNODES (default 1)           number of worker pods (= yaml replicas)
#      POD_NAME (…-worker-R)        node rank source; absent -> rank 0
#      ITER / EVAL_ITER (20000)     checkpoint iteration
#      OUT_DIR (default /mnt/ddn/tmp/wilor_eval_cosmos3/iter<ITER>)
#      N_GPUS (8)  PER_EPISODE (5)  EPISODES (0=all)
#      CKPT_WAIT_MIN (0)            poll this many minutes for the checkpoint dir to land
#                                   (lets the job be applied BEFORE training finishes)
#      RUN_ROOT / EXP / PAIRS / CHUNK / NUM_STEPS / VAE   v10-3s defaults below
#      WV (/workspace/wilor_env)    pod-local wilor env location for stage 2
set -u
REPO=/mnt/ddn/cosmos-framework
cd "$REPO"

ITER="${ITER:-${EVAL_ITER:-20000}}"
OUT_DIR="${OUT_DIR:-/mnt/ddn/tmp/wilor_eval_cosmos3/iter${ITER}}"
NNODES="${NNODES:-1}"
POD_NAME="${POD_NAME:-local-worker-0}"
NODE_RANK=${POD_NAME##*-}
N_GPUS="${N_GPUS:-8}"
NUM_SHARDS=$((NNODES * N_GPUS))
SHARD_OFFSET=$((NODE_RANK * N_GPUS))
PER_EPISODE="${PER_EPISODE:-5}"
EPISODES="${EPISODES:-0}"
CKPT_WAIT_MIN="${CKPT_WAIT_MIN:-0}"
RUN_ROOT="${RUN_ROOT:-/mnt/ddn/tmp/cosmos_egoverse_3s_v10/cosmos3_action/action_sft/action_policy_egoverse_3s_repro}"
EXP="${EXP:-action_policy_egoverse_3s_nano}"
PAIRS="${PAIRS:-/mnt/ddn/dataset/egoverse_pairs_v10_test}"
CHUNK="${CHUNK:-88}"
NUM_STEPS="${NUM_STEPS:-35}"
VAE="${VAE:-/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth}"
WILOR_METRIC="${WILOR_METRIC:-/mnt/ddn/tmp/wilor_metric.py}"
WILOR_CONF="${WILOR_CONF:-0.01}"
WV="${WV:-/workspace/wilor_env}"
VENV="$REPO/.venv"

# ---- wait for the checkpoint (job may be applied before training finishes) ----
CKPT="$RUN_ROOT/checkpoints/iter_$(printf '%09d' "$ITER")"
waited=0
until [ -d "$CKPT" ]; do
  [ "$waited" -ge $((CKPT_WAIT_MIN * 60)) ] && { echo "ERROR: checkpoint not found after ${CKPT_WAIT_MIN}min: $CKPT"; exit 1; }
  echo ">>> waiting for checkpoint $CKPT (${waited}s)"; sleep 60; waited=$((waited + 60))
done
sleep 30   # let the last shards finish writing

BAR="$OUT_DIR/_barrier"
mkdir -p "$OUT_DIR/logs" "$BAR"
echo ">>> wilor_eval_cosmos3 MULTINODE iter=$ITER node=$NODE_RANK shards $SHARD_OFFSET..$((SHARD_OFFSET + N_GPUS - 1)) of $NUM_SHARDS"
echo "    ckpt=$CKPT per_ep=$PER_EPISODE episodes=$EPISODES -> $OUT_DIR  $(date -u +%FT%TZ)"

# ---- stage 1: this node's N_GPUS shards ----
pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
  g=$((SHARD_OFFSET + i))
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HOME="${HF_HOME:-/mnt/ddn/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  PYTHONPATH="$REPO" \
  "$VENV/bin/python" -m cosmos_framework.scripts.decode_dump_egoverse \
    --experiment "$EXP" --run-root "$RUN_ROOT" --iter "$ITER" \
    --pairs-dir "$PAIRS" --chunk-length "$CHUNK" \
    --per-episode "$PER_EPISODE" --episodes "$EPISODES" --num-steps "$NUM_STEPS" \
    --vae-path "$VAE" --shard "$g" --num-shards "$NUM_SHARDS" --out_dir "$OUT_DIR" \
    > "$OUT_DIR/logs/decode_shard_$g.log" 2>&1 &
  pids+=($!)
  echo "  stage1 global shard $g on GPU $i (pid ${pids[-1]})"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
[ "$fail" -gt 0 ] && echo "WARNING: $fail stage-1 shards failed on node $NODE_RANK (see $OUT_DIR/logs)"
touch "$BAR/node_${NODE_RANK}.done"
echo ">>> node $NODE_RANK stage1 done ($(ls "$OUT_DIR"/*.npz 2>/dev/null | wc -l) npz so far)  $(date -u +%FT%TZ)"

# ---- stage 2: node 0 only, after all nodes' stage 1 ----
[ "$NODE_RANK" = "0" ] || { echo ">>> node $NODE_RANK exiting (stage2 runs on node 0)"; exit 0; }
echo ">>> node0 waiting for $NNODES nodes' stage1 barrier ..."
for _ in $(seq 1 720); do
  [ "$(ls "$BAR"/node_*.done 2>/dev/null | wc -l)" -ge "$NNODES" ] && break
  sleep 15
done
NPZ=$(ls "$OUT_DIR"/*.npz 2>/dev/null | wc -l)
echo ">>> all nodes done: $NPZ npz dumps"
[ "$NPZ" -gt 0 ] || { echo "ERROR: no dumps produced"; exit 1; }

echo ">>> setting up pod-local wilor_env at $WV"
WV="$WV" bash /mnt/ddn/cosmos_policy/tools/setup_wilor_env.sh
echo ">>> stage2 WiLoR + image metrics (conf=$WILOR_CONF)"
CUDA_VISIBLE_DEVICES=0 HF_HOME="${HF_HOME:-/mnt/ddn/hf_cache}" \
  "$WV/bin/python" "$WILOR_METRIC" "$OUT_DIR" "$OUT_DIR/wilor_metrics.json" "$WILOR_CONF" \
  2>&1 | tee "$OUT_DIR/logs/wilor_stage2.log"

echo "=== DONE -> $OUT_DIR/wilor_metrics.json ==="
python3 -c "import json;print(json.dumps(json.load(open('$OUT_DIR/wilor_metrics.json'))['summary'], indent=2))"

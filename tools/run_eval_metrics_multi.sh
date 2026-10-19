#!/bin/bash
# Multi-GPU / multi-node sharded shape-metric eval for Cosmos3 (tools/eval_egoverse_metrics.py),
# then merge. Each shard = one plain-python process on one GPU (no torchrun/uv — the eval loads
# the DCP checkpoint DIRECTLY with parallelism shard=1/replicate=1). Mirrors cosmos_policy's
# run_batch_8gpu.sh: node R runs local GPUs 0..N-1 as GLOBAL shards SHARD_OFFSET..+N-1 of
# NUM_SHARDS; node 0 (DO_MERGE=1) waits for all shard CSVs then merges.
#
# env:  CKPT=<iter_<N> DCP dir>  N_GPUS=8  OUT=<dir>  EXP=<experiment>  CHUNK=148
#       PAIRS=<v9 test dir>  PER_EPISODE=0(all windows)  TOPK=1  EPISODES=0  NUM_STEPS=35
#       SHARD_OFFSET=0  NUM_SHARDS=N_GPUS  DO_MERGE=1  VAE=<wan vae>
set -u
REPO=/mnt/ddn/cosmos-framework
cd "$REPO"

CKPT="${CKPT:?set CKPT to the iter_<N> DCP checkpoint dir}"
N_GPUS="${N_GPUS:-8}"
OUT="${OUT:-/mnt/ddn/tmp/cosmos3_eval}"
EXP="${EXP:-action_policy_egoverse_5s_nano}"
CHUNK="${CHUNK:-148}"
PAIRS="${PAIRS:-/mnt/ddn/dataset/egoverse_pairs_v9_test}"
NORM="${NORM:-cosmos_framework/data/vfm/action/datasets/egoverse_normalization.json}"  # egodex: egodex_c3_normalization.json
PER_EPISODE="${PER_EPISODE:-0}"     # 0 = ALL ~1s windows
TOPK="${TOPK:-1}"                   # same-top-k + cross-top-k demos
EPISODES="${EPISODES:-0}"
NUM_STEPS="${NUM_STEPS:-35}"
VAE="${VAE:-/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
NUM_SHARDS="${NUM_SHARDS:-$N_GPUS}"
DO_MERGE="${DO_MERGE:-1}"
VENV="$REPO/.venv"
LOG="$OUT/shard_logs"
mkdir -p "$OUT" "$LOG"

echo ">>> cosmos3 eval: local GPUs 0..$((N_GPUS-1)) → global shards $SHARD_OFFSET..$((SHARD_OFFSET+N_GPUS-1)) of $NUM_SHARDS"
echo "    ckpt=$CKPT  exp=$EXP  pairs=$PAIRS  per_ep=$PER_EPISODE topk=$TOPK steps=$NUM_STEPS  out=$OUT"

pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
  g=$((SHARD_OFFSET + i))
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HOME="${HF_HOME:-/mnt/ddn/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  PYTHONPATH="$REPO" \
  "$VENV/bin/python" tools/eval_egoverse_metrics.py \
    --experiment "$EXP" --checkpoint "$CKPT" --chunk-length "$CHUNK" \
    --pairs-dir "$PAIRS" --norm "$NORM" --per-episode "$PER_EPISODE" --topk "$TOPK" --episodes "$EPISODES" \
    --num-steps "$NUM_STEPS" --vae-path "$VAE" \
    --shard "$g" --num-shards "$NUM_SHARDS" --out-dir "$OUT" \
    > "$LOG/shard_$g.log" 2>&1 &
  pids+=($!)
  echo "  launched global shard $g (local GPU $i, pid ${pids[-1]}) -> $LOG/shard_$g.log"
done

echo ">>> $N_GPUS local shards launched; waiting..."
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "  a local shard FAILED (see $LOG)"; fail=$((fail + 1)); fi
done
[ "$fail" -gt 0 ] && echo "WARNING: $fail local shards failed"

if [ "$DO_MERGE" != "1" ]; then
  echo ">>> DO_MERGE=0 — local shards done, exiting."
  exit 0
fi

echo ">>> merge node: waiting for all $NUM_SHARDS shard CSVs ..."
for _t in $(seq 1 240); do
  have=$(ls "$OUT"/eval_metrics_shard*of"$NUM_SHARDS".csv 2>/dev/null | wc -l)
  [ "$have" -ge "$NUM_SHARDS" ] && break
  sleep 30
done
echo ">>> merging $have/$NUM_SHARDS shard CSVs..."
PYTHONPATH="$REPO" "$VENV/bin/python" tools/eval_egoverse_metrics.py \
  --merge --num-shards "$NUM_SHARDS" --out-dir "$OUT"
echo "======================================================================"
echo " DONE. eval_metrics.csv + eval_metrics_summary.json in $OUT"
echo "======================================================================"

#!/bin/bash
# WiLoR-based alignment + video-quality eval for the Cosmos3 EgoVerse v10 3s policy,
# directly comparable to cosmos_policy's (2.5) tools/wilor_eval_all.sh numbers:
#   stage 1 (this repo's .venv, GPUs 0..N_GPUS-1, sharded):
#       cosmos_framework/scripts/decode_dump_egoverse.py -> per-sample npz in the EXACT
#       2.5 decode_dump contract (jpg-encoded gen/gt frames + FK head-frame wrists).
#   stage 2 (wilor_env, GPU 0, REUSED VERBATIM):
#       /mnt/ddn/tmp/wilor_metric.py <dump> <json> 0.01
#       -> SSIM/PSNR/LPIPS/FID + pose_action_corr summary.
#
# env: ITER (default 20000)        checkpoint iteration of the v10 3s run
#      OUT_DIR (default /mnt/ddn/tmp/wilor_eval_cosmos3/iter<ITER>)
#      N_GPUS (default 2)          local GPUs / stage-1 shards (CUDA_VISIBLE_DEVICES=0,1)
#      PER_EPISODE (default 5)     query windows per episode (2.5 used PER_EPISODE=5)
#      EPISODES (default 0)        cap total windows (0 = none); use small for smoke
#      RUN_ROOT / EXP / PAIRS / CHUNK / NUM_STEPS  override the v10-3s defaults
#      SKIP_STAGE2=1               dump only
set -u
REPO=/mnt/ddn/cosmos-framework
cd "$REPO"

ITER="${ITER:-20000}"
OUT_DIR="${OUT_DIR:-/mnt/ddn/tmp/wilor_eval_cosmos3/iter${ITER}}"
N_GPUS="${N_GPUS:-2}"
PER_EPISODE="${PER_EPISODE:-5}"
EPISODES="${EPISODES:-0}"
RUN_ROOT="${RUN_ROOT:-/mnt/ddn/tmp/cosmos_egoverse_3s_v10/cosmos3_action/action_sft/action_policy_egoverse_3s_repro}"
EXP="${EXP:-action_policy_egoverse_3s_nano}"
PAIRS="${PAIRS:-/mnt/ddn/dataset/egoverse_pairs_v10_test}"
CHUNK="${CHUNK:-88}"
NUM_STEPS="${NUM_STEPS:-35}"
VAE="${VAE:-/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth}"
WILOR_PY="${WILOR_PY:-/mnt/ddn/ckpt/wilor_env/bin/python}"
WILOR_METRIC="${WILOR_METRIC:-/mnt/ddn/tmp/wilor_metric.py}"
WILOR_CONF="${WILOR_CONF:-0.01}"
VENV="$REPO/.venv"

CKPT="$RUN_ROOT/checkpoints/iter_$(printf '%09d' "$ITER")"
[ -d "$CKPT" ] || { echo "ERROR: checkpoint dir not found: $CKPT"; exit 1; }
mkdir -p "$OUT_DIR/logs"
echo ">>> wilor_eval_cosmos3 iter=$ITER  ckpt=$CKPT"
echo "    exp=$EXP pairs=$PAIRS chunk=$CHUNK per_ep=$PER_EPISODE episodes=$EPISODES -> $OUT_DIR"

# ---- stage 1: sharded decode dump on local GPUs ----
pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HOME="${HF_HOME:-/mnt/ddn/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  PYTHONPATH="$REPO" \
  "$VENV/bin/python" -m cosmos_framework.scripts.decode_dump_egoverse \
    --experiment "$EXP" --run-root "$RUN_ROOT" --iter "$ITER" \
    --pairs-dir "$PAIRS" --chunk-length "$CHUNK" \
    --per-episode "$PER_EPISODE" --episodes "$EPISODES" --num-steps "$NUM_STEPS" \
    --vae-path "$VAE" --shard "$i" --num-shards "$N_GPUS" --out_dir "$OUT_DIR" \
    > "$OUT_DIR/logs/decode_shard_$i.log" 2>&1 &
  pids+=($!)
  echo "  stage1 shard $i on GPU $i (pid ${pids[-1]}) -> $OUT_DIR/logs/decode_shard_$i.log"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
[ "$fail" -gt 0 ] && echo "WARNING: $fail stage-1 shards failed (see $OUT_DIR/logs)"
NPZ=$(ls "$OUT_DIR"/*.npz 2>/dev/null | wc -l)
echo ">>> stage1 done: $NPZ npz dumps in $OUT_DIR"
[ "$NPZ" -gt 0 ] || { echo "ERROR: no dumps produced"; exit 1; }

# ---- stage 2: WiLoR + image metrics (existing script, UNMODIFIED) ----
[ "${SKIP_STAGE2:-0}" = "1" ] && { echo ">>> SKIP_STAGE2=1 — dump only."; exit 0; }
echo ">>> stage2 WiLoR metrics ($WILOR_PY $WILOR_METRIC conf=$WILOR_CONF)"
CUDA_VISIBLE_DEVICES=0 HF_HOME="${HF_HOME:-/mnt/ddn/hf_cache}" \
  "$WILOR_PY" "$WILOR_METRIC" "$OUT_DIR" "$OUT_DIR/wilor_metrics.json" "$WILOR_CONF" \
  2>&1 | tee "$OUT_DIR/logs/wilor_stage2.log"

echo "=== DONE -> $OUT_DIR/wilor_metrics.json ==="
python3 -c "import json;print(json.dumps(json.load(open('$OUT_DIR/wilor_metrics.json'))['summary'], indent=2))"

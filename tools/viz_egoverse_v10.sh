#!/bin/bash
# One-command viz rerun for the EgoVerse v10 3s policy: renders N sample videos
# (demo | GT | gen side-by-side + skeleton + combined report) at a given checkpoint
# iteration — same style/outputs as the earlier /mnt/ddn/tmp/cosmos3_viz_iter10000
# (which wrapped tools/eval_egoverse_generate.py, see /mnt/ddn/tmp/run_c3_viz.sh).
#
# env: ITER (default 20000)   OUT_DIR (default /mnt/ddn/tmp/cosmos3_viz_v10_iter<ITER>)
#      N (default 6 samples)  GPU (default 0)
#      RUN_ROOT / EXP / PAIRS / CHUNK override the v10-3s defaults
#      EXTRA="..." extra eval_egoverse_generate.py args (e.g. --cross-emb --query-emb aria)
set -u
REPO=/mnt/ddn/cosmos-framework
cd "$REPO"
export VIRTUAL_ENV="$REPO/.venv" PATH="$REPO/.venv/bin:$PATH"
unset PYTHONHOME
export HF_HOME="${HF_HOME:-/mnt/ddn/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ITER="${ITER:-20000}"
OUT_DIR="${OUT_DIR:-/mnt/ddn/tmp/cosmos3_viz_v10_iter${ITER}}"
N="${N:-6}"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-/mnt/ddn/tmp/cosmos_egoverse_3s_v10/cosmos3_action/action_sft/action_policy_egoverse_3s_repro}"
EXP="${EXP:-action_policy_egoverse_3s_nano}"
PAIRS="${PAIRS:-/mnt/ddn/dataset/egoverse_pairs_v10_test}"
CHUNK="${CHUNK:-88}"

CKPT="$RUN_ROOT/checkpoints/iter_$(printf '%09d' "$ITER")"
[ -d "$CKPT" ] || { echo "ERROR: checkpoint dir not found: $CKPT"; exit 1; }
echo ">>> viz v10 3s iter=$ITER ($N samples) -> $OUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$REPO" \
python tools/eval_egoverse_generate.py \
  --experiment "$EXP" --checkpoint "$CKPT" \
  --chunk-length "$CHUNK" --num-samples "$N" --num-steps 35 \
  --pairs-dir "$PAIRS" \
  --vae-path /mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth \
  --out-dir "$OUT_DIR" ${EXTRA:-}
echo "=== C3 v10 VIZ DONE -> $OUT_DIR ==="

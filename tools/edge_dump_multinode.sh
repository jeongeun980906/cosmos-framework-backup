#!/bin/bash
# cosmos3 EDGE 덤프 전용 드라이버 (wilor_eval_cosmos3_multinode.sh 의 stage-1 만).
# WiLoR/FID stage-2 는 하지 않는다 — 행동 오차(msen/l1n/acc@tau, DTW)만 필요하기 때문.
#
# 산출: $OUT_DIR/s<shard>_<k>.npz  (gen_a/gt_a 89x57, fkL/fkR, gen/gt JPEG, idx, emb)
#       -> 182_norm_acc_posthoc.py / 173_abc_dump_metrics.py 가 그대로 읽는다.
#
# env:
#   NNODES(1) N_GPUS(8) POD_NAME(...-worker-R)   샤딩
#   ITER(8000)                                   post-train 체크포인트 iteration
#   CKPT                                         명시 DCP 경로 (주면 RUN_ROOT/ITER 무시)
#                                                -> base 비교용: /mnt/ddn/ckpt/cosmos3_edge_dcp/model
#   RUN_ROOT EXP PAIRS DATA_ROOT CHUNK NUM_STEPS GUIDANCE VAE
#   PER_EPISODE(4) EPISODES(0=all)
#   DEMO_MODE(random|A|B|C)                      A/B/C 셀 (decode_dump_egoverse.py 에 추가됨)
#   ACTION_SOURCE(state) USE_STATE(1)            데모 조건이 액션 헤드로 새지 않게 반드시 state
set -u
REPO=/mnt/ddn/cosmos-framework
cd "$REPO"
VENV="$REPO/.venv"

NNODES="${NNODES:-1}"
N_GPUS="${N_GPUS:-8}"
POD_NAME="${POD_NAME:-x-worker-0}"
NODE_RANK="${POD_NAME##*-}"
case "$NODE_RANK" in ''|*[!0-9]*) NODE_RANK=0 ;; esac
NUM_SHARDS=$((NNODES * N_GPUS))
SHARD_OFFSET=$((NODE_RANK * N_GPUS))

ITER="${ITER:-8000}"
EXP="${EXP:-action_policy_egoverse_v10_icl_edge}"
RUN_ROOT="${RUN_ROOT:-/mnt/ddn/tmp/egoverse_v10_icl_edge_4node/cosmos3_action/action_sft/action_policy_egoverse_v10_icl_edge}"
PAIRS="${PAIRS:-/mnt/ddn/dataset/egoverse_pairs_v10_test}"
DATA_ROOT="${DATA_ROOT:-/mnt/ddn/dataset}"
# he/egodex 는 기본 egoverse_normalization.json 에 블록이 없어 KeyError 가 난다.
# 각자 파일을 NORM 으로 넘긴다 (he_c3_normalization.json / egodex_c3_normalization.json).
NORM="${NORM:-cosmos_framework/data/generator/action/datasets/egoverse_normalization.json}"
CHUNK="${CHUNK:-88}"
NUM_STEPS="${NUM_STEPS:-35}"
GUIDANCE="${GUIDANCE:-1.5}"
VAE="${VAE:-/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth}"
PER_EPISODE="${PER_EPISODE:-4}"
EPISODES="${EPISODES:-0}"
DEMO_MODE="${DEMO_MODE:-random}"
# SUBSET: 183_fixed_eval_subset.py 가 만든 eval_fixed200.json.
# 주면 쿼리+데모 쌍이 파일로 고정되고 PER_EPISODE/EPISODES/DEMO_MODE 는 무시된다.
# HE/EgoDex 는 **반드시** 이걸 쓴다 (안 쓰면 모델마다 다른 윈도우를 평가하게 된다).
SUBSET="${SUBSET:-}"
ACTION_SOURCE="${ACTION_SOURCE:-state}"
CKPT="${CKPT:-}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"

mkdir -p "$OUT_DIR/logs"
if [ -n "$CKPT" ]; then
  CK_ARGS=(--checkpoint "$CKPT")
  echo ">>> explicit checkpoint: $CKPT"
else
  CK_ARGS=(--run-root "$RUN_ROOT" --iter "$ITER")
  D="$RUN_ROOT/checkpoints/iter_$(printf '%09d' "$ITER")"
  [ -d "$D" ] || { echo "ERROR: checkpoint dir not found: $D" >&2; exit 1; }
fi
[ -f "$PAIRS/windows.npz" ] || { echo "ERROR: $PAIRS/windows.npz 없음" >&2; exit 1; }
[ -f "$PAIRS/pairs.npz" ]   || { echo "ERROR: $PAIRS/pairs.npz 없음" >&2; exit 1; }

echo ">>> edge dump  exp=$EXP  demo=$DEMO_MODE  pairs=$PAIRS  data_root=$DATA_ROOT  subset=${SUBSET:-none}"
echo "    shards ${SHARD_OFFSET}..$((SHARD_OFFSET + N_GPUS - 1))/${NUM_SHARDS}  "\
"per_ep=$PER_EPISODE episodes=$EPISODES steps=$NUM_STEPS guidance=$GUIDANCE  out=$OUT_DIR"

pids=()
for g in $(seq 0 $((N_GPUS - 1))); do
  SH=$((SHARD_OFFSET + g))
  CUDA_VISIBLE_DEVICES="$g" \
  "$VENV/bin/python" -m cosmos_framework.scripts.decode_dump_egoverse \
    --experiment "$EXP" "${CK_ARGS[@]}" \
    --pairs-dir "$PAIRS" --data-root "$DATA_ROOT" --norm "$NORM" --chunk-length "$CHUNK" \
    --per-episode "$PER_EPISODE" --episodes "$EPISODES" \
    --num-steps "$NUM_STEPS" --guidance "$GUIDANCE" \
    --action-source "$ACTION_SOURCE" --use-state \
    --demo-mode "$DEMO_MODE" ${SUBSET:+--subset "$SUBSET"} \
    --vae-path "$VAE" --shard "$SH" --num-shards "$NUM_SHARDS" --out_dir "$OUT_DIR" \
    > "$OUT_DIR/logs/decode_shard_$SH.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo ">>> dumps: $(ls "$OUT_DIR"/*.npz 2>/dev/null | wc -l)  (rc=$rc)"
[ "$rc" = 0 ] && echo EDGE_DUMP_DONE || echo "EDGE_DUMP_FAILED — $OUT_DIR/logs/ 확인"
exit $rc

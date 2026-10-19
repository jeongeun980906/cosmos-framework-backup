#!/bin/bash
# ONLINE (simulation) eval of the RoboCasa all-24-task Cosmos3-Edge policy, on one
# 8-GPU node. A flock'd queue of (task, seed) atoms feeds 8 workers until it drains,
# then a per-task success-rate summary is written to $OUT/summary.txt — same shape as
# the cosmos_policy rc24 eval yamls, so the two land in the same table.
#
# Unlike the cosmos_policy evals (which load the model in-process), Cosmos3 needs TWO
# venvs, because robosuite/mujoco pins conflict with the training stack:
#
#   GPU g:  [policy server]  cosmos-framework .venv, port 800(g+1)
#              ^ HTTP
#           [eval client]    robocasa venv (CLIENT_PY), mujoco EGL on the same GPU
#
# No DCP -> .pt conversion step: the framework server loads a training-time DCP
# directly via --config-file.
#
# Env knobs (override on launch):
#   ITER=20000  TRIALS=50  SEEDS="195"  NUM_STEPS=15  OPEN_LOOP=16
#   RUN_ROOT=/mnt/ddn/tmp/robocasa24_c3edge_base   EXP_NAME=action_policy_robocasa_edge
#     (mid arm: RUN_ROOT=..._mid  EXP_NAME=action_policy_robocasa_edge_frommid)
#   CLIENT_PY=/mnt/ddn/cosmos_policy/.venv/bin/python
#   CKPT_WAIT_MAX=93600   # 26h — apply this job before training finishes and it waits
#   EVAL_GPUS="0 1"       # default "0 1 2 3 4 5 6 7"; fewer GPUs = same protocol, longer
#
# NOTE on --resolution 480: the 3-cam stack is 672x224, whose min side is 224, so the
# server's shape-derived tier would be "256" — a silent train/eval canvas mismatch
# (training snaps to 832x480). Do not drop the flag.
set -uo pipefail

REPO=/mnt/ddn/cosmos-framework
cd "$REPO"
export VIRTUAL_ENV="$REPO/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME

ITER=${ITER:-20000}
TRIALS=${TRIALS:-50}          # paper protocol: 50 trials/task
SEEDS=${SEEDS:-195}           # cosmos_policy arms are all seed 195 alone -> keep comparable
NUM_STEPS=${NUM_STEPS:-15}    # denoising steps (15 ~2x faster than 30, ~same quality)
OPEN_LOOP=${OPEN_LOOP:-16}    # execute 16 of the 32 predicted steps before re-querying
RUN_ROOT=${RUN_ROOT:-/mnt/ddn/tmp/robocasa24_c3edge_base}
# Which arm's run dir to read. base: action_policy_robocasa_edge
#                              mid : action_policy_robocasa_edge_frommid
EXP_NAME=${EXP_NAME:-action_policy_robocasa_edge}
# Which DomainAwareLinear row the action heads index. MUST match what the checkpoint
# was TRAINED with: action2llm/llm2action are per-domain weight tables, so querying a
# row the run never trained returns garbage rather than an error. Default "robocasa"
# -> domain 5. The dom3 arm trains on "hand_pose" (domain 3) and must be evaluated
# with DOMAIN_NAME=hand_pose.
DOMAIN_NAME=${DOMAIN_NAME:-robocasa}
# Where the 7 action channels sit inside the 64-wide tensor. MUST match the value the
# checkpoint was TRAINED with (dataset action_channel_offset). 33 = Reef9, the EgoVerse
# right end-effector block, used by the arm33 arm. Reading the wrong window returns
# plausible-looking garbage rather than an error.
ACTION_CHANNEL_OFFSET=${ACTION_CHANNEL_OFFSET:-0}
CLIENT_PY=${CLIENT_PY:-/mnt/ddn/cosmos_policy/.venv/bin/python}
# Which GPUs get a (server, worker) pair. Default = the 8-GPU pod this script was
# written for; the task queue is flock'd and atomic, so any count drains it correctly
# and only the wall clock changes (24 tasks x 50 trials is ~11 h on 8 GPUs). Set e.g.
# EVAL_GPUS="0 1" to run the same protocol on a 2-GPU box.
EVAL_GPUS=${EVAL_GPUS:-"0 1 2 3 4 5 6 7"}
_NGPU=$(echo $EVAL_GPUS | wc -w)
CKPT_WAIT_MAX=${CKPT_WAIT_MAX:-$((26 * 3600))}
NOTE=${NOTE:-${EXP_NAME#action_policy_robocasa_}-ckpt$((ITER / 1000))k}

CKPT_DIR=$RUN_ROOT/cosmos3_action/action_sft/$EXP_NAME
ITER_PAD=$(printf "iter_%09d" "$ITER")
CKPT=$CKPT_DIR/checkpoints/$ITER_PAD
NORM=$REPO/cosmos_framework/data/generator/action/datasets/robocasa_hdf5_normalization.json
OUT=${OUT:-$RUN_ROOT/eval_${ITER}_full}
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS"
exec > >(tee -a "$LOGS/driver.log") 2>&1
echo ">>> c3edge rc24 eval start $(date -u +%FT%TZ)  CKPT=$CKPT TRIALS=$TRIALS SEEDS=$SEEDS"

export PYTHONPATH=$REPO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export HF_HOME=${HF_HOME:-/mnt/ddn/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
# `uvx` is required by the Edge VLM processor fetch even when HF_HUB_OFFLINE resolves
# it from the shared cache (CheckpointDirHf shells out to `uvx hf download`).
export PATH="$HOME/.local/bin:$PATH"
command -v uvx >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# The cosmos_policy .venv's interpreter is a symlink into the ephemeral /workspace uv
# dir, so it must be recreated in a fresh pod before the client can run (same trick as
# the cosmos-robocasa24-*-eval-8gpu.yaml jobs).
uv python install 3.10 >/dev/null 2>&1 || true
[ -x "$CLIENT_PY" ] || { echo "ERROR: CLIENT_PY=$CLIENT_PY not executable (need a robocasa venv)" >&2; exit 1; }
"$CLIENT_PY" -c "import robosuite, robocasa; print('client venv OK')" || exit 1
[ -f "$NORM" ] || { echo "ERROR: missing normalization stats $NORM" >&2; exit 1; }

# ---- wait for the checkpoint (apply before training ends and this just blocks) ----
_t0=$(date +%s)
while [ ! -d "$CKPT" ]; do
    _el=$(( $(date +%s) - _t0 ))
    [ $_el -gt $CKPT_WAIT_MAX ] && { echo "ERROR: ${CKPT_WAIT_MAX}s 안에 $ITER 체크포인트가 나타나지 않음" >&2; exit 1; }
    [ $(( _el % 1800 )) -lt 300 ] && echo ">>> ${ITER} 대기중 ($((_el/60))분 경과)"
    sleep 300
done
# A DCP dir appears before it is fully written; wait for .metadata plus a settled size.
_p=0
while true; do
    _c=$(du -sb "$CKPT" 2>/dev/null | cut -f1 || echo 0)
    [ -f "$CKPT/model/.metadata" ] && [ "$_c" -gt 0 ] && [ "$_c" = "$_p" ] && break
    _p=$_c; sleep 20
done
echo ">>> checkpoint ready: $CKPT ($(du -sh "$CKPT" | cut -f1))  $(date -u +%FT%TZ)"

# ---- 8 policy servers, one per GPU ----
echo ">>> launching $_NGPU servers on GPUs [$EVAL_GPUS] (iter=$ITER, num_steps=$NUM_STEPS)..."
for g in $EVAL_GPUS; do
    PORT=$((8001 + g))
    CUDA_VISIBLE_DEVICES=$g python -u -m cosmos_framework.scripts.action_policy_server_libero \
        --checkpoint-path "$CKPT" --config-file "$CKPT_DIR/config.yaml" \
        --action-stats-path "$NORM" --action-stats-key robocasa --action-normalization minmax \
        --action-channel-offset "$ACTION_CHANNEL_OFFSET" \
        --use-state --state-stats-key robocasa_state \
        --resolution 480 --raw-action-dim 7 --action-chunk-size 32 --fps 20 \
        --append-duration-fps False --append-resolution-info False \
        --num-steps "$NUM_STEPS" --port "$PORT" > "$LOGS/server_gpu${g}.log" 2>&1 &
done
for g in $EVAL_GPUS; do
    PORT=$((8001 + g))
    for _ in $(seq 1 60); do
        curl -s -m 3 "http://127.0.0.1:${PORT}/info" >/dev/null 2>&1 && break
        sleep 10
    done
    curl -s -m 4 "http://127.0.0.1:${PORT}/info" >/dev/null 2>&1 \
        && echo "    gpu$g port$PORT READY" \
        || { echo "    gpu$g port$PORT FAILED"; tail -5 "$LOGS/server_gpu${g}.log"; }
done

# ---- 24 atomic tasks; the long ones (PnP, OpenDoubleDoor) first so the tail is short ----
QF=$LOGS/queue.txt
cat > "$QF" << 'Q'
OpenDoubleDoor
PnPCounterToSink
PnPCounterToMicrowave
PnPCounterToCab
PnPCabToCounter
PnPSinkToCounter
PnPStoveToCounter
PnPCounterToStove
PnPMicrowaveToCounter
CloseDoubleDoor
CoffeeSetupMug
CoffeeServeMug
OpenSingleDoor
CloseSingleDoor
OpenDrawer
CloseDrawer
TurnOnStove
TurnOffStove
TurnOnSinkFaucet
TurnOffSinkFaucet
TurnSinkSpout
TurnOnMicrowave
TurnOffMicrowave
CoffeePressButton
Q
sed -i 's/^ *//' "$QF"
# Expand into (task, seed) atoms. With SEEDS="195" that is 24 atoms; adding seeds scales it.
awk -v seeds="$SEEDS" 'NF {n=split(seeds,a," "); for(i=1;i<=n;i++) print $1, a[i]}' "$QF" > "$QF.tmp"
mv "$QF.tmp" "$QF"

next_task() {  # atomic pop of the first queue line
    (
      flock 9
      line=$(head -n1 "$QF" 2>/dev/null)
      [ -n "$line" ] && sed -i '1d' "$QF"
      echo "$line"
    ) 9>"$QF.lock"
}

worker() {
    local g=$1 line task seed port
    port=$((8001 + g))
    while true; do
        line=$(next_task)
        [ -z "$line" ] && break
        task=${line%% *}; seed=${line##* }
        echo "[gpu$g] START $task seed=$seed $(date -u +%T)"
        # EGL/GPU selection: robosuite constrains this in two places —
        #  (1) binding_utils.py:33  assert MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES (substring)
        #  (2) egl_context.py:40    unset MEDI makes CVD the EGL index, which must satisfy
        #      0 <= idx < len(eglQueryDevicesEXT()); in-pod that length is 1.
        # So set CVD="$g,0" with MEDI=0: the assert passes, EGL index 0 is always valid.
        # (g=0 -> CVD="0".)
        _cvd=$g; [ "$g" != "0" ] && _cvd="$g,0"
        CUDA_VISIBLE_DEVICES=$_cvd MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYTHONPATH="$REPO" \
        "$CLIENT_PY" -u -m cosmos_framework.scripts.eval_robocasa_client \
            --server-url "http://127.0.0.1:${port}" \
            --tasks "$task" --num-trials-per-task "$TRIALS" \
            --num-open-loop-steps "$OPEN_LOOP" --seed "$seed" \
            --domain-name "$DOMAIN_NAME" \
            --out-dir "$LOGS/out_${task}_seed${seed}" \
            > "$LOGS/task_${task}_seed${seed}.log" 2>&1
        echo "[gpu$g] DONE $task seed=$seed exit=$? $(date -u +%T)  $(grep -h 'OVERALL' "$LOGS/task_${task}_seed${seed}.log" | tail -1)"
    done
}

# Collect the WORKER pids and wait on those only. A bare `wait` also waits on the
# server jobs started with `&` above — they serve until killed, so it never returns
# and the summary below is never written even though every task finished.
worker_pids=()
for g in $EVAL_GPUS; do worker $g & worker_pids+=($!); done
wait "${worker_pids[@]}"
echo ">>> all workers done; shutting down servers"
pkill -f "cosmos_framework.scripts.action_policy_server_libero" 2>/dev/null

S=$OUT/summary.txt
{
    echo "==== robocasa all-24 online eval: $NOTE ($(date -u +%FT%TZ)) ===="
    echo "ckpt: $CKPT"
    echo "trials/task: $TRIALS  seed: $SEEDS  open_loop: $OPEN_LOOP  denoise_steps: $NUM_STEPS"
    echo "stack: cosmos3-edge (server+client), resolution 480, use_state, 3-cam concat"
    echo "domain: $DOMAIN_NAME   action_channel_offset: $ACTION_CHANNEL_OFFSET"
    for f in "$LOGS"/out_*/report.json; do
        [ -f "$f" ] || continue
        "$CLIENT_PY" - "$f" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for task, r in d.get("per_task", {}).items():
    print(f"{task:<34} {r['success_rate']:.4f}  ({r['successes']}/{r['episodes']})")
PY
    done
    echo
    echo "--- per-task mean over seeds ${SEEDS} ---"
    for f in "$LOGS"/out_*/report.json; do
        [ -f "$f" ] || continue
        "$CLIENT_PY" - "$f" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for task, r in d.get("per_task", {}).items():
    print(task, r["success_rate"])
PY
    done | awk 'NF==2 {s[$1]+=$2; n[$1]++} END {t=0; c=0; for (k in s) {m=s[k]/n[k]; printf "%-24s %.4f (n=%d)\n", k, m, n[k]; t+=m; c++} printf "\nMEAN over %d tasks: %.4f\n", c, t/c}' | sort
} | tee "$S"
echo ">>> c3edge rc24 eval COMPLETE $(date -u +%FT%TZ)  summary: $S"

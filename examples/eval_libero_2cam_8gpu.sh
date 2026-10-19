#!/bin/bash
# 8-GPU parallel LIBERO eval for the lerobot/libero 2-cam + use_state policy.
# Launches 8 single-GPU policy servers (ports 8001..8008) + 8 eval clients, each
# client owning a disjoint (suite, task-slice). Aggregates the per-GPU JSON
# reports into one per-suite + overall summary. ~8x faster than a single GPU.
#
# Env knobs (override on launch):
#   ITER=8000  TRIALS=50  NUM_STEPS=15
#   CKPT_DIR=/mnt/ddn/tmp/cosmos_libero/cosmos3_action/action_sft/action_policy_libero_repro
set -uo pipefail

REPO=/mnt/ddn/cosmos-framework
cd "$REPO"
# Activate the relocated venv the same way the training pod does (works in-container).
export VIRTUAL_ENV="$REPO/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME

ITER=${ITER:-8000}
TRIALS=${TRIALS:-50}
NUM_STEPS=${NUM_STEPS:-15}
CKPT_DIR=${CKPT_DIR:-/mnt/ddn/tmp/cosmos_libero/cosmos3_action/action_sft/action_policy_libero_repro}
NORM=cosmos_framework/data/vfm/action/datasets/lerobot_libero_normalization.json
ITER_PAD=$(printf "iter_%09d" "$ITER")
OUT=/mnt/ddn/outputs/libero_2cam_full_${ITER}
mkdir -p "$OUT"

# In-pod: ffmpeg (torchcodec) + libegl (mujoco) come from apt — no conda
# LD_LIBRARY_PATH hack. For local 2-GPU testing, export LD_LIBRARY_PATH yourself.
export PYTHONPATH=. MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
# The pod's /workspace is fresh (no HF cache) → point HF_HOME at the shared,
# pre-seeded cache so the Qwen3-VL tokenizer (server) and libero assets (client)
# load WITHOUT re-downloading. Seeded once from a working node into /mnt/ddn/hf_cache.
export HF_HOME=${HF_HOME:-/mnt/ddn/hf_cache}
# Seed libero's ~/.libero/config.yaml so the package does NOT block on its
# interactive first-run "specify dataset folder? (Y/N)" prompt (fresh-HOME pod has
# no stdin → EOFError). Assets auto-download to ~/.cache/libero from the shared HF
# cache (jadechoghari/libero-assets) — offline since HF_HOME points there.
mkdir -p "$HOME/.libero"
[ -f "$HOME/.libero/config.yaml" ] || cp /mnt/ddn/hf_cache/libero_config.yaml "$HOME/.libero/config.yaml"

# 8-way split: 4 suites x 2 task-halves
SUITE=(libero_spatial libero_spatial libero_object libero_object libero_goal libero_goal libero_10 libero_10)
TIDS=("0 1 2 3 4" "5 6 7 8 9" "0 1 2 3 4" "5 6 7 8 9" "0 1 2 3 4" "5 6 7 8 9" "0 1 2 3 4" "5 6 7 8 9")

echo ">>> Launching 8 servers (iter=$ITER, num_steps=$NUM_STEPS)..."
for i in $(seq 0 7); do
  PORT=$((8001 + i))
  CUDA_VISIBLE_DEVICES=$i python -u -m cosmos_framework.scripts.action_policy_server_libero \
    --checkpoint-path "$CKPT_DIR/checkpoints/$ITER_PAD" --config-file "$CKPT_DIR/config.yaml" \
    --action-stats-path "$NORM" --action-normalization minmax \
    --raw-action-dim 7 --action-chunk-size 16 --fps 10 --num-steps "$NUM_STEPS" \
    --use-state --state-stats-key state --port "$PORT" > "$OUT/server_gpu${i}.log" 2>&1 &
done

echo ">>> Waiting for servers to be ready..."
for i in $(seq 0 7); do
  PORT=$((8001 + i))
  for _ in $(seq 1 40); do
    curl -s -m 3 "http://127.0.0.1:${PORT}/info" >/dev/null 2>&1 && break
    sleep 8
  done
  curl -s -m 4 "http://127.0.0.1:${PORT}/info" >/dev/null 2>&1 \
    && echo "    GPU$i port$PORT READY" || { echo "    GPU$i port$PORT FAILED"; tail -3 "$OUT/server_gpu${i}.log"; }
done

echo ">>> Launching 8 clients..."
for i in $(seq 0 7); do
  PORT=$((8001 + i))
  python -u -m cosmos_framework.scripts.eval_libero_client \
    --server-url "http://127.0.0.1:${PORT}" \
    --task-suites "${SUITE[$i]}" --task-ids ${TIDS[$i]} \
    --num-trials-per-task "$TRIALS" \
    --two-cam --image-size 512 --flip-mode 180 --gripper-map identity \
    --no-save-videos --out-dir "$OUT/gpu${i}" > "$OUT/client_gpu${i}.log" 2>&1 &
done
wait
echo ">>> All clients done. Aggregating..."

python - "$OUT" <<'PY'
import sys, glob, json
from collections import defaultdict
out = sys.argv[1]
agg = defaultdict(lambda: [0, 0])  # suite -> [succ, ep]
# dedupe: the recursive glob already covers the direct path — set() avoids double-counting.
for jp in sorted(set(glob.glob(f"{out}/gpu*/**/libero_eval_*.json", recursive=True))):
    d = json.loads(open(jp).read())
    for s, r in d.get("suites", {}).items():
        agg[s][0] += r["successes"]; agg[s][1] += r["episodes"]
print("\n========== LIBERO 2-cam FULL EVAL ==========")
ts, te = 0, 0
for s in sorted(agg):
    sc, ep = agg[s]; ts += sc; te += ep
    print(f"  {s:16s}: {sc/ep:6.1%}  ({sc}/{ep})")
print(f"  {'OVERALL':16s}: {ts/te:6.1%}  ({ts}/{te})" if te else "  no results")
json.dump({s: {"successes": v[0], "episodes": v[1]} for s, v in agg.items()} | {"overall": {"successes": ts, "episodes": te}},
          open(f"{out}/AGGREGATE.json", "w"), indent=2)
print(f"Aggregate: {out}/AGGREGATE.json")
PY
echo ">>> Done."

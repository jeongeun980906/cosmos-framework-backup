# Closed-loop evaluation for Cosmos3 action policies (LIBERO & PushT)

This guide runs **closed-loop simulator evaluation** (real task success rate) for
action policies post-trained with the recipes in
[`action_policy_libero_posttrain.md`](./action_policy_libero_posttrain.md) and the
PushT recipe.

## Architecture: server + simulator client

Evaluation is split into two processes that talk over HTTP, so the simulator
dependencies never have to coexist with the training/model stack:

```
[simulator + eval client]  --POST /predict {image, prompt, domain_name}-->  [policy server]
   (libero / gym-pusht)     <--{action: [[D], ...], video} -----------------  (cosmos-framework + GPU)
```

* **Policy server** — `cosmos_framework.scripts.action_policy_server_libero`.
  Domain-generic: it reads `domain_name` per request, so the *same* server serves
  both LIBERO (`domain_name="libero"`, 7D) and PushT (`domain_name="pusht"`, 2D).
  It also **denormalizes** the predicted action, so you must pass the same stats
  JSON the policy was trained with via `--action-stats-path`.
* **Eval client** — `eval_libero_client.py` / `eval_pusht_client.py`. Steps the
  simulator, queries the server, executes the action chunk open-loop, scores
  success, and writes a JSON report + rollout videos.

Run the server in the cosmos-framework `.venv` (needs a GPU); run the client in a
separate sim venv (CPU/EGL).

---

## 1. Export the trained checkpoint (one-time)

The server can load a training-time DCP directly (with `--config-file`), or a
consolidated export (recommended):

```bash
PYTHONPATH=. python -m cosmos_framework.scripts.export_model \
  --checkpoint-path /path/to/job/checkpoints/iter_000040000 \
  --config-file    /path/to/train-output/config.yaml \
  -o               /path/to/train-output/model
```

---

## 2. LIBERO

### 2.1 Install the LIBERO simulator (separate venv)

Mirrors the Cosmos-Policy `libero` dependency group — install the PyPI `libero`
package and let it pull `robosuite` transitively (do NOT pin robosuite yourself;
that causes a resolver conflict). The client needs no torch from us, but `libero`
pulls a full stack, so use an isolated py3.10 venv:

```bash
cd cosmos-framework/tools/eval_envs
uv venv --python 3.10 libero-client
source libero-client/bin/activate
uv pip install bddl cloudpickle draccus easydict gym "imageio[ffmpeg]" \
               libero mujoco==3.3.2 requests pillow

# First import writes ~/.libero/config.yaml interactively; answer N for defaults:
yes N | python -c "from libero.libero import benchmark, get_libero_path; print('LIBERO OK')"
```

LIBERO downloads its bddl/asset files (~586 files) to `~/.cache/libero` on first
env creation. For headless servers set `MUJOCO_GL=egl` (the client defaults to
this). Verified: `OffScreenRenderEnv` renders 256×256 agentview + wrist images
under EGL, and `eval_libero_client.py` runs the full 10-task rollout loop.

### 2.2 Start the policy server (cosmos-framework venv, GPU)

```bash
PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_libero \
  --checkpoint-path /path/to/train-output/model \
  --action-stats-path cosmos_framework/data/vfm/action/datasets/libero_lerobot_normalization.json \
  --action-normalization minmax --raw-action-dim 7 \
  --action-chunk-size 16 --fps 20 --port 8000
```

### 2.3 Run the eval (libero-client venv)

```bash
MUJOCO_GL=egl python -m cosmos_framework.scripts.eval_libero_client \
  --server-url http://127.0.0.1:8000 \
  --task-suites libero_spatial libero_object libero_goal libero_10 \
  --num-trials-per-task 50 --num-open-loop-steps 8 \
  --out-dir outputs/libero_eval
```

Runs 50 episodes × 10 tasks × 4 suites = 2000 trials and prints a per-suite +
overall success rate, writing `outputs/libero_eval/libero_eval_<ts>.json`.

> **Gripper convention check.** The policy outputs the dataset's gripper ∈ [0, 1];
> the robosuite controller wants [-1, 1] (−1 = open). The client defaults to
> `--gripper-map rescale` (`2g−1`). Watch the first rollout video — if the gripper
> grasps when it should release, switch to `--gripper-map invert`.

---

## 3. PushT

### 3.1 Install gym-pusht (separate venv)

```bash
cd cosmos-framework/tools/eval_envs
uv venv --python 3.10 pusht-client
source pusht-client/bin/activate
uv pip install gym-pusht requests pillow "imageio[ffmpeg]"
# gym-pusht needs the older pymunk/gymnasium API (matches the Cosmos-Policy pusht
# group); newer pymunk drops Space.add_collision_handler:
uv pip install "pymunk==6.6.0" "gymnasium==0.29.1"
```

### 3.2 Export PushT stats (one-time) & start the server

```bash
# stats: re-derive the exact in-memory min/max the policy trained with
python tools/gen_pusht_normalization.py \
  --root /path/to/PushT-Cosmos-Policy-lerobot/pusht-baseline-top100

PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_libero \
  --checkpoint-path /path/to/pusht-train-output/model \
  --action-stats-path cosmos_framework/data/vfm/action/datasets/pusht_lerobot_normalization.json \
  --action-normalization minmax --raw-action-dim 2 \
  --action-chunk-size 8 --fps 10 --port 8000
```

### 3.3 Run the eval (pusht-client venv)

```bash
python -m cosmos_framework.scripts.eval_pusht_client \
  --server-url http://127.0.0.1:8000 \
  --num-episodes 50 --num-open-loop-steps 8 \
  --out-dir outputs/pusht_eval
```

Success = T-block coverage ≥ 0.95 (standard gym-pusht criterion). Writes
`outputs/pusht_eval/pusht_eval_<ts>.json`.

---

## Knobs that matter

| Flag | Meaning |
|------|---------|
| `--action-chunk-size` (server) | Action steps the policy predicts per query. Must match the trained `chunk_length` (LIBERO 16, PushT 8). |
| `--num-open-loop-steps` (client) | How many of those steps to execute before re-querying. Lower = more reactive, slower. |
| `--action-stats-path` (server) | The stats JSON the policy was normalized with. **Required** for correct denormalization. |
| `--raw-action-dim` (server) | Unpadded action dim used for masking (LIBERO 7, PushT 2). |
| `MUJOCO_GL` | `egl` for headless GPU servers; `glfw` for a local display. |

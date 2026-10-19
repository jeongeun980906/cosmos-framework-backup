# Cosmos3-Edge RoboCasa 24-task action-policy SFT

Post-trains `nvidia/Cosmos3-Edge` into a RoboCasa kitchen policy on **the same data
the Cosmos-Policy RoboCasa runs use** — `nvidia/RoboCasa-Cosmos-Policy`
`success_only/` (24 tasks, 1199 demos, 316,995 frames), read in place from HDF5.
The point is a like-for-like arm next to the Cosmos-Policy (Predict2.5-2B) runs, so
the action target is normalized with the release's own `dataset_statistics.json`
min/max and both stacks optimize the identical scale.

| Piece            | Path                                                                                            |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| Dataset          | `cosmos_framework/data/generator/action/datasets/robocasa_hdf5_dataset.py` (`RoboCasaHDF5Dataset`) |
| SFT wrapper      | `get_action_robocasa_sft_dataset` in `.../datasets/action_sft_dataset.py`                       |
| Norm stats       | `.../datasets/robocasa_hdf5_normalization.json` (from `tools/gen_robocasa_normalization.py`)    |
| Experiment       | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robocasa_edge.py` |
| Run TOML         | `examples/toml/sft_config/action_policy_robocasa_edge.toml` (+ `_smoke`, `_frommid`)            |
| Train (k8s)      | `/mnt/ddn/kube_yaml/cosmos-robocasa24-c3edge-{base,mid}-3node.yaml`                             |
| Inference server | `cosmos_framework/scripts/action_policy_server_libero.py` (domain-generic)                      |
| Closed-loop eval | `cosmos_framework/scripts/eval_robocasa_client.py` + `examples/eval_robocasa_24task_8gpu.sh`    |
| Eval (k8s)       | `/mnt/ddn/kube_yaml/cosmos-robocasa24-c3edge-{base,mid}-eval-8gpu.yaml`                         |

## 1. Data

No conversion step. `RoboCasaHDF5Dataset` reads the release tree directly — the
frames are uncompressed contiguous `uint8` inside the HDF5, so random window reads
are cheap and re-encoding 135 GB to LeRobot video would buy nothing.

```
<root>/<group>/<Task>/<date>/demo_gentex_im128_randcams.hdf5
  data/demo_N   attrs: task_description
    actions        (T, 12)   [:7] = [dpos(3), daxang(3), gripper]; [7:] = mobile base (dropped)
    obs/robot0_agentview_left_rgb / _right_rgb / robot0_eye_in_hand_rgb   (T, 224, 224, 3)
    obs/ee_pos (T,3)  obs/ee_ori (T,3, axis-angle)  obs/gripper_states (T,2)
```

Per sample: a 33-frame window (chunk 32 @ 20 Hz, VAE-legal `33 = 1 + 4*8`), the three
cameras **vertically concatenated** into one 672×224 frame, and a `(33, 7)` action
tensor. Row 0 is the absolute proprio `[ee_pos(3), ee_ori(3), gripper_states[0]]`
(the `use_state` conditioning row); rows 1..32 are the delta-EE commands
`actions[t0:t0+32]`.

Both normalization blocks live in one JSON, and the **same file** is passed to the
eval server so train-time normalization and eval-time denormalization cannot drift:

* `"robocasa"` — the delta action. `min`/`max` copied verbatim from the release
  `dataset_statistics.json` (a fresh scan agrees to 3e-8).
* `"robocasa_state"` — the absolute proprio. Separate because the kitchen pose range
  reaches x ≈ 6.2 m; the delta stats would clamp it to the rail.

Regenerate with:

```bash
python tools/gen_robocasa_normalization.py \
  --dataset-root /mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only
```

### Proprio goes in through the action stream

There is no separate proprio input in Cosmos3. `use_state=True` makes the action
tensor `(chunk+1, D)`, and `build_sequence_plan_from_mode` then returns
`condition_frame_indexes_action=[0]` (Case B, `transforms.py:308-330`) — row 0 is
clean conditioning, the rest is supervised. This is the same path DROID, PushT and
the GR1/EgoVerse Edge recipes use; the LIBERO `frame_wise_relative` recipe is the one
that runs without it.

## 2. Train

```bash
export ROBOCASA_ROOT=/mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only
export EDGE_LOAD_PATH=/mnt/ddn/ckpt/cosmos3_edge_dcp     # DCP PARENT — the loader appends /model
export WAN_VAE_PATH=/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/mnt/ddn/tmp/robocasa24_edge
export LD_LIBRARY_PATH=''                               # NGC container: avoid the torch._C import error

# smoke: 20 iters, 1 GPU, one task — surfaces a config/shape break in ~2 min
python -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/action_policy_robocasa_edge_smoke.toml

# full: 1 node x 8 GPU
torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/action_policy_robocasa_edge.toml
```

lr 1e-4, warmup 500, cycle 20000, `save_iter=5000` (~120 GB of checkpoints), FSDP
shard 8 × replicate 3 = 24 ranks, global batch `max_samples_per_batch(16) × 24 = 384`
(the shipped jobs are 3-node; set `replicate = 1` in the TOML for a single node).

**Parallelism must be set in the TOML, never on the CLI.** `[model]` is a structured
node, so a trailing `-- model.parallelism.data_parallel_replicate_degree=3` raises
`ConfigKeyError: Key 'parallelism' is not in struct` on rank 0 and brings the whole job
down at rendezvous — the other workers then die with a C10d `RendezvousConnectionError`,
which is a downstream symptom, not the cause. (The resolved path is actually
`model.config.parallelism`; `trainer.*` and `dataloader_train.*` CLI overrides do work.)

**LR schedule.** `cycle_lengths` *includes* `warm_up_steps`: the multiplier ramps
0 → 1 over the first 500 steps, then decays linearly to `f_min = 0` at exactly step
20000 — monotonic, and it never bottoms out mid-run. **Keep `cycle_lengths ==
trainer.max_iter`.** If `cycle < max_iter`, `LambdaLinearScheduler.find_in_interval`
returns `None` past the cycle end and the run dies with a `TypeError` *at the end* of
training; if `cycle > max_iter`, the run stops at a still-high LR (PushT's
`cycle 40000 / max_iter 20000` ends at ~0.5 × peak).

**Training budget.** 20K iters at global batch 384 = 24.3 epochs over the
315,796-window dataset. The Cosmos-Policy RoboCasa arm runs 20K iters at global batch
600 = 37.9 epochs over an equally sized dataset (316,995 frames — the two are within
0.4%). Raising `max_samples_per_batch` to 25 makes iter count, global batch and epochs
all match, at ~44 h instead of ~28 h. (The one-off
`cosmos-robocasa24-t5-paper-4node.yaml` "paper schedule" — 45K iters at global batch
800, 113.6 epochs — is not what the other Cosmos-Policy arms use.)

Measured (2×H100, shard 2 — a strict upper bound on per-rank memory vs the 8-way
production shard): **5.0 s/iter at 16 samples/rank, peak 38.8 GB of 80** → ~28 h for
20K iters. (25/rank: 7.9 s/iter, 43.1 GB. batch 1: 0.38 s/iter.)

### Two arms: `base` vs `mid`

The recipe ships as a pair that differs in exactly one thing — where the weights
started:

| arm | experiment | loads | `keys_to_skip_loading` |
| --- | --- | --- | --- |
| **base** | `action_policy_robocasa_edge` | public `nvidia/Cosmos3-Edge` (`EDGE_LOAD_PATH`) | `["net_ema."]` |
| **mid** | `action_policy_robocasa_edge_frommid` | our EgoVerse-trained Edge (`EGOVERSE_EDGE_LOAD_PATH`) | `[]` — generation **and** action heads |
| **midhr** | `action_policy_robocasa_edge_frommid_headreset` | same EgoVerse checkpoint | `["net_ema.", "action2llm", "llm2action", "action_modality_embed", "action_pos_embed"]` — backbone only |

**Why midhr exists.** Measured over the first 100 iters, mid runs ~1.4 *higher* than
base (0–49: 11.09 vs 9.63; 50–99: 3.50 vs 2.19) — the warm start costs rather than
helps at the start. Total loss is `10x(vision) + 10x(action)`, so the base/mid pair
alone cannot say whether that is (a) the EgoVerse action head being wrong for this
embodiment — it spends 57 of its 64 channels on hand_pose (domain 3) where RoboCasa
uses 7 delta-EE channels (domain 5) — or (b) the backbone failing to transfer. midhr
keeps the backbone and drops the heads: converging to base-or-better says (a), staying
worse says (b). Expect midhr to start highest of the three (its heads init random, not
from the Edge base's trained ones) — judge it on final success rate, not on where the
curve starts.

Everything else (batch, lr, schedule, `save_iter`, data) is identical, so a base/mid
difference reads as EgoVerse pre-training transfer and nothing else. The default mid
source is the v10 ICL 8K checkpoint — the one the `cosmos3-edge-{he,egodex}` dump jobs
call "우리 v10 ICL post-train 8K":

```
/mnt/ddn/tmp/egoverse_v10_icl_edge_4node/cosmos3_action/action_sft/\
  action_policy_egoverse_v10_icl_edge/checkpoints/iter_000008000
```

`load_training_state` stays `False`, so the mid arm does **not** resume EgoVerse's
optimizer/scheduler/iteration — it runs a fresh 20K schedule off those weights.

Caveat, and it is not small: unlike the GR1 from-mid pair (57-dim hand_pose → 57-dim
hand_pose), this crosses **action spaces**. EgoVerse trained the head on 57-dim
hand_pose (`domain_id` 3); RoboCasa uses the 7-dim delta-EE command (`domain_id` 5).
Shapes still line up (both pad to `max_action_dim=64`), so the head starts from
hand_pose-tuned weights and re-adapts under its 5x lr multiplier — but what this arm
actually tests is **backbone** transfer. The `domain_id` 5 embedding row is untouched
by EgoVerse training (only domain 3 received gradients), so it still carries the Edge
base value.

**`uvx` must be on PATH.** The Edge VLM processor is fetched via
`CheckpointDirHf.download()`, which shells out to `uvx hf download`. With
`HF_HUB_OFFLINE=1` it resolves from `/mnt/ddn/hf_cache` without network, but the
binary itself must exist or model init dies with `FileNotFoundError: 'uvx'`. Both
kube yamls install it pod-locally.

## 3. Closed-loop eval

The policy server is domain-generic; RoboCasa needs three flags beyond the LIBERO
invocation:

```bash
python -m cosmos_framework.scripts.action_policy_server_libero \
  --checkpoint-path <run>/checkpoints/iter_000020000 --config-file <run>/config.yaml \
  --action-stats-path cosmos_framework/data/generator/action/datasets/robocasa_hdf5_normalization.json \
  --action-stats-key robocasa --action-normalization minmax \
  --use-state --state-stats-key robocasa_state \
  --append-duration-fps False --append-resolution-info False \
  --resolution 480 --raw-action-dim 7 --action-chunk-size 32 --fps 20 --port 8000
```

Then the client, in a venv that has robocasa (robosuite/mujoco pins conflict with the
training env):

```bash
MUJOCO_GL=egl PYTHONPATH=/mnt/ddn/cosmos-framework \
  /mnt/ddn/cosmos_policy/.venv/bin/python -m cosmos_framework.scripts.eval_robocasa_client \
    --server-url http://127.0.0.1:8000 --num-trials-per-task 50 --num-open-loop-steps 16
```

The client talks to **`/predict_batch`**, not `/predict`. Both run the same diffusion
forward and return byte-identical actions (verified: `max diff 0.0`), but `/predict`
also VAE-decodes the 33-frame generated video and base64s it back — which the client
discards. Measured on one H100 at `num_steps=15`: **0.61 s vs 3.60 s per request, 5.9x**.
Eval wall-clock is request-bound, so that is roughly ~2 h vs ~6 h for the full 24 x 50
sweep on 8 GPUs. `--want-generated-video` takes the slow path when you actually want to
look at the world model's output (`--save-videos` records the ENV frames and does not
need it).

Or the whole 24-task sweep on one 8-GPU node: `bash examples/eval_robocasa_24task_8gpu.sh`,
which is what `cosmos-robocasa24-c3edge-eval-8gpu.yaml` runs. It waits for the
checkpoint (so it can be applied before training finishes), starts 8 servers, feeds a
flock'd queue of the 24 tasks to 8 clients, and writes `summary.txt` in the same shape
as the Cosmos-Policy `cosmos-robocasa24-*-eval-8gpu.yaml` jobs so both land in one
table. Naming follows that convention: arm `c3edge`, train `…-c3edge-3node.yaml`,
eval `…-c3edge-eval-8gpu.yaml`.

### Verified: train-time and eval-time inputs are identical

The loop has been run locally twice (150-iter checkpoint, `PnPStoveToCounter`),
and — more usefully — the same dataset window was pushed through BOTH paths and the
model-facing tensors compared: the training path (`RoboCasaHDF5Dataset` →
`ActionTransformPipeline`) against the server's real `_prep_policy_item` fed the
window's frame 0 and raw proprio.

| | result |
| --- | --- |
| video tensor | both `(3, 33, 832, 480)`; **frame 0 pixel-identical**, `max diff = 0` |
| `action[0]` (state row) | identical to 1e-6 — same separate-stats normalization |
| `sequence_plan` | identical (`vision [0]`, `action [0]`) |
| prompt string | **identical** (after the fix below) |
| `raw_action_dim` | 7 on both sides |
| action de/normalization | round-trip exact: `max err 6e-8` over a 32-step chunk |

The rollout itself returns a `(32, 7)` chunk — the `use_state` row correctly dropped —
with every value inside the release's action range, the gripper column in `[-1, 1]`,
and no collapse across the chunk. Success is 0% at 150 iters, as expected.

Four real defects that comparison caught:

* **`import robocasa` registers the kitchen envs.** Without it `robosuite.make` raises
  `Environment PnPStoveToCounter not found`; the client now imports it in `_make_env`.
* **The prompt did not match.** Training appended a viewpoint sentence and an
  `IdleFrames: N out of 33` sentence the server has no way to produce, and the two
  sides disagreed on the duration and resolution numbers (training measures the
  *padded* 832x480 canvas, the server the *pre-pad* 672x224 frame). The recipe now
  turns all four appenders off, so the prompt is `CAPTION_PREFIX + task` and nothing
  else. Those sentences were constant across every sample here anyway — and
  `IdleFrames` is computed from the ground-truth action chunk, i.e. a cue that
  disappears at eval.
* **The server could not see the recipe's flag.** It resolves `append_duration_fps`
  by exact key name inside `dataloader_train`, but the action getters spell it
  `append_duration_fps_timestamps`, so it silently defaulted to True. Hence the new
  explicit `--append-duration-fps` / `--append-resolution-info` overrides, which the
  eval script passes as `False`. **Do not drop them** — without them the served
  prompt regains a sentence training never had.

* **The client was paying 5.9x for a video it discards.** It queried `/predict`, which
  VAE-decodes the generated 33-frame clip; it now queries `/predict_batch`, which skips
  that and returns identical actions. See §3.

One observation, not a defect: robosuite 1.5 logs `Loading controller configuration
from … default_pandaomron.json`, so the 1.4-style OSC_POSE dict passed via
`controller_configs` (`DEFAULT_CONTROLLER_CONFIG` / `--controller-config-pkl`) appears
to be superseded by the composite-controller path. Do not assume that dict is in
effect. The Cosmos-Policy harness passes the same dict to the same robosuite build, so
both arms run the same controller — which is what parity needs.

### Parity items — check these first if success rate is near zero

* **`--resolution 480`.** The 3-cam stack is 672×224, whose *min* side is 224, so the
  server's shape-derived tier is `"256"` while training snapped to the 480 tier
  (832×480 canvas, ~277 px per view). Dropping the flag is a silent canvas mismatch.
* **Vertical flip.** The env renders upside-down and the dataset was recorded with
  `np.flipud` per camera, so the client flips too (`--no-flip-images` is debug-only).
* **Camera order.** `[agentview_left, agentview_right, eye_in_hand]`, top to bottom,
  in both `DEFAULT_CAMERAS` and `CAMERA_OBS_KEYS`.
* **Prompt.** `CAPTION_PREFIX + task` — the client duplicates the constant (to stay
  torch-free), so the two must be edited together.
* **Gripper.** RoboCasa stores the raw env command in `[-1, 1]`, so `--gripper-map
  identity` (LIBERO needs `rescale`; flip to `invert` only if rollout videos show the
  gripper inverted).
* **State channels.** `[eef_pos, quat2axisangle(eef_quat), gripper_qpos[0]]` — the
  live-env equivalents of `obs/ee_pos`, `obs/ee_ori`, `obs/gripper_states[0]`.

## 4. Reading the comparison honestly

This recipe is **not** a reimplementation of the Cosmos-Policy RoboCasa model. It
reproduces the data, the action space, the target scale and (at 3 nodes) the training
budget. The remaining differences cut both ways.

Missing here:

* the **value-function head**;
* the **future-proprio** auxiliary target (`predict_future_states=True` in the
  Cosmos-Policy dataset);
* the 50/50 mixing of `all_episodes` rollouts (this trains on `success_only` demos);
* T5 text conditioning — Cosmos3-Edge conditions on the VLM prompt directly, which
  also means an unseen `B`-split instruction phrasing is not an embedding-cache miss
  the way it is for the T5-conditioned runs.

Richer here — **future-video prediction**. Cosmos3 generates video by default
(`vision_gen=True`), and in `wam` mode only frame 0 is conditioning
(`condition_frame_indexes_vision=[0]`), so all 32 remaining frames of the 3-camera
stack are generated and supervised at `loss_scale=10.0`. The Cosmos-Policy layout
predicts a **single** future frame per camera, at `t + chunk_size`
(`robocasa_dataset.py:886`). The 3510 tokens/sample this recipe spends are largely
that full-video target, so it is a capability difference, not just a cost.

Camera fidelity runs the other way: each view lands at ~277 px here versus the 224 px
Cosmos-Policy feeds per camera, because the 480 tier was chosen over the 256 tier the
GR1 Edge recipe uses (which would cut each view to ~106 px).

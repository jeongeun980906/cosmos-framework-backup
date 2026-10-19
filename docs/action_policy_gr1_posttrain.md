# GR1 conditioning-policy post-training (Cosmos3-Nano)

Second-stage fine-tuning that turns the EgoVerse-pretrained action policy into a
**GR1 bimanual manipulation policy** for robocasa. The policy predicts a 32-step
`action57` chunk for the GR1 humanoid (the *query* embodiment) **conditioned on a
retrieved demo clip** — at `t=0` a same-task TaskDemo opening, at `t>0` a
`SinkEnvForPlay` play clip matched from the robot's recent motion. Closed-loop eval
(server + robocasa client) is in §8; the retrieval contract is
[`gr1_inference_retrieval_spec.md`](./gr1_inference_retrieval_spec.md).

This doc covers **how to launch training, every knob that makes GR1 differ, the
stride-4 / per-timestep dataloader, how to port the recipe to another robot, what
code was added/changed, and how the inference server transforms model output into
env actions.**

---

## 1. What the recipe does

* **Query embodiment** `gr1`: the GR1 humanoid. Trained to output its own next
  32-step future given (a) the current 3-camera obs, (b) its absolute state, and
  (c) a conditioning demo clip.
* **Conditioning / demo embodiment** `mimicdroid`: `SinkEnvForPlay` play demos
  (`t>0`) and the 12 canonical TaskDemos (`t=0`). Both are robocasa, 20 Hz,
  `action57` — same units as GR1, which is why matching works directly (§6).
* **Warm-start**, not from scratch: continue from the EgoVerse-5s checkpoint
  (`iter_000010000`), **loading** the generation + action heads (`keys_to_skip_loading=[]`)
  instead of re-initializing them.
* **Data (2026-07-01, eval-heavy)**: 50 GR1 demos/task (4 tasks). Each = **35
  eval-style + 15 diverse** demos, because the seen online eval is a **single fixed
  style+layout per task** (see [[gr1-seen-eval-style-layout]] and §2). Concentrating
  on the eval condition maximizes seen-eval score; 15 diverse curb overfitting.

---

## 2. Data pipeline (model side) — augmented TaskDemos → training pairs

The GR1 training episodes are **augmented TaskDemos replayed onto the GR1 robot**,
then converted to egoverse zarr and paired. Full chain (all re-runnable):

```
 augment_pnp.py / augment_door.py / gen_pnp_cabinet.py      # 1. augment a TaskDemo:
   → DemoTwoHand hdf5 (varied style + spawn), per task         randomize style/spawn, retarget
                                                                traj; eval-style-concentrated
                                                                (--styles <eval_style>)
 robocasa/scripts/replay_to_gr1.py                          # 2. replay onto GR1 humanoid
   → GR1/<task>/003/demo_im128_notp.hdf5  (50 demos/task)      (reachability filter, records a
                                                                recorded home→demo-start APPROACH
                                                                + closed-loop settle so t=0 is valid)
 egoverse_pair_builder/73_gr1_to_egoverse57.py             # 3. → egoverse/<task>_003__demo_{0..49}
   (DST=/mnt/ddn/dataset/egoverse; no argparse — edit DST)      action57 + state57 zarr (200 eps)
 tools/add_concat_view_3cam.py --apply                     # 4. add images.{sideleft,sideright}
   → images.{front_1,sideleft,sideright} on all 200 eps         from the raw hdf5 (sidecams_done)
 egoverse_pair_builder/78_future_pairs_abstraj.py --play 1x # 5a. t>0 pairs: GR1 PAST window ↔ play
   → egoverse_pairs_v2/gr1_future_pairs_1x.npz                  clip, ABS both-hand 6D-pose DTW
 egoverse_pair_builder/77_gr1_t0_taskdemo.py --speed 1x     # 5b. t=0 pairs: force map to same-task
   → egoverse_pairs_v2/gr1_t0_taskdemo.npz                      TaskDemo opening [0:32]
 tools/build_gr1_posttrain_pairs.py --speeds 1x \           # 6. merge t>0 + t=0 → training windows
   --out /mnt/ddn/dataset/egoverse_pairs_gr1_1x                 (the config's pairs_dir)
```

Result (last run): `egoverse_pairs_gr1_1x` `windows=14159` (gr1=12475 [t>0 12275 +
t=0 200] over **200 episodes = 50/task×4**, mimicdroid=1684). `action57`/`state57`
layout = `[HEAD9 | Leef9 | Lhand15 | Reef9 | Rhand15]` (see
[`action57_state57_convention.md`](./action57_state57_convention.md)); `action` is a
per-step **body-frame delta**, `state` is the absolute waist-frame pose.

> Only the GR1 **episodes** changed (new eval-heavy demos), so only steps 3–6 rerun.
> The 12 canonical TaskDemos (t=0 pool) and the `SinkEnvForPlay` play pool are
> unchanged. 2x is optional — the 1x config reads `egoverse_pairs_gr1_1x` only.

---

## 3. Launch (how to train)

No dedicated `launch_sft_action_policy_gr1.sh`; drive the shared launcher with the
GR1 TOML. The recipe TOML selects the registered experiment and sets iters/ckpt.

```bash
TOML_FILE=examples/toml/sft_config/action_policy_gr1_posttrain_1x_repro.toml \
POSTTRAIN_LOAD_PATH=<EgoVerse-5s ckpt dir>/checkpoints/iter_000010000 \
WAN_VAE_PATH=<Wan2.2_VAE.pth> \
NPROC_PER_NODE=8 \
bash -c 'source examples/_sft_launcher_common.sh'
```

or the underlying command directly:

```bash
POSTTRAIN_LOAD_PATH=... WAN_VAE_PATH=... \
torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/action_policy_gr1_posttrain_1x_repro.toml
```

* TOML `experiment = action_policy_gr1_posttrain_1x_nano`, `max_iter = 10000`,
  `[checkpoint] load_path = ${POSTTRAIN_LOAD_PATH}`. **`load_path` must be an
  EgoVerse-5s checkpoint dir, NOT the Cosmos3-Nano base** (we warm-start the trained
  policy). Leave `GR1_PAIRS_DIR` unset → the 1x experiment's default
  `egoverse_pairs_gr1_1x` applies.
* **Smoke (2 GPU, fits by dropping batch + EMA):**
  `TOML_FILE=examples/toml/sft_config/action_policy_gr1_posttrain_smoke.toml`.
* **1x vs 2x**: `action_policy_gr1_posttrain_1x_nano` (play 1x, `egoverse_pairs_gr1_1x`)
  and `..._2x_nano` (play 2x) are two SEPARATE runs to compare play speeds. Seen eval
  used 1x.

---

## 4. STRIDE=4 → per-timestep training (IMPORTANT — do not drop)

The pair builder (`78_future_pairs_abstraj`) emits query boundaries only at
**`STRIDE=4`** (`range(4, n, 4)`): one training window every 4th frame. Training on
those windows alone would **skip 3 of every 4 frames** — the policy would never see
the in-between phases. We fix this in the dataloader so **every timestep is a
training target**:

* `EgoVersePairedDataset.read_window(widx, frame_offset=δ)` (in
  `egoverse_zarr_dataset.py`) shifts the read start by δ raw frames — **video,
  action deltas, and `state0` all advance in lockstep** — without changing clip
  length (the `(target_frames, chunk)` come from the stored span and are δ-invariant).
* `__getitem__` samples ONE `δ ∈ {0 .. min(STRIDE−1, room_q, room_c)}` per item and
  passes the **SAME δ to BOTH the query and the demo**, so the condition and target
  stay phase-aligned (at δ=1 the condition is the demo advanced by 1, target = GR1
  `action[1:33]`, etc.).
* Config knob **`per_frame_stride`** (`_ds["per_frame_stride"] = _GR1_QUERY_STRIDE = 4`).
  **Default is `1` = original behavior**; only the GR1 path opts in, so all other
  recipes are unaffected. δ is clamped against `target_frames` so the action read is
  always in-bounds.
* At **inference** the analogue is automatic — the robot is physically at frame `t`,
  so no special handling is needed (see §8 / `gr1_inference_retrieval_spec.md` §6b).

This is required: without `per_frame_stride`, the 4× stride would leave 75% of frames
untrained.

---

## 5. Config knobs that make GR1 differ

`action_policy_gr1_posttrain_1x_nano` ← `..._5s_nano` ← `action_policy_egoverse_5s_nano`.
The deltas vs the EgoVerse pre-train (all on
`_ds = _cfg[...]["datasets"]["egoverse"]["dataset"]`):

| knob | value | why |
|---|---|---|
| `pairs_dir` | `${GR1_PAIRS_DIR,/mnt/ddn/dataset/egoverse_pairs_gr1_1x}` | GR1 query → play/taskdemo windows (§2) |
| `chunk_length` | **32** | 32-step action chunk (EgoVerse used 28) |
| `fps` | **20.0** | GR1+play are 20 Hz (human rigs are 30); drives PE fps-modulation + per-emb frame count |
| `use_state` | **True** | prepend absolute `state0` as conditioning `action[0]` → tensor `(33,57)` = state0 + 32 deltas |
| `cams_override` | `{mimicdroid,gr1: (front_1, sideleft, sideright)}` | 3-cam vertical concat (matches §2 step 4) |
| `speed_stats_path` | **None** | no 0–1 speed conditioning for GR1 |
| `per_frame_stride` | **4** | per-timestep training (§4) |
| `checkpoint.keys_to_skip_loading` | **`[]`** | LOAD gen + action heads from the warm-start ckpt (base recipe inits them fresh) |

Normalization: per-embodiment stats keyed `"<emb>"` (action deltas) and `"<emb>_state"`
(absolute state0) — GR1 and mimicdroid each carry their own; loaded as torch tensors
in the dataset ctor. De-quantile at inference uses the same q01/q99 (§8).

---

## 6. Porting the recipe to another robot (dataloader)

To train the SAME conditioning recipe for a different robot `X`, you produce
`X`-embodiment egoverse episodes + pairs and register `X`; the dataloader/model are
generic. Steps:

1. **Register the embodiment** in `EMB_REGISTRY` (`egoverse_zarr_dataset.py`):
   ```python
   "X": EgoVerseEmbSpec(root="egoverse", dim=57, domain="hand_pose", fps=<hz>,
                        cams=("images.front_1", ...))   # or set via cams_override
   ```
   `root` = subdir under `data_root`; `dim` = action/state channel count; `fps`
   drives PE fps-modulation and the per-emb frame count; `domain` selects the
   domain-id embedding.
2. **Convert `X` demos → egoverse zarr** `X_.../action{dim}/action` + `state{dim}/state`
   (mirror `73_gr1_to_egoverse57.py`: write per-step body-frame **deltas** into
   `action`, absolute waist-frame pose into `state`, natural-language
   `task_description` in attrs). Add extra cams with `add_concat_view_3cam.py` if the
   reader will use `cams_override`.
3. **Build pairs** = query windows (from `X` episodes) matched to conditioning demos,
   in the `q_ep/m_ep/w0` layout `build_gr1_posttrain_pairs.py` expects (adapt
   `78_future_pairs_abstraj` for the descriptor/gating; add a t=0 rule if wanted).
   Output a `pairs_dir` (`windows.npz` + `pairs.npz`).
4. **Point the config** at it and set the per-emb knobs (§5): `pairs_dir`,
   `chunk_length`, `fps`, `use_state`, `cams_override`, `per_frame_stride`. Provide
   `"<X>"` + `"<X>_state"` normalization stats.
5. **Warm-start or fresh**: `keys_to_skip_loading=[]` to continue a trained policy,
   or the base default to init heads fresh.

The training unit is the leading `chunk_length+1` frames of each window; the reader
(`read_window`) handles video decode + concat, delta normalization, `state0`
prepend, δ shifting (§4), and captioning — all driven by `EMB_REGISTRY[emb]`. Nothing
model-side is embodiment-specific.

---

## 7. Code added / changed (this work)

**New files** (GR1 conditioning stack):
* `cosmos_framework/data/vfm/action/datasets/egoverse_zarr_dataset.py` —
  `EgoVersePairedDataset` + `EMB_REGISTRY`/`EgoVerseEmbSpec`; `read_window(frame_offset)`,
  `per_frame_stride` (§4), `use_state` state0 prepend, `cams_override` concat.
* `cosmos_framework/scripts/action_policy_server_gr1.py` — inference server (§8).
* `cosmos_framework/scripts/eval_gr1_client.py` — robocasa GR1 env client (§8).
* `cosmos_framework/scripts/gr1_online_retrieval.py` — live descriptor + matcher.
* configs: `action_policy_egoverse_{5s_,}nano.py`,
  `action_policy_gr1_posttrain_{1x,2x,5s}_nano.py` (+ `libero_nano`, `pusht_nano`).
* pair builders: `egoverse_pair_builder/{78_future_pairs_abstraj,77_gr1_t0_taskdemo}.py`,
  `tools/build_gr1_posttrain_pairs.py`, `tools/add_concat_view_3cam.py`,
  `tools/build_taskdemo_retrieval_pool.py`.

**Modified** (`git diff --stat`, model/data core to support paired conditioning +
per-frame + 3-cam concat): `model/vfm/omni_mot_model.py` (+126),
`data/vfm/action/datasets/action_sft_dataset.py` (+141),
`data/vfm/sequence_packing.py` (+138), `scripts/action_policy_server_libero.py` (+122),
`data/vfm/joint_dataloader.py`, `model/vfm/utils/data_and_condition.py`,
`data/vfm/action/datasets/__init__.py`, `data/vfm/action/domain_utils.py`,
`configs/base/config.py`, `utils/distributed.py`.

---

## 8. Inference server — architecture & transformations

Two processes over HTTP (they need different venvs): **server** (cosmos-framework
venv, the GPU model) and **client** (mimicdroid-robocasa venv, owns the GR1 env).
Format was verified **bit-exact** by a round-trip test (train↔eval identical).

```
 eval_gr1_client.py            action_policy_server_gr1.py
 ─────────────────             ──────────────────────────
 robocasa GR1 env  ──obs──▶  proprio {pL,pR,quatL,quatR,gripL,gripR}
   3 agentview cams ─images─▶  + 3 images (vertical concat)
                              │
                              ├─ retriever.push(proprio)   (build the live descriptor)
                              ├─ retrieve:  t=0 → TaskDemo opening [0:32] (gated by FULL task)
                              │             t>0 → play PAST-window abs-traj match
                              ├─ model.generate_samples_from_batch(query, condition)
                              │     → action57 chunk  (33,57) use_state → drop state0 → (32,57)
                              │     → de-quantile:  a = (a_norm+1)/2 · (q99−q01) + q01
                              └─ _to_env_action(act, proprio)  ▼
   env.step(env_action[t]) ◀──env_action (32,14)──────────────┘
```

### 8.1 Client obs → proprio / images
`get_proprio(obs)`: `pL,pR = robot0_base_to_{left,right}_eef_pos`,
`quatL,quatR = ..._eef_quat` (xyzw), `gripL,gripR = (last_cmd+1)/2 ∈[0,1]` (1=closed;
the client converts its `[−1,1]` command before sending).
`_images(obs)`: the 3 agentview cams (`center,sideleft,sideright`), each **flipped
upright `[::-1]`** (env obs is bottom-origin; training renders are upright), sent as
`req["images"]` in the SAME cam order as `cams_override`.

### 8.2 Server proprio → state57 (query conditioning input)
`_proprio_to_state57` builds the absolute `state57` the model was trained on:
`[HEAD9 | abs9_L | hand15_L | abs9_R | hand15_R]`, where `abs9(pos,quat)=[pos,
R[:,0], R[:,1]]` (rotation matrix cols 0,1) and `hand15` = the predefined
OPEN15→CLOSED15 lerp at the grasp signal. This becomes the query's `state0`
(normalized with `gr1_state` stats).

### 8.3 Retrieval (t=0 vs t>0)  — see `gr1_inference_retrieval_spec.md`
* `retr_task` = FULL task (e.g. `PnPSinkToCabinetL3`) gates taskdemo retrieval to
  that task's own demos; `task = _base_task(retr_task)` (strips L2/L3) is used for the
  language prompt.
* `t=0` (`_chunk==0`): `retriever.retrieve(retr_task)` then force `w0=0` → the
  same-task **TaskDemo opening `[0:32]`**.
* `t>0`: `play_retriever.retrieve(task)` → a `SinkEnvForPlay` PAST-window abs-traj
  match; condition = that play clip's FUTURE `[m_w0:m_w0+32]`.
* No hits yet (not enough history) → return a hold action.

### 8.4 Model output → env_action (`_to_env_action`) — the key transformation
The generated `action57` chunk is per-step **body-frame deltas** `eef9 = [Δpos(3),
6D-rot(ΔR cols 0,1)]` per arm. Convert to the GR1 OSC_POSE env action:

1. **De-quantile** `act_norm → act` with the per-emb `q01/q99`.
2. **Integrate onto the LIVE pose** (`_integrate_arm`, anchored on the measured eef
   from proprio, so each chunk re-anchors — no cross-chunk drift):
   `p_{t+1} = p_t + R_t·Δpos`, `R_{t+1} = R_t·ΔR` (`_sixd_to_R` = Gram-Schmidt on the
   two stored columns). Output = per-step absolute `(pos3, axis-angle3)` in the GR1
   **body-eef** frame (the frame `73_gr1_to_egoverse57` built from
   `robot0_base_to_*_eef_quat`).
3. **Body-eef → OSC control (gripper-site) frame** (`_apply_ctrl_offset`): apply the
   per-arm rigid offset `_CTRL_OFF[{L,R}]` (`p + R·p_off`, `axisangle(R·R_off)`) —
   GR1-robot geometry constants solved from the h5 on static frames (recon ≈1–2° /
   0.4–0.7 cm). **Critical**: the control frame is the SITE frame, ~offset from the
   body-eef quat.
4. **Grip**: `g = _grip_from_hand15(hand15) ∈[0,1]` (project onto OPEN15→CLOSED15),
   then `2g−1`; `idx12 ← Lhand15`, `idx13 ← Rhand15` (mirrors converter `GRIP_CMD`).
5. Assemble `env_action = [Rpos3, Raa3, Lpos3, Laa3, grip12, grip13]` (14-dim,
   block 0:6 = right, 6:12 = left, absolute base-frame, axis-angle). The client steps
   each of the 32 rows: `for a in env_action: env.step(a)`.

### 8.5 Launch
```bash
# server (cosmos venv)
python -m cosmos_framework.scripts.action_policy_server_gr1 \
  --experiment action_policy_gr1_posttrain_1x_nano --checkpoint <iter_dir> \
  --pool /mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool.npz --pool-kind taskdemo \
  --play-pool <play pool.npz> --pairs-dir /mnt/ddn/dataset/egoverse_pairs_gr1_1x \
  --vae-path <Wan2.2_VAE.pth> --port 8000 [--save-video-dir ...]
# client (mimicdroid-robocasa venv)
MUJOCO_GL=egl .venv/bin/python cosmos_framework/scripts/eval_gr1_client.py \
  --server http://127.0.0.1:8000 --task PnPSinkToCabinet --episodes 6 [--video-dir ...]
```
Optional `--save-video-dir` writes per-chunk `[obs | generated | retrieved-demo]`
panels + a `cond_ep_XXX.mp4` condition stream for debugging. Retrieval knobs
(`ZW=5.0`, `SHORTLIST=300`, per-task grip weight) MUST match the training builder —
see `gr1_inference_retrieval_spec.md` §7 checklist.

---

## 9. Related
* Retrieval contract & descriptor: [`gr1_inference_retrieval_spec.md`](./gr1_inference_retrieval_spec.md)
* action57/state57 layout: [`action57_state57_convention.md`](./action57_state57_convention.md)
* Sibling recipes: [`action_policy_libero_posttrain.md`](./action_policy_libero_posttrain.md),
  [`action_policy_droid_posttrain.md`](./action_policy_droid_posttrain.md)
* Memory: [[gr1-seen-eval-style-layout]], [[gr1-mimicdroid-online-eval]],
  [[gr1-future-pair-pipeline]]

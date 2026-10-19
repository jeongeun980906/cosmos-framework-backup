# GR1 Closed-Loop Eval Runbook

Example-first guide to **running** the GR1 one-shot policy eval (server + client rollout,
success rates). For the *why* behind the rules see
[`cosmos_policy/docs/gr1_delta_short_stack.md`](../../cosmos_policy/docs/gr1_delta_short_stack.md) §5
(bridge/floor/spawn rationale) and
[`gr1_data_eval_redesign.md`](gr1_data_eval_redesign.md) (seen/L2/L3 tier design).

---

## 0. Architecture (30-second version)

Two processes talking over HTTP (`/reset`, `/push`, `/predict`):

```
┌─ eval_gr1_server.py ─────────────┐        ┌─ eval_gr1_client.py ──────────┐
│ cosmos_policy .venv               │        │ mimicdroid-robocasa .venv      │
│ owns: GPU model + demo retrieval  │◀──HTTP─▶│ owns: robocasa GR1 env         │
│ /predict -> {action57, env_action}│        │ steps env, checks success      │
└───────────────────────────────────┘        └────────────────────────────────┘
   1 server per GPU (port 8700+g)                1 client per atom (task×demo)
```

- **Server**: `/mnt/ddn/cosmos_policy/cosmos_policy/scripts/eval_gr1_server.py` — loads the
  2.5 GR1 checkpoint, retrieves the one-shot demo, returns a 14-dim `env_action`.
- **Client**: `/mnt/ddn/cosmos-framework/cosmos_framework/scripts/eval_gr1_client.py` — runs
  the robocasa env, pins the one-shot demo(s), records video, prints success.
- **env_action** = `[Rpos3, Raa3, Lpos3, Laa3, gripL, gripR]`, OSC_POSE absolute base-frame,
  gripper +1 = close. Chunk = 32 steps; client pushes real per-step proprio back each chunk.

---

## 1. Quick start — one task on one GPU

Two terminals (or one server + backgrounded client). Uses GPU 0.

### 1a. Launch the server
```bash
cd /mnt/ddn/cosmos_policy
CK=/mnt/ddn/tmp/gr1_posttrain_v6_statedmid_448/cosmos_policy/cosmos_v2_finetune/\
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_dmid/checkpoints/model_000006000.pt

HF_HOME=/mnt/ddn/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 \
.venv/bin/python -m cosmos_policy.scripts.eval_gr1_server \
  --checkpoint "$CK" \
  --experiment cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_dmid \
  --pred-target abs_action57 --resolution 448 \
  --pinned-advance mono --pin-lock-ep --mono-floor 8 \
  --port 8700
```
Wait for the model to load (~2–3 min; the VAE + policy). Server is ready when port 8700 accepts
connections. (The math-SDPA fix is applied automatically — see §5.)

### 1b. Launch the client (single demo, 10 episodes)
```bash
cd /mnt/ddn/cosmos-framework
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=/mnt/ddn/cosmos-framework \
/mnt/ddn/mimicdroid-robocasa/.venv/bin/python cosmos_framework/scripts/eval_gr1_client.py \
  --server http://127.0.0.1:8700 \
  --oneshot-pool3 --task PnPSinkToCabinet --demo-only demo_1 \
  --episodes 10 --max-steps 400 \
  --video-3cam --video-dir /mnt/ddn/tmp/eval_videos
```

### 1c. Read the output
Per episode and final line (client.py:488-490):
```
[PnPSinkToCabinet] ep 3/10 init=taskdemo_..._demo_1 -> SUCCESS (12.4s)  running 2/3
...
GR1 PnPSinkToCabinet: 7/10 success (70%)
```
- `-> SUCCESS | fail` per episode, `running s/n` = cumulative.
- Final `GR1 <task>: s/N success (P%)`.
- **Reporting convention**: always `성공수/판정수` (✓ = 10 ep done, <10 = in progress).
- Videos land in `--video-dir` as `[center|sideleft|sideright]` strips.

> `MUJOCO_EGL_DEVICE_ID` matches `CUDA_VISIBLE_DEVICES`. In the pod each worker restricts
> `CUDA_VISIBLE_DEVICES=$g`, so EGL auto-picks the only visible device (id not needed there).

---

## 1.5 Two run modes — ICL (demo-conditioned) vs BC (plain)

The same server/client pair runs both modes; **the experiment name decides**, and the client
must match. Mixing them (plain ckpt + `--oneshot-pool3`, or ICL ckpt without it) silently
produces garbage numbers, not an error.

| | **ICL (one-shot / conditioned)** | **BC (plain)** |
|---|---|---|
| experiment | no `_plain`, e.g. `..._gr1_ret_v6_448_state` | contains `_plain`, e.g. `..._gr1_plain_state_v6` |
| server sample | demo-conditioned layout, retrieval + pinned demo pointer | `_build_sample_plain`: `[cur_state, cur_image, ACTION, future×8]`, `num_cond=2`, all ret idx = −1, `wm_mask=0`, `demo=None` |
| server flags | `--pinned-advance mono --pin-lock-ep --mono-floor 8` meaningful | pointer flags ignored (no demo pointer) |
| client | **must pass `--oneshot-pool3`** | **must OMIT `--oneshot-pool3`** |
| task rules (§4) | bridge / floor / spawn all apply | same — bridge is client-side (uses taskdemo `actions[0]`), so `--approach-demo-t0` etc. still apply |
| known post-train runs | `gr1_posttrain_v6_448_state` (base), `_state_frombv2v10`, `_state_frombv2`, `_state_fromsmid` | `gr1_posttrain_v6_448_plainbc_base`, `_plainbc_frombv2v10` (BC mid), `_plainbc_fromtransfer` (RET mid) |

```bash
# ICL atom (conditioned) — note --oneshot-pool3
... eval_gr1_server ... --experiment cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state ...
... eval_gr1_client.py --server http://127.0.0.1:$PORT --oneshot-pool3 \
    --task PnPSinkToCabinet --demo-only demo_1 --episodes 10 --max-steps 400 ...

# BC (plain) atom — same task flags, NO --oneshot-pool3
... eval_gr1_server ... --experiment cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6 ...
... eval_gr1_client.py --server http://127.0.0.1:$PORT \
    --task PnPSinkToCabinet --demo-only demo_1 --episodes 10 --max-steps 400 ...
```

Checkpoints are DCP; export once per iter before serving:
```bash
cd /mnt/ddn/cosmos_policy
HF_HOME=/mnt/ddn/hf_cache .venv/bin/python - "$CKROOT/iter_000006000/model" out.pt <<'PY'
import sys
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
dcp_to_torch_save(sys.argv[1], sys.argv[2])
PY
```

---

## 1.6 Video-only conditioning eval (`eval_gr1_server_vidonly`)

For checkpoints trained with **`no_ret_action=True`** (experiment
`..._gr1_ret_v6_448_state_vidonly` — the retrieved demo conditions with VIDEO ONLY, the
RET_ACTION latent slot is dropped). Serving a vidonly ckpt through the normal server (or a
normal ckpt through this one) silently mis-lays the sample — wrong numbers, no error.

**Server** — swap ONLY the module name; every other flag identical to §1a. The script
auto-injects `--experiment ..._v6_448_state_vidonly` and `--pairs-dir
/mnt/ddn/dataset/egoverse_pairs_gr1_v6` when omitted:
```bash
cd /mnt/ddn/cosmos_policy
HF_HOME=/mnt/ddn/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 \
.venv/bin/python -m cosmos_policy.scripts.eval_gr1_server_vidonly \
  --checkpoint <vidonly_ckpt.pt> --pred-target abs_action57 --resolution 448 \
  --pinned-advance mono --mono-floor 8 \
  --port 8700
```

**Client** — completely unchanged (`--oneshot-pool3` + the §4 task rules). It cannot tell
which server it is talking to.

What the subclass changes (single override, `GR1PolicyVidOnly._build_sample_v2`):
o=0 layout — `state_t` 20→**19**, `num_conditional_frames` 11→**10**, pixel pack 77→**73f**
(`[state 1f | demo 32f | cur 4f | act 4f | future 32f]`), `retrieved_action_latent_idx=-1`.
Retrieval, pinned-demo pointer, HTTP API, decode: inherited from the base server verbatim.

**Execution pitfalls (2026-07-31 campaign):**
- **First-predict warmup crash**: launching a client the moment the port opens can kill it
  silently at `ep0 t=0` (0/0-episode log) when two servers warm up at once. Sleep ~30 s
  between `port_up` and the first client; re-run any atom whose log has 0 judged episodes.
- **NUL-contaminated client logs**: the video writer can leak binary bytes into the client
  log, after which plain `grep` treats it as binary and returns nothing. Count with
  `tr -d '\0' < log | grep -ac ' -> SUCCESS'` and cross-check the client's own
  `running s/n` counter (last ` -> ` line) — the counter is authoritative.
- Ckpt must be the converted `.pt` (§1.5 DCP export), not the DCP dir.

Reference numbers (vidcond 8K full, mono no-pinlock, 10 ep): L1 100/120 (Cabinet 20/30 ·
Plate 20/30 — the only no-seesaw ckpt), CabinetL2 17/30, PlateL2 16/40, **Door L2/L3 all 0**
(demo-action-required transfers die without the RET_ACTION slot), FaucetL3 19/30.

---

## 2. Full 8-GPU pod eval (all tasks × demos)

The `cosmos-gr1-fulleval-*.yaml` PyTorchJobs run the whole matrix on one 8-GPU box.

### 2a. Launch
```bash
kubectl apply -f /mnt/ddn/kube_yaml/cosmos-gr1-fulleval-v6base-8gpu.yaml
```

### 2b. How it works
- **1 PyTorchJob, 1 Worker, 8 GPUs** (single node; "distributed" = 8 local servers).
- **Atom = (task × demo)** → 10 episodes. e.g. queue line
  `TurnOnFaucet --demo-only demo_1 --episodes 10`. The v6base queue = **37 atoms / 12 tasks**
  (seen tasks have 3 demos, Plate-L2 has 4).
- **flock queue**: `queue2.txt`; each worker pops the top line atomically
  (`flock 9; head -n1; sed -i '1d'`).
- **8 workers** (GPU g = 0..7): worker g starts its own server on `8700+g`
  (`MASTER_PORT=29600+g`), waits for the port, then loops popping atoms and running the client
  against `127.0.0.1:8700+g`.
- **Logs** under `$OUT=/mnt/ddn/tmp/gr1_v6base_8k_fulleval/logs`:
  `task2_<task>_<demo>.log`, `server2_gpu$g.log`, `driver2.log`; summary → `summary2.txt`,
  `FULLEVAL2_DONE.marker`. Videos → `$OUT/videos_<task>`.

### 2c. Watch progress (per-atom success)
```bash
OUT=/mnt/ddn/tmp/gr1_v6base_8k_fulleval/logs
for f in $OUT/task2_*.log; do
  s=$(grep -cE ' -> SUCCESS' "$f"); j=$(grep -cE ' -> (SUCCESS|fail)' "$f")
  mk=$([ "$j" -ge 10 ] && echo ✓); printf '%-40s %s/%s %s\n' "$(basename "$f" .log)" "$s" "$j" "$mk"
done
```

### 2d. Split L2/L3 into separate pods (faster)
L2/L3 atoms are slower — the `*-l2-8gpu.yaml` / `*-l3-8gpu.yaml` variants queue only that tier.
Same port/flock/EGL pattern; only the queued atoms + checkpoint differ. Launch both in
parallel for ~2× wall-clock.

---

## 3. Command reference

### Server flags (the ones you actually pass)
| Flag | v6 value | Meaning |
|---|---|---|
| `--checkpoint` | (req) | 2.5 GR1 `.pt` (DCP→pt export) |
| `--experiment` | `..._v6_448_state_dmid` | drives layout auto-detect (`_state`, `_short`, `_plain`, `v2`) |
| `--pred-target` | `abs_action57` | prediction space → env action. **Must match the checkpoint's target.** |
| `--resolution` | `448` | image size S. **v6 = 448** (default 224 is wrong for v6) |
| `--pinned-advance` | `mono` | one-shot demo pointer policy (`match`/`lockstep`/`mono`) |
| `--pin-lock-ep` | (set) | lock the pinned demo to its episode after first re-match |
| `--mono-floor` | `8` | min advance per re-match; band `[last+floor, last+48]`. 8=slow-robot, 16=fast tasks |
| `--port` | `8700+g` | HTTP port |
| `--shift` | (unset) | leave unset → reads `model.config.shift` (do NOT hardcode 3.0) |
| `--grid4` / `--plate-state-delta` | off | 4-view grid / hybrid-model plate decode |

### Client flags (per atom)
| Flag | Meaning |
|---|---|
| `--server http://127.0.0.1:PORT` | server URL |
| `--task <Name>` | task (see §4) |
| `--oneshot-pool3` | v6 one-shot: pin the task's 3 taskdemos, server matches across them |
| `--demo-only demo_N` | run only that demo (split a task across workers) |
| `--episodes 10` | episodes per atom |
| `--max-steps 400` | env steps per episode |
| `--video-3cam --video-dir DIR` | 3-cam strip mp4 per episode |
| `--approach-demo-t0` | **bridge**: move eef to demo `actions[0]` before rollout (see §4) |
| `--approach-min-steps N` | skip bridge if it needs < N rows (20 default; 0 = always) |
| `--mono-floor 16` | per-episode floor override sent in `/reset` (FaucetL3) |
| `--raw-spawn` | skip the PnPSinkToCabinet* default anchor/jitter |
| `--spawn-offset-x/y-cm`, `--spawn-jitter-cm` | deterministic spawn shift / jitter |

There is **no `--pred-target`/`--resolution` on the client** (server-only).

---

## 4. Tasks & rules (with example commands)

### Task list / tiers
- **seen**: `TurnOnFaucet`, `PnPSinkToRightCounterPlate`, `PnPSinkToCabinet`, `CloseLeftCabinetDoor`
- **L2** (held-out style/objects, same layout): `PnPSinkToRightCounterPlateL2`, `PnPSinkToCabinetL2`,
  `CloseLeftCabinetDoorL2`, `CloseRightCabinetDoorL2`
- **L3** (held-out layout): `TurnOnFaucetL3`, `PnPSinkToRightCounterPlateL3`, `CloseLeftCabinetDoorL3`,
  `PnPSinkToMicrowaveTopL3` (the fixed-fixture PnP L3 analog; PnPSinkToCabinet has no L3 env class)

### Rule 1 — Bridge tasks (`--approach-demo-t0`)
Bridge = `CloseLeft/RightCabinetDoorL2/L3`, `TurnOnFaucetL3`, `PnPSinkToMicrowaveTopL3`.
FaucetL3 + Microwave also force it with `--approach-min-steps 0`. PnP-L2 = **no** bridge (raw spawn).
```bash
# CloseLeftCabinetDoorL3 (bridge)
... eval_gr1_client.py --server http://127.0.0.1:8700 --oneshot-pool3 \
    --task CloseLeftCabinetDoorL3 --demo-only demo_1 --episodes 10 --max-steps 400 \
    --approach-demo-t0 --video-3cam --video-dir $OUT/videos_CloseLeftCabinetDoorL3
```

### Rule 2 — Floor (`--mono-floor`)
`TurnOnFaucetL3` uses **16**; everything else the server default **8**.
```bash
# TurnOnFaucetL3 (bridge + floor 16 + force bridge)
... --task TurnOnFaucetL3 --oneshot-pool3 --demo-only demo_2 --episodes 10 \
    --approach-demo-t0 --approach-min-steps 0 --mono-floor 16 ...
```

### Rule 3 — Spawn / anchor
- `PnPSinkToCabinet*`: default anchor `x+2cm (RIGHT), y+3cm (FORWARD), jitter 1cm, object=vegetable`
  (matches training density F+3/R+2±1). Disable with `--raw-spawn`.
- **CabinetL2 is f3r2 too** (user 2026-07-28): do NOT pass `--raw-spawn` on `PnPSinkToCabinetL2` —
  the client default anchor is the protocol. Historical caveat: the 2026-07-27 tmid8k pod full
  eval and same-day local plain runs measured CabinetL2 **with `--raw-spawn`** (fixed demo spawn);
  those numbers are not comparable to f3r2±1cm ones.
- `PnPSinkToMicrowaveTopL3`: **no anchor** (raw demo spawn + bridge) — anchor flags were never
  used in any eval yaml; the Cabinet-only client default does not apply to it.
- `PnPSinkToRightCounterPlate*`: **raw spawn** (corn rolls off the container edge if jittered).
```bash
# Plate L3 — raw spawn (no anchor/jitter)
... --task PnPSinkToRightCounterPlateL3 --oneshot-pool3 --demo-only demo_1 \
    --episodes 10 --raw-spawn ...
```

### Rule 4 — Seen style+layout
Each *seen* task is pinned to the single canonical (layout, style) frozen in its taskdemo
ep_meta (not a code constant). L2 = same layout + held-out style/objects; L3 = held-out layout.
Seen-tier single-demo sweep: `--oneshot-demo taskdemo_<task>_<task>_D_demo_N` (manifest mode).

---

## 5. Troubleshooting (symptom → diagnose → fix)

### A. `/predict` returns 500 "No available kernel" (SDPA)
- **Cause**: video-model VAE attention has `head_dim>128` → flash/mem-eff/cuDNN SDPA disabled;
  if math is also off, no kernel exists.
- **Fix**: already handled in the server (force-enables math SDPA, disables the others,
  right after arg-parse). If you see it, you're on an **old server** — pull the current
  `eval_gr1_server.py`.

### B. Server hangs at boot — GPU shows 0 MiB, log empty
- **Cause**: CUDA context corruption from repeated hard-kills on that GPU.
- **Fix**:
  ```bash
  pkill -9 -f eval_gr1_server; sleep 12
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # confirm 0 MiB
  # then relaunch the server
  ```

### C. Worker prints "SERVER DIED", tails a truncated log
- **Cause**: server crashed mid-run — usually **OOM** or an intermittent MuJoCo/CUDA kernel
  crash, NOT the thing the traceback first blames.
- **Fix**: read `server2_gpu$g.log` for the real error. `backoffLimit:0` / `restartPolicy:Never`
  → no auto-respawn; re-queue that atom (put its line back in `queue2.txt`) or restart the pod.

### D. "NCCL hang" / job stuck early
- **Reality**: on this stack a "NCCL hang" is almost always **data FileNotFound** or a
  **worker OOM-death**, not a SHARP/NCCL issue. Don't chase NCCL — grep the worker/server log
  for `FileNotFound`, `CUDA out of memory`, missing pool/manifest paths.

### E. Client can't connect (`Connection refused`)
- **Cause**: server still loading (VAE ~2–3 min) or died at boot.
- **Fix**: the pod's `port_up` probe waits up to 20 min; manually, poll
  `(exec 3<>/dev/tcp/127.0.0.1/8700) 2>/dev/null && echo up`. If it never comes up, check the
  server log for `Loaded`/model-on-GPU and GPU memory usage.

### F. Background launch "failed (exit 1)" but GPU still busy
- **Cause**: the wrapper shell died but the **disowned server survives as an orphan** holding
  the GPU.
- **Fix**: `pkill -9 -f "eval_gr1_server.*<ckpt-substring>"`, verify GPU 0/0, relaunch cleanly.
  Prefer clean single-checkpoint launches over overlapping sweeps.

### G. Everything runs but success is 0% / weird
- **Checklist**:
  - `--pred-target` matches the checkpoint (v6 state models = `abs_action57`).
  - `--resolution 448` for v6 (224 silently mis-scales).
  - `--shift` left UNSET (hardcoding 3.0 is a train/eval mismatch).
  - Bridge/floor/spawn flags match the task's rule (§4) — a missing `--approach-demo-t0` on a
    bridge task tanks it; a stray anchor on Plate rolls the object off.
  - `--experiment` matches the checkpoint's training layout (short vs video, state vs delta).

### H. Intermittent EGL/MuJoCo kernel crash on 8-GPU
- **Cause**: EGL device contention / driver hiccup.
- **Fix**: ensure each worker has `CUDA_VISIBLE_DEVICES=$g` (server + client both see only GPU g);
  `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` exported. Re-queue the crashed atom.

---

## 6. Cheat-sheet

```bash
# one server (GPU g), v6 video model
CUDA_VISIBLE_DEVICES=g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
MASTER_PORT=$((29600+g)) .venv/bin/python -m cosmos_policy.scripts.eval_gr1_server \
  --checkpoint $CK --experiment $EXP --pred-target abs_action57 --resolution 448 \
  --pinned-advance mono --pin-lock-ep --mono-floor 8 --port $((8700+g))

# one client atom (add --approach-demo-t0 for bridge tasks; --mono-floor 16 for FaucetL3;
#                  --raw-spawn for Plate*)
MUJOCO_GL=egl PYTHONPATH=/mnt/ddn/cosmos-framework \
/mnt/ddn/mimicdroid-robocasa/.venv/bin/python cosmos_framework/scripts/eval_gr1_client.py \
  --server http://127.0.0.1:$((8700+g)) --oneshot-pool3 --task $TASK --demo-only demo_$N \
  --episodes 10 --max-steps 400 --video-3cam --video-dir $OUT/videos_$TASK

# full 8-GPU matrix
kubectl apply -f /mnt/ddn/kube_yaml/cosmos-gr1-fulleval-v6base-8gpu.yaml
```

**Related docs**: `cosmos_policy/docs/gr1_delta_short_stack.md` (§5 rule rationale, §7 ops),
`gr1_data_eval_redesign.md` (tier design), `gr1_inference_retrieval_spec.md` (retrieval contract),
`gr1_v6_data_law.md` (t0/t>0 retrieval law).

# EgoVerse `action57` / `state57` convention — layout, scripts, transformation math

This documents the unified 57-D action/state representation used across the EgoVerse
conditioning-policy data: **which scripts build/convert it**, and **how the transformation
is done** (with the exact equations). Applies to human rigs (aria / MECKA / scale) and robot
rigs (mimicdroid GR1 play, GR1 task demos, robocasa TaskDemos). Cosmos 2.5 (`cosmos_policy`)
wraps this as a 59-D union — see §6.

---

## 1. The canonical 57-D layout

```
head9(0:9) | Leef9(9:18) | Lhand15(18:33) | Reef9(33:42) | Rhand15(42:57)
```

- Each `eef9` / `head9` block = `[pos3 (or Δpos3), 6Drot6]`, where the **6D rotation** = the first
  two columns of the rotation matrix `[R[:,0], R[:,1]]` (Zhou et al. continuity rep; 3rd column
  recovered by cross product).
- Each `hand15` = **5 fingertips × 3 coords**, MANO tip indices `TIPS=[4,8,12,16,20]`
  (thumb, index, middle, ring, pinky), in the **wrist/eef-local frame**.
- **`state57`** = ABSOLUTE pose per frame (length `n`). **`action57`** = per-step DELTAS `t→t+1`
  (length `n-1`).
- **Units: meters.** Human reference frame = **waist@frame0** (fixed per episode: 0.5 m below
  the first-frame head, yaw-only gravity-aligned heading — see §3e); robot frame =
  base-relative eef. (Round-trip test scales pos error ×100 to print cm.)

Layout appears verbatim in `convert_taskdemos.py`, `10_mimicdroid_convert.py`,
`cosmos_policy/.../egoverse_ret_dataset.py`, `egoverse_pair_builder/recovered/eg_action57.py`.

---

## 2. Scripts that PRODUCE / CONVERT action57 & state57

### Human rigs (aria / MECKA / scale — real MANO fingertips, real head motion)
Run under **`/mnt/ddn/EgoVerse/emimic/bin/python`** (egomimic venv, py3.11).
| Script | Role |
|---|---|
| `EgoVerse/egomimic/scripts/data_download/process_egoverse.py` | UPSTREAM (superseded): raw obs → head-frame `*_transformed`. REPLACED in-place by `egoverse_pair_builder/recovered/eg_convert*.py` with the **waist@frame0** transform (§3e); episodes carry `waistframe_transformed=True`. |
| `egoverse_pair_builder/recovered/eg_state57.py` | ABSOLUTE `state57` from `*_transformed` obs → `state57/{state,valid}` zarr (Pool(12) over `/mnt/ddn/dataset/egoverse`). |
| `egoverse_pair_builder/recovered/eg_action57.py` (+ `_v2`, `_par`, `_fix`, `_fix143`) | DELTA `action57` from the same obs → `action57/{action,valid}`. |
| `cosmos-framework/tools/fix_aria_eef_wrist.py` | In-place CORRECTOR: re-bases aria `eef9` rotation `ee_pose`→`wrist_pose`, recomputes Δpos in the new frame (aria ee/wrist quats differ). Idempotent, default dry-run. |

### Robot rigs (fixed head, synthetic `hand15` from gripper signal)
Run under **`/mnt/ddn/EgoVerse/emimic/bin/python`** (pair-builder) or **cosmos-framework `.venv`** (tools).
| Script | Role |
|---|---|
| `egoverse_pair_builder/10_mimicdroid_convert.py` | Reference robot converter. action57/state57 from robosuite `robot0_base_to_{left,right}_eef_{pos,quat}` + grip. Defines `HEAD9/IDENT9/OPEN15/CLOSED15` + `Rmat/eef9/grip_cmd/hand15`. |
| `egoverse_pair_builder/73_gr1_to_egoverse57.py` | GR1 task demos → `egoverse_gr1/<task>__<demo>` (imports 10's helpers). |
| `cosmos-framework/tools/convert_taskdemos.py` | robocasa TaskDemos → `egoverse/taskdemo_*` (online-eval demo pool). Self-contained `_eef9/_grip/_hand15/_abs9` + constants. |
| `cosmos-framework/tools/convert_taskdemos_2x.py` | 2×-speed: raw obs `::2` subsample, **re-derive** deltas on subsampled poses (imports `convert_taskdemos` via importlib). |
| `egoverse_pair_builder/58_upgrade57_1x.py` | Upgrades legacy `action20/state20` zarrs → action57/state57 in place (20→57 via HEAD9/IDENT9 + hand15 lerp), deletes 20-D groups. |

### Runtime (produces state57 live at eval, not a file writer)
- `cosmos-framework/cosmos_framework/scripts/action_policy_server_gr1.py` — `_state57_from_proprio`
  (abs state57 from live env proprio), `_integrate_arm` (inverts action57 deltas → absolute). cosmos-framework `.venv`.

---

## 3. Transformation math

### 3a. `eef9` / `head9` delta (action57) — body-frame delta
`_eef9` (`convert_taskdemos.py`, `10_mimicdroid_convert.py`, `recovered/eg_action57.py`):
```
Rt = R(quat_t),  Rn = R(quat_{t+1})
Δpos = Rtᵀ · (pos_{t+1} − pos_t)            # position delta in the frame at t
ΔR   = Rtᵀ · R_{t+1}                          # relative rotation, current-frame local
eef9 = [ Δpos(3), ΔR[:,0](3), ΔR[:,1](3) ]   # 6D rot = first two cols of ΔR
```
i.e. the 6D rotation delta is `R_t⁻¹ @ R_{t+1}` (`R_t⁻¹ = R_tᵀ`), stored as its first two columns.

### 3b. `abs9` (state57) — absolute pose
```
abs9 = [ pos(3), R(q)[:,0](3), R(q)[:,1](3) ]   # world/base absolute pose, 6D rot
```

### 3c. `hand15` — fingertips in wrist-local frame
Human:
```
tips  = keypoints[:, TIPS, :]                 # (·,5,3), MANO tips 4,8,12,16,20
rel   = R_wristᵀ · (tips − wrist_pos)         # 5 tips in wrist local frame
hand15 = rel.reshape(15)                       # meters
```
state57 → tips at `t` in wrist(`t`) frame; action57 → tips at `t+1` in eef(`t+1`) frame.

Robot (no real fingers → synthetic lerp by gripper signal `g∈[0,1]`):
```
g = (clip(action[:, grip_dim], -1, 1) + 1)/2   # grip_dim 12=L, 13=R
hand15(g) = (1−g)·OPEN15 + g·CLOSED15          # OPEN15/CLOSED15 hardcoded
```

### 3d. `head9` — human vs robot
- **Human** (real head motion): state57 head9 = `abs9(obs_head_pose_transformed)`; action57 head9 = `delta9(head_pose)`.
- **Robot** (fixed camera): state57 head9 = constant `HEAD9 = [-0.60,0,1.17, 0.033,-0.999,-0.003, 0.818,0.028,-0.574]`; action57 head9 = constant `IDENT9 = [0,0,0, 1,0,0, 0,1,0]` (zero-motion identity delta).

### 3e. Human reference frame = WAIST@frame0 (fixed per episode)
`egoverse_pair_builder/recovered/eg_convert.py` (replaces the older per-frame head-frame
transform from `process_egoverse.py` IN-PLACE; episodes are stamped
`waistframe_transformed=True`):
```
pos0 = head_pose[0].pos − 0.5·UP              # waist proxy: 50 cm below the first-frame head
R0   = [heading, UP×heading, UP]              # yaw-only, gravity-aligned;
                                              # heading = mean of the first 15 head headings
pose_waist = [R0ᵀ·(p − pos0), R0ᵀ·R_pose]     # FIXED transform for the whole episode
kp_waist   = R0ᵀ·(kp − pos0)
```
So the `*_transformed` obs consumed by `eg_state57`/`eg_action57` live in a single
episode-constant, gravity-aligned frame anchored below the initial head — NOT a moving head
frame. Walking episodes therefore have legitimate multi-meter position spans.

### 3f. Server-side inverse (integration) — the exact round-trip
`action_policy_server_gr1.py`:
- `_sixd_to_R(c0,c1)` — Gram-Schmidt: `a=c0/|c0|; b=normalize(c1−(a·c1)a); R=[a,b,a×b]`.
- `_integrate_arm(p0,R0,eef9)`: per step `p ← p + R·Δpos ; R ← R·ΔR` → absolute (exact inverse of §3a).
- `_grip_from_hand15(h15)`: inverts the lerp `g = clip(((h15−OPEN15)·axis)/(axis·axis), 0,1)`, `axis=CLOSED15−OPEN15`.

---

## 4. Reconstruction / round-trip verification
- **`cosmos-framework/tools/test_gr1_action57_roundtrip.py`** — authoritative proof (run under `.venv`,
  imports the real server helpers). Checks on real GR1 zarrs:
  - (A) `state57[0]` + action57 deltas → `_integrate_arm` must reproduce `state57[1:]` (PASS: `pos<0.1cm, rot<0.5°`).
  - (B) hand15→g projection self-consistency.
  - (C) grounds in raw h5: recompute abs9 from source `base_to_*_eef_{pos,quat}`, confirm == zarr state57;
    h5→eef9→integrate reproduces the h5 absolute eef. Prints `WORST … pos …cm | rot …° | grip …` + `ROUND-TRIP PASS/FAIL` (the "0.000cm / 0.028°"-style log).
- `tools/fix_aria_eef_wrist.py` — inline `recon_rot_err` self-check per side (dry-run before `--apply`).
- `egoverse_pair_builder/recovered/eg_verify.py` / `eg_verify2.py` — coarser dataset-level checks.

---

## 5. Quick equation reference
```
abs9(pos,q)  = [pos, R(q)[:,0], R(q)[:,1]]                                  # state57 pose block
eef9(t)      = [ R_tᵀ(pos_{t+1}−pos_t), (R_tᵀR_{t+1})[:,0], (R_tᵀR_{t+1})[:,1] ]  # action57 delta block
hand15       = ( R_wristᵀ (fingertips − wrist_pos) ).flatten()             # human, wrist-local, meters
hand15_robot = (1−g)·OPEN15 + g·CLOSED15,   g=(clip(a[dim],−1,1)+1)/2       # dim 12=L, 13=R
head9        = abs9(head) | delta9(head)  [human]   vs   HEAD9 | IDENT9 constants [robot]
integrate    = p←p+R·Δpos ; R←R·(6D→R)                                     # server inverse, exact round-trip
```

---

## 6. `action59` (Cosmos 2.5 / `cosmos_policy`) vs `action57` (Cosmos 3 / `cosmos-framework`)

`action59` is **not a richer action** — it is a **masked union space** superimposing human `action57`
and eva-robot `action20` so both share one 59-D vector + a per-sample mask. Defined in
`cosmos_policy/cosmos_policy/datasets/egoverse_ret_dataset.py`:
```
SL_HEAD 0:9 | SL_LEEF 9:18 | SL_LHAND 18:33 | SL_LGRIP 33:34 |
SL_REEF 34:43 | SL_RHAND 43:58 | SL_RGRIP 58:59
```
The +2 dims over 57 are the two scalar gripper dims `SL_LGRIP(33)` and `SL_RGRIP(58)` for the eva
`action20 = [Leef9|Lgrip1|Reef9|Rgrip1]`. This **shifts Reef/Rhand by +1** vs the 57 layout.

Packing (`_action59`): human `action57` → scatter into `HEAD,LEEF,LHAND,REEF,RHAND` (grips 0, masked out);
eva `action20` → scatter into `LEEF,LGRIP,REEF,RGRIP` (head/hands masked). **No 57↔59 numeric conversion** —
masked scatter/gather into a shared union. Stats: `action59_stats.json = {"lo":[59], "hi":[59]}` (per-dim
1st/99th percentile). Cosmos 3 drops the union and uses native `action57` directly (`egoverse_zarr_dataset.py`,
server uses 57).

**Update (2026-07-01): the cosmos_policy egoverse-ret path now also uses native `action57` (`ACTION_DIM=57`).**
Since the v9 pairing (`egoverse_pairs_v9`) is human-only (queries + matches are aria/scale/MECKA, never eva),
the eva grippers `Lgrip`/`Rgrip` are always dead → dropped. `_action57` returns raw human action57 (mask all 1);
`action_dim=57` in the experiment config; stats file is `action57_stats.json`. Because `use_action_projection=False`
(action is tiled, not projected), 59→57 has no base-checkpoint weight impact. eva (`action20`) if ever loaded
scatters `Leef9`/`Reef9` only (grips masked). See `cosmos_policy/cosmos_policy/datasets/egoverse_ret_dataset.py`.

---

## 7. Python env per script (wrong env ⇒ faiss/zarr/transformers import fails)
| Scripts / task | Env | Notes |
|---|---|---|
| `cosmos-framework/tools/*`, `cosmos_framework/scripts/*` (convert_taskdemos, roundtrip test, action_policy_server, eval_egoverse_generate) | `/mnt/ddn/cosmos-framework/.venv/bin/python` | py3.13; Cosmos 3 |
| `cosmos_policy/cosmos_policy/**` (egoverse-ret dataset, train, `eval_egoverse_ret`, `save_egoverse_t5_text_embeddings`) | `/mnt/ddn/cosmos_policy/.venv/bin/python` | py3.10; Cosmos 2.5. On kube: per-pod `uv sync --extra cu128 --group pusht` |
| `egoverse_pair_builder/*` (matching build 70–74, `recovered/eg_*.py`, `10_mimicdroid_convert`, `73_gr1_to_egoverse57`), `process_egoverse.py` | `/mnt/ddn/EgoVerse/emimic/bin/python` | py3.11; egomimic venv, has `faiss` + numcodecs zarr reader |

Run prefixes (copy-paste):
```
# matching build / action57 producers (egoverse_pair_builder)
cd /mnt/ddn/dataset/egoverse_pair_builder && /mnt/ddn/EgoVerse/emimic/bin/python 70_desc_v9.py --init
# cosmos-framework tools (convert / roundtrip / offline-generate eval)
cd /mnt/ddn/cosmos-framework && .venv/bin/python tools/test_gr1_action57_roundtrip.py
# cosmos_policy (2.5) t5 build / dataset / eval — needs GPU + HF cache offline
cd /mnt/ddn/cosmos_policy && CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=/mnt/ddn/cosmos_policy \
  .venv/bin/python -m cosmos_policy.datasets.save_egoverse_t5_text_embeddings --out_dir /mnt/ddn/dataset/egoverse_pairs_v9
```


## abs_action57 (v6, 2026-07-14)
- 57 레이아웃 동일: head9(HEAD9 상수) | Leef9 | Lhand15 | Reef9 | Rhand15
- **원천 = 명령**: env actions (site frame 절대 pose 타깃) — pos + 회전행렬 앞 2열(6D), hand15 = 명령 grip의 OPEN/CLOSED lerp
- **타이밍: row t = act[t]** (act[t]가 s[t+1]을 만듦) — n−1 행
- 용도: GR1 post-train 예측 타깃 (state-chunk 대체). 실행 = 예측을 env 액션으로 직행 (site frame, 변환/오프셋 불요)
- stats: min/max+5% 마진 전 차원 (`abs_action57_stats_pctl.json` — 이름과 달리 percentile 아님 주의)
- 참고: action57(기존)은 실측 delta, state57 회전은 body frame (site 대비 ~180°) — 매칭·조건은 state57, 타깃·실행은 abs_action57로 분리

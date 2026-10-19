# MimicDroid (GR1 bimanual robocasa play) → egoverse conversion

How the MimicDroid play demos are converted into the egoverse-style `action57`/`state57`
zarr format, the transformation math, the I/O contract, and the python env. Companion to
`action57_state57_convention.md` (the 57-D layout/math) and `egoverse_pairing_v9_log.md`
(matching). All scripts below are in `/mnt/ddn/dataset/egoverse_pair_builder/`.

**Python env: ALL MimicDroid scripts run under `/mnt/ddn/EgoVerse/emimic/bin/python` (py3.11).**
They import numpy/h5py/zarr/numcodecs/PIL/scipy; poses come straight from the baked h5 (NO
robosuite). `faiss` is used only in the build/match stage. Exception: `12_mimicdroid_match.py`
does `sys.path.insert(0,"/mnt/ddn/cosmos-framework")` and loads torch + T5-11b (GPU) for its
viz-only retrieval. No `.venv` interpreter is used by any MD script.

> **MD in v8 AND (as of 2026-07-01) v9.** MimicDroid rows go into `egoverse_pairs_v8` (via
> `62_combine_v8.py`). The v9 builder `73` is human-only ("NO-MD"), but **`73b_add_v8_md.py`** now
> appends the v8 MD windows+pairs AS-IS onto the v9 TRAIN manifest (`egoverse_pairs_v9`). This is
> valid with NO reindexing because the v8 eg-window block `[0:Neg]` is byte-identical to the v9
> windows table (Neg=1046962; verified ep/w0 match). The 20 MD play episodes live under
> `/mnt/ddn/dataset/egoverse/*SinkEnvForPlay*demo*` (moved there for GR1) with `action57`, so both
> the cosmos-framework and cosmos_policy readers load them via the human root (`isev=False` →
> `action57`). The v9 TEST split (`egoverse_pairs_v9_test`) stays human-only. `egoverse_mimicdroid/`
> is now EMPTY — the raw MD zarrs live under `egoverse/`.

---

## Pipeline (raw h5 → zarr → descriptor → pairs → manifest)

| Stage | Script | Role |
|---|---|---|
| convert (1×) | `10_mimicdroid_convert.py` | raw `demo_im128_notp.hdf5` + `subgoal.json` → per-demo zarr v3 at `egoverse_mimicdroid/{dir}__{demokey}` (front JPEG + **action57/state57**, 20 Hz) |
| 2× data | `MimicDroidDataset/bake_training2_2x.py` | bake a 2×-speed h5 copy `training/`→`training2/` (subsample + action resample); then re-run `10_ --src …/training2 --dst egoverse_mimicdroid_2x` |
| 20→57 upgrade | `58_upgrade57_1x.py` | one-time in-place migration of already-converted episodes from legacy `action20/state20` → `action57/state57`; deletes the 20-D groups |
| grip / subgoal patch | `17_patch_grip.py`, `18_patch_subgoals.py` | post-hoc fixes to the grip channel / `subgoals` attr (diagnostic-era) |
| descriptor | `11_mimicdroid_desc.py` | sliding-window motion descriptor over MD zarr → `egoverse_mimicdroid/desc_mimicdroid.npz` |
| MD pairs | `44_v8_md12.py` → `46_v8_samples.py` | 12 MD-internal matches per MD subgoal window → `egoverse_mimicdroid/v8_md_pairs.npz` |
| match/2× viz | `12_mimicdroid_match.py`, `56_md12_2x.py` | render-only retrieval / 1×+2× md12 clips |
| manifest | `62_combine_v8.py` | merge human rows (`v8_human_pairs.npz`) + MD rows (`v8_md_pairs.npz`) → `egoverse_pairs_v8/{windows,pairs}.npz` |

---

## `10_mimicdroid_convert.py` — the converter

`SRC=/mnt/ddn/dataset/MimicDroidDataset/training`, `DST=/mnt/ddn/dataset/egoverse_mimicdroid`,
`FPS=20`, `IMG_KEY="robot0_agentview_center_image"`, `GRIP_CMD=(12,13)` (action dim 12=L grip,
13=R grip cmd), `JPEG_Q=92`.

**Reads** per demo from `data/{key}/obs`: `robot0_base_to_{left,right}_eef_{pos(n,3),quat(n,4 xyzw)}`
and `data/{key}/actions (n,14)`; the (128,128,3) `agentview_center` images.

> ⚠️ **The docstring is STALE** — it says "action20d (Leef9|Lgrip1|Reef9|Rgrip1)", but the current
> code (Jun 26) writes native **action57** (`head9|Leef9|Lhand15|Reef9|Rhand15`).

### Math
- **grip** `g = (clip(actions[:,dim],-1,1)+1)/2 ∈ [0,1]` (0 open, 1 closed); `Lg=g(·,12)`, `Rg=g(·,13)`.
- **eef9 delta** (base-relative, per hand, length n−1), `Rm`=quat→matrix, `Rtᵀ=Rm[:-1]ᵀ`:
  `dpos = Rtᵀ·(pos_{t+1}−pos_t)`, `dR = Rtᵀ·Rm_{t+1}`, `eef9 = [dpos3 | dR[:,0] | dR[:,1]]`.
- **abs9** (state) `abs9(pos,q) = [pos | Rm[:,0] | Rm[:,1]]`.
- **hand15** (no real fingers → grip lerp): `hand15(g) = (1−g)·OPEN15 + g·CLOSED15` (5 fingertip XYZ in EE frame; both hands share the table):
  - `OPEN15  = [0.0005,0.0958,-0.034, 0.0285,0.0222,-0.1263, 0.0085,0.0217,-0.1285, -0.0116,0.0222,-0.1282, -0.0314,0.0229,-0.1259]`
  - `CLOSED15= [0.0052,0.0775,-0.0701, 0.0266,0.044,-0.0916, 0.0085,0.0441,-0.1016, -0.0097,0.044,-0.0937, -0.0278,0.044,-0.0919]`
- **head9** (fixed camera): state uses `HEAD9 = [-0.60,0,1.17, 0.033,-0.999,-0.003, 0.818,0.028,-0.574]`; action uses `IDENT9 = [0,0,0, 1,0,0, 0,1,0]` (zero-motion delta).

### Writes (zarr v3, groups/keys)
- `images.front_1` — (n,) variable_length_bytes, each a zstd JPEG(q92) of the agentview image.
- `action57/action` (n−1,57 f64) = `[IDENT9 | Leef9 | hand15(Lg[1:]) | Reef9 | hand15(Rg[1:])]` (grip sampled at **t+1** to align the delta window); `action57/valid` (n−1,3 bool) = `[1 | lv | rv]`.
- `state57/state` (n,57 f64) = `[HEAD9 | abs9(L) | hand15(Lg) | abs9(R) | hand15(Rg)]`; `state57/valid` (n,3) all-ones.
- `meta/eef_speed` (n,2 f64) per-hand `‖Δpos‖·FPS` m/s (frame 0 = 0).
- Root attrs: `embodiment="mimicdroid"`, `robot="GR1_DemoTwoHand"`, `env="SinkEnvForPlay"`, `task_name="play"`, `task_description="play"`, `fps=20`, `domain="hand_pose"`, `layout="head9|Leef9|Lhand15|Reef9|Rhand15"`, `head="fixed_camera"`, `hand="predefined_open_closed_lerp"`, `mimicdroid_play=True`, `subgoals=json.dumps(subgoals)`, `n_subgoals`.

Driver: `--dir all` iterates digit-named dirs; each demo key → `{DST}/{dir}__{key}`.

---

## `58_upgrade57_1x.py` — legacy 20→57 in-place upgrade
Same constants as `10_`. Legacy layouts: `state20 = [Labs9|Lgrip|Rabs9|Rgrip]` (idx 0:9,9,10:19,19),
`action20 = [Ldelta9|Lgrip(t+1)|Rdelta9|Rgrip(t+1)]`. Expansion:
```
st57 = [broadcast(HEAD9,n) | st20[:,0:9] | hand15(st20[:,9]) | st20[:,10:19] | hand15(st20[:,19])]
ac57 = [broadcast(IDENT9,m) | ac20[:,0:9] | hand15(ac20[:,9]) | ac20[:,10:19] | hand15(ac20[:,19])]
```
`state57.valid` = ones(n,3); `action57.valid` = `[1 | aval_L | aval_R]`. Then **deletes** `action20`/`state20`
and stale `*_done` attrs. Idempotent (skips if `state57` present and `state20` gone).

---

## 2×-speed (data-level) — `MimicDroidDataset/bake_training2_2x.py`
`new_T = round((T-1)/2)+1`, `idx = round(linspace(0,T-1,new_T))`. Every per-frame dataset subsampled by
`idx`; the (n,14) absolute `actions` **resampled** via `resample_actions(…,2.0)`: pos block lerp, rotation
block **Slerp** (rotvec at b+3:b+6), grip dims (12,13) **nearest-neighbor**. `subgoal.json` start/end
rescaled to the new timeline. Then `10_ --src …/training2 --dst egoverse_mimicdroid_2x`. (Note:
`56_md12_2x.py` is a DIFFERENT, render-only 2× — same window played at 2× subsample, not a data conversion.)

---

## `11_mimicdroid_desc.py` — descriptor → `desc_mimicdroid.npz`
Reuses the canonical human descriptor `01_desc_5s_si.py` (`blocks()`, `seq_parts()` via importlib) but
overrides `WIN=100, STRIDE=20` (5 s/1 s @20 Hz; vs egoverse `WIN=150, STRIDE=30` @30 Hz — both subsample to
the same TS so cross-Hz comparable). Robot-style: **fingers forced to 0**, grasp = gripper scalar. Windows
tagged by the **midpoint** subgoal type/lang. Output fields: `M,E,G, Swrist(40), Sfinger(=0), ep, w0, sgtype, sglang`.

> ⚠️ **Consistency flag:** `md_get()` reads **`state20/state`** (pre-upgrade 20-D layout), while `10_`/`58_`
> now write only `state57`. Either the descriptor was run before `58_`, or it needs a `state57` variant
> (`95_desc_from_a57.py` is the 57-D-native reader).

---

## Subgoals
`subgoal.json` per training dir: `{file, subsample, demos:{demokey:[{start,end,lang,type, [obj,obj_lang,src,dst]}…]}}`.
Types: `door_close, door_open, pick_place, faucet_on, faucet_off`; `pick_place` carries `obj/obj_lang/src/dst`
(e.g. `src="the sink", dst="the cabinet"`). `10_.load_subgoals` clips each `end` to `n` and stores the list
verbatim as the root attr `subgoals` (+`n_subgoals`). Downstream pairing branches on `type`+`src`/`dst`
(sink_pick / cab_up …) via `subgoal_of(ep,w0)` (entry with `start ≤ mid < end`). GR1 future-pairs (`78`) gate
on these same subgoal types (`TASK2REQ`).

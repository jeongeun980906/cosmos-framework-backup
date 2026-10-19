# EgoVerse v11 build — scale-corrupt cleaning + pairing rebuild (2026-07-15)

v11 = the v10 (3 s) pairing pipeline, **rebuilt byte-identically on scale-corruption-cleaned
source data**. No logic changed — only the source zarrs were cleaned and every output path
was repointed `*_v10.py`/`*_3s.py` → `*_v11.py`. Env: `/mnt/ddn/EgoVerse/emimic/bin/python`
(py3.11). See also memory `egoverse-v11-build`, `scale-corrupt-hand-tracking`.

---

## 1. Root cause: scale hand-tracking glitch bursts

The `scale` embodiment's action57/state57 normalization stats were exploding (action absmax
**161,213**, hand15 median absmax **125,963 m**, maxstd 295) while aria/MECKA/mimicdroid/gr1
were all clean (absmax ~1.0, hand15 ~0.14 m).

- **Source, not transform.** The raw `left.obs_keypoints_transformed` has isolated frames at
  ~200 km absmax (e.g. frame 3563 = 230,572) while the same frame's other hand/head are normal.
  hand15 = `R_wristᵀ(tips − wrist_pos)` — R orthogonal, cannot amplify → the garbage passes
  straight through (161k derived < 230k source). The waist@frame0 anchor is head-based (clean)
  and maps normal frames to [-1,1] correctly; only the burst frames are source garbage.
- **Shape:** short bursts (median 0.7 s, max 1.7 s @30fps = 13–51 frames), sparse spikes over
  0.1–1.4 % of an episode. Per-hand independent dropout (R-only 13, L-only 5, both 2). Channels
  = hand15 (18:33, 42:57) + some eef pos (9,10,11 / 33,34,35). Head clean.
- state (13.2 % of eps flagged) ≫ action (1.4 %): state is absolute pose (one bad frame flags),
  action is a delta (stays small on clean spans).
- **Why the guards missed it:** `eg_convert.py` (the current waist-frame generator) never calls
  `should_drop_episode`; `smooth_keypoints` never calls `reject_outliers_xyz` (pose only);
  med5/savgol9 low-pass can't remove a 13-frame burst; the drop threshold is a fraction (>1 %)
  so sub-1 % bursts on long eps slip through.

## 2. Cleaning (reversible, executed 2026-07-15)

Physically-impossible frames flagged with a CAP (pos>10 m / rot>1.5 / hand>1 m) → **82 corrupt
episodes** (action 43 / state 82) of 6,051 scale eps (1.35 %). Two dispositions by
clean-contiguous-run analysis:

- **DELETE 55** (clean_ratio < 80 % or near-total corruption) → whole dir moved to
  `/mnt/ddn/dataset/egoverse_removed_scale_corrupt/<ep>`.
- **TRIM 27** (longest clean run ≥ 90 f AND ≥ 80 % clean) → raw 21-array `[s:e]` slice
  (action57 delta is `[s:e-1]`; handles the 2-length regime raw≈3600 / proc≈3569; vlen images
  sliced byte-exact; `eef_speed_*` attrs recomputed over the kept slice). Original → `_trim_originals/<ep>`.
- Every retained value **bit-verified == original slice** before swap; 0 skips. Fully reversible
  via `egoverse_removed_scale_corrupt/clean_manifest.json`.

**Validation (real disk):** scale maxstd action 295 → **0.049** (aria 0.037 / MECKA 0.060),
state 295 → **0.466** (aria 0.465 / MECKA 0.490); hand15 125,963 m → **0.85 m**. scale eps
6,051 → 5,996. Two-decimal match to aria/MECKA.

Artifacts (session scratch): `/mnt/ddn/tmp/scale_noise_scan/` (scan/plan/plots/`batch_clean.py`).

## 3. v11 pairing rebuild = v10 logic, repointed

Clones (dir-only sed repoint from `*_v10.py`/`*_3s.py`; diff-verified 0 logic lines changed):
`egoverse_pair_builder/{01_v11,11_v11,70_3s_v11,71_3s_v11,72_3s_v11,46_v11,62_v11,73_v11,73b_v11,74_v11}.py`.
Orchestrators: `run_v11_pipeline.sh` (01→73b), `run_v11_md.sh` (46 --build → 62 → 73b).
Intermediate `egoverse_pairs_v2_3s_v11` / `egoverse_pairs_v8_3s_v11`; final
**`egoverse_pairs_v11`** (+ `_test`). `t5_embeddings.pkl` reused (same `lang_taxonomy` vocab).

- **Hyperparameters identical to v10**: NPE=40, SAME_FLOOR=0.80, SAME_N=8, CROSS_MAX=7,
  SIM_RATIO=1.6, DTW_BAND=6, MINK=5, MAXK=20, HAND_W=3.0, OPP_THRESH=0.25.
- **MD 4-task gates** (46_v11): TurnOnFaucet→FAUC (scale-only); CloseCabinetDoor→DC (8 DC_PFX
  eps, **last 20 %** = `0.8*last`, tuned 40→10→5→20 %); PnPSinkToCabinet→SINKPICK(grasp)+CABUP(release);
  PnPSinkToPlate→PLATE. door_open→DOOROPEN (first ≤150 f). MD full re-match (option A) because
  copying v10's MD referenced deleted eps in 32.7 % of queries (via human8 slots).

**Final v11:** windows **1,085,542** (MECKA 557,843 / scale 367,085 / aria 110,953 / eva 33,176 /
mimicdroid 16,485); pairs **1,039,861** (human 1,023,666 + MD 16,195).
**Deleted-ep leak check: window 0, match 0.**

## 4. Render verification

`74_v11.py` renders human `[Q|SAME|X1|X2]` cross-emb + MD `[MDQ|HUM|MD1x|MD2x]` samples.
Fixed to true 3.0 s playback (`clip n=33→72`, `mimwrite fps=8→24` = 72/24 = 3.00 s). 25 clips,
all 3.00 s, at `/mnt/ddn/tmp/egoverse_v11_render/` (+ `_montage_mid.png`). Confirms SAME 1 +
cross 7 per human query, MD 8 human + 12 MD (6× 2x), and clean scale columns (glitch 0).

## 5. Training

Training normalization + configs live in **cosmos_policy** — see
`cosmos_policy/docs/egoverse_v11_training.md`. (cosmos-framework's
`egoverse_normalization.json` is a separate artifact for the GR1 server / LIBERO action
policy and is NOT used by cosmos_policy mid-training.)

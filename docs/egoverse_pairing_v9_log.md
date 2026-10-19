# EgoVerse demo↔query pairing — v9 working log

> **Later builds:** v11 (scale-corrupt-cleaned rebuild of the v10 3 s pairing, 2026-07-15) →
> `docs/egoverse_v11_build_log.md`. Training-side (v11 configs / normalization) →
> `cosmos_policy/docs/egoverse_v11_training.md`.

Living log so we STOP re-deriving the same conclusions. Append decisions here.
Pair-builder dir: `/mnt/ddn/dataset/egoverse_pair_builder/`. Python env (has faiss):
`/mnt/ddn/EgoVerse/emimic/bin/python`. Data: `/mnt/ddn/dataset/egoverse/<ep>/`.
Fixed eval set = the egoverse_gen samples in `/mnt/ddn/tmp/egoverse_gen/*_action.npz`.
Renders → `/mnt/ddn/tmp/egoverse_a57desc/`.

## Current v9 descriptor (`95_desc_from_a57.py`) — CONFIRMED
- Source = the policy's own **state57 / action57** (aria-fixed), NOT raw obs.
  - state57 layout: `head9(0:9) | Leef9(9:18) | Lhand15(18:33) | Reef9(33:42) | Rhand15(42:57)`.
- Per-frame, per hand: `[start-rel pos3 + 6D rot6 + action-eef delta9 + hand15] = 33` → both hands **66-D/frame**, subsample to SEQ_TS=40.
- Normalization: **iso unified** = per-window per-channel z-score `(F-mean)/(std+1e-6)` for same AND cross. (Removes magnitude → matches SHAPE. Class gate keeps same-emb task-correct.)
- Finger weight: `FING_W=0.25` (15 finger chans/hand down-weighted; wrist trajectory leads).
- Handedness: **reach-Lfrac + threshold penalty**. `sc = DTW + HAND_W*max(0, |Lf_q-Lf_c| - OPP_THRESH)`, `HAND_W=3.0`, `OPP_THRESH=0.25`. Lfrac = left-hand share of **REACH** (peak eef displacement `max|pos_t-pos_0|` per hand) — see rejected list for why not path-length.
- Shortlist: same-emb from FINE class; cross-emb from FINE, else fall back to COARSE class (MECKA-only fine-classes reach scale/aria).
- Episode rules: exclude the query's OWN episode; `EP_CAP=2` windows/episode.
- Constraints: `SAME_N=8`, cross dynamic count = clip((score≤best×SIM_RATIO=1.6),0,**`CROSS_MAX=7`**), `MAXK=20`, `NPE=40`, `WIN=150`. (CROSS_MAX lowered 12→**7** for the v9 full build, 2026-07-01 — 12 was too many; still dynamic.)

## v9 FULL BUILD → training pipeline (2026-07-01) — CANONICAL, RAN
The above descriptor (95) was eval-only. The at-scale build ports it into the v8 machinery (61) over **all 1,015,122 kept human windows**. Chain (in `/mnt/ddn/dataset/egoverse_pair_builder/`):
1. **`70_desc_v9.py`** — precompute the v9 descriptor for every window → 3 index-aligned memmaps `desc_v9_{V9(N,40,66 f16),lfrac,valid}.npy` (per-window iso baked in, DTW=plain L2). Shard by episode (`--init` then `--start/--end`). valid = 1,015,122 (= kept).
2. **`71_build_v9_human.py`** — 61 clone, but DTW on `desc_v9` + handedness penalty + `lang_taxonomy` fine/coarse gate. Cross shortlist = faiss on **whiten(M) motion-only** (no text → no t5 bias) within (emb,fine) bucket, fallback (emb,coarse). `CROSS_MAX=7`. Shardable. Out `v9_human_pairs.npz`.
3. **`72_run_v9_parallel.py N`** — fans out 71 as N shards, merges. **Use N≤12** (cgroup mem cap = **480 GB**; 24 shards OOM-killed the first shards. 12 shards ≈ 340 GB peak, ~60 min wall due to CPU contention).
4. **`73_build_v9_manifest.py`** — materialize + **train/test split**. Holdout = **150 episodes, stratified by (emb,class)** proportion, EPISODE-level (no window leak), **LOOSE** (removed as training queries only; still usable as demos). Writes `egoverse_pairs_v9/{windows,pairs}.npz` (TRAIN, 1,001,990 rows), `egoverse_pairs_v9_test/{...}` (TEST, 13,132 rows, 150 eps), `egoverse_pairs_v9/holdout_eps.txt`. windows table identical for both; only `query_idx` differs.
5. **`74_render_v9.py --pairs <npz>`** — side-by-side `[Q | SAME | CROSS#1 | CROSS#2]` front-cam mp4s for eyeball validation. `→ /mnt/ddn/tmp/egoverse_v9_render/`.

Stats: avg 11.5 matches/query (same 5.2 + cross 6.3, max cross 7); 0 zero-match; 95,668 (9.4%) have 0 cross (no cross-emb analog — e.g. MECKA-isolated classes); 0 queries with >2 windows from one episode.

**PYTHON ENV per step (IMPORTANT — wrong env = faiss/zarr/transformers import fails):**
- Steps 1–5 (matching build `70`–`74`, incl. `72_run_v9_parallel.py` which hardcodes it): **`/mnt/ddn/EgoVerse/emimic/bin/python`** (py3.11, has `faiss` + numcodecs zarr reader). Run from `/mnt/ddn/dataset/egoverse_pair_builder/`.
- Cosmos 3 training / offline-generate eval (`cosmos_framework.scripts.train`, `tools/eval_egoverse_generate.py`, `rerender_skeleton_v9fix.py`): **`/mnt/ddn/cosmos-framework/.venv/bin/python`** (py3.13).
- Cosmos 2.5 training / eval on this pairing (`cosmos_policy` egoverse-ret, native action57): **`/mnt/ddn/cosmos_policy/.venv/bin/python`** (py3.10; per-pod `uv sync --extra cu128 --group pusht`). See [[cosmos-policy-v9-action57]].
- action57/state57 producers/converters: see the env table in `docs/action57_state57_convention.md` §7.

**Wiring (2026-07-01):**
- Training config `action_policy_egoverse_nano.py` `EGOVERSE_PAIRS_DIR` default → **`egoverse_pairs_v9`** (was v8).
- Offline eval `tools/eval_egoverse_generate.py` + `eval_egoverse_report.py` `--pairs-dir` default → **`egoverse_pairs_v9_test`** (TEST split only).

## Head-pose skeleton render fix (`tools/eval_egoverse_generate.py:_fk_skeleton`) — clarified 2026-07-01
The head-static complaint (`aria2MECKA_527034_skeleton.mp4`) was a **stale file** (Jun 30 03:43, rendered 15 h BEFORE the fix saved 18:35), NOT a live bug. `_fk_skeleton` DOES integrate head9 deltas (`HP[t+1]=HP[t]+HR[t]·Δpos`, per-frame world map). The model DOES predict non-constant head9. Two facts to remember:
- **MECKA ≈ fixed-camera rig**: GT head median translation **1.6 cm** → its skeleton head is *correctly* near-static. Don't validate the head fix on MECKA.
- **aria/scale have real head motion** (median ~11 cm, up to 21 cm). Validate the fix there. `tools/rerender_skeleton_v9fix.py <label>...` re-renders any saved `egoverse_gen/*_action.npz` with the current renderer (NO model/GPU) — proven movers: `MECKA2scale_98750` (head 27 cm), `aria2scale_250564` (22 cm), `scale2aria_62799` (rot 101°).

## Data facts about CLASS LABELS (scanned 2026-06-30, 13457 episodes, 32 task_names)
- `task_name` attr is CLEAN for **aria** and **scale**; e.g. cup = `cup_on_saucer`(aria 276) / `flagship_put_cup_on_saucer`(scale 411) / `freeform_put_cup_on_saucer`(scale 109).
- **MECKA_BIMANUAL has task_name = "debug" for ALL 6714 episodes** — it is a PLACEHOLDER, not junk. The real activity is ONLY in `task_description` (verbose "The person is ..."). **NEVER drop "debug" → that deletes all of MECKA.**
- `annotations` array is EMPTY (shape (0,)) — there is NO per-frame subgoal annotation in the data.
- Dishwasher family is cleanly separable by task_name (aria/scale): `freeform_do_dishes`(679) / `freeform_clean_up_the_dishes`(74) = hand-washing; `freeform_load_dishwasher`(222) / `freeform_unload_dishwasher`(211) = machine. My keyword `coarse()` lumped these → q203607 mis-mapped.
- cup has NO MECKA episodes (aria + scale only).
- => Gate must be HYBRID: **task_name for aria/scale, task_description-keywords for MECKA**.

## FINALIZED taxonomy (2026-06-30) → `lang_taxonomy.py`
Replaces keyword coarse()/fine(). `activity(emb,task_name,desc)`: aria/scale strip task_name
prefix/suffix; MECKA classify task_description (98% coverage, ~2% "other" long-tail). Canonical
distribution (aria/scale/MECKA) and cross-emb overlap:
- cross-emb shared: cup_on_saucer(310/520/0), bagging_groceries(211/1233/0), put_object_in_container(1/550/534),
  do_dishes(0/753/1084), sort_bins(0/195/1334), unpack_grocery(0/1471/15), stack(0/12/304), pick_place(135/0/30).
- MECKA-only (same-emb at canonical level): fill_scoop(1478), seal_wrap(405), assemble(715), fold(270), clean_wipe(321).
- scale-only: tidy_kitchen(462), utensils_in_drawer(422), load_dishwasher(257), unload_dishwasher(211).
- SUPER super-groups for cross-emb FALLBACK (only when exact canonical has no cross member). Opposites
  kept apart: bagging≠unpack, load≠unload. packaging={bagging,put_object_in_container,fill_scoop,seal_wrap,
  utensils_in_drawer} (aria212/scale2205/MECKA2417 — so MECKA fill/seal reach scale packaging); dishes; sorting;
  cup; cleaning={clean_wipe,tidy_kitchen}; dishwasher_load; dishwasher_unload; stack; pickplace; assemble; fold.
Cached per-episode tags: `/mnt/ddn/tmp/egoverse_langtags.npy` (cols: epkey, embodiment, task_name, task_description).

## Taxonomy gate INTEGRATED into `95_desc_from_a57.py` (2026-06-30) — FIXED bugs
wfine = canonical activity (via lang_taxonomy + `/mnt/ddn/tmp/egoverse_langtags.npy`); wcoarse = super_of.
Verified on egoverse_gen: q203607 load_dishwasher → MECKA "loading" (was "unloads"; load/unload/do_dishes
distinct canonicals but share 'dishes' super so dishwasher cross-fallback finds a dish scene); 250564 now
unpack_grocery→MECKA "unpacking" (bagging≠unpack separated); put_object_in_container cross-emb across
aria/scale/MECKA; MECKA-only seal_wrap/fill_scoop reach scale via 'packaging' super; assemble→NO-CROSS
same-emb fallback. To port to the real build: same wfine/wcoarse swap in `61_build_v8_human.py`.

## CUP sub-segmentation investigation (2026, exhaustive — see before re-attempting)
scale cup_on_saucer episodes are LONG (49-127s) REPEATED tasks: `place cup → reset-pose → cup_reset(grasp cup, lift, move saucer) → reset-pose → place …`. The "reset-pose" occurs at EVERY transition (both hands fingers EXTENDED/open, held). aria cup just repeats clean place (NO reset) — only scale needs cleaning. Desired: relabel the cup_reset segments (desc "grasp cup lift it, and move the saucer") as a separate task so they don't pollute place matching. Status: NOT solved. What was tried + RESULT (do not repeat blindly):
- reset-POSE detector that WORKS partially: per-episode z-normalized BOTH-hand fingertip-extension (mean tip→wrist dist, state57 hand15) `zR>0.2 & zL>0.2`, dwell ≥0.6s, NO motion cap. Validated vs user frame-labels: ep97512 recall 8/9, ep98643 9/10, FP low. Finger-extension is the signal ("손 피고있고"); head-relative palm orientation and inter-hand distance and hand-switch all FAILED on the user's full label set.
- BUT pose detection is NOT perfect-recall: long ~10s gaps = missed poses → the place/cup_reset alternation breaks.
- place vs cup_reset are KINEMATICALLY INDISTINGUISHABLE (the difference is WHICH object moves — cup vs saucer — which is not in the 57-D proprio). Verified: vertical-motion anchor (place=down vs cup_reset=lift) is noisy, does NOT alternate. Start phase unknown + trash at episode start make alternation-from-start unsafe.
- CONCLUSION: proprioception cannot segment cup place/cup_reset to a reliable bar. The only signal with the discriminating info is VISION (which object moved).
- VISION attempts (2026, offline env, no internet): (1) DINOv2 vit-base (cached at ~/.cache/huggingface + /mnt/ddn/.hf_cache) frame-embedding + kmeans → clusters by SCENE/episode appearance/lighting, NOT by place/cup_reset action. (2) Qwen3-VL-2B-Instruct (.venv python, transformers 4.57.6 supports qwen3_vl; emimic's 4.53 does NOT) → forced CUP/SAUCER binary collapses to always-CUP; open-ended "which object is moved" gives a FLICKER of real signal (sometimes says "moving a saucer" at reset times) but inconsistent/ungrounded — not reliable. (3) Qwen3-VL-8B is only PARTIALLY cached → cannot load offline. Net: no offline method meets "절대 실수 금지". Vision via a properly-cached LARGER VLM (online) asking "is the cup or the saucer being moved" is the most promising untested path — the 2B flicker suggests a stronger VLM could do it.
- PRAGMATIC PATH (recommended): apply the working reset-POSE detector to drop transitional-pose windows from scale cup (partial cleanup), ship v9 with the good taxonomy, defer full cup place/cup_reset split to an online VLM pass or data-prep subgoal labels. Reusable detector = both-hand finger-extension dwell above.

## DECISIONS / OPEN
1. **cup sub-meaning — DECIDED 2026-06-30: KEEP AS-IS (existing taxonomy gate), do NOT sub-segment.**
   `*_put_cup_on_saucer` mixes place / move-saucer / handover within one label, but this is NOT separable
   offline (exhaustively proven — see "CUP sub-segmentation investigation" above: proprioception ALL features
   failed; static-frame vision (DINOv2) clusters by scene; Qwen3-VL-2B unreliable; 8B not cached). User call:
   ship cup at the gate level (`cup_on_saucer` canonical) like every other class; revisit only with an online
   larger VLM or source-level subgoal labels. The reset-POSE detector + cup_reset relabel are SHELVED (not used).
2. **dishwasher mapping** (q203607): fix expected once gate moves to task_name (separates do_dishes/load/unload). Verify after.
3. cross-emb task_name correspondence map needed (aria `cup_on_saucer` ↔ scale `*_put_cup_on_saucer`; dishwasher families; pack/grocery families; MECKA via description-keywords).

## TRIED & REJECTED — do NOT repeat
- **t5 / action-cosine embedding gate** — vocab-fragile, can't bridge cross-emb terse↔verbose. BANNED (data: same-class act-cos median 0.33–0.43).
- **path-length Lfrac** for handedness — points the WRONG way (q62799 path→R, but visually-active arm is L). Use REACH (peak displacement).
- **ungated handedness penalty (HAND_W=2.0, linear)** — regresses two-handed tasks (grocery 250564 "bagging"→"wiping", 312301 "packing"→"cleaning").
- **query-asymmetry gate** for handedness (GATE_LO/GATE_HI) — can't separate 62799 (needs fix) from 312301 (must stay). Replaced by candidate-diff THRESHOLD.
- **dropping all task_name=="debug" episodes** — that is ALL of MECKA. The trap. Classify MECKA by description instead.
- **pack_container fine-class split** — narrowed the pool, made "Put Object in Container" worse than v8. Removed.

## GR1 post-train pairing: PAST/causal (75) is canonical — forward (67/68) is DEPRECATED (2026-06-30)
GR1↔play demo retrieval for the conditioning policy has TWO builders; only ONE is correct, and a run
was trained on the wrong one. Recorded here so we never re-tangle it.
- **CANONICAL (since 2026-06-30) = `78_future_pairs_abstraj.py`** (PAST/causal, ABS-TRAJ desc — REPLACES 75):
  descriptor = per-frame BOTH-hand ABSOLUTE 6D pose `[Lpos3,L6Drot,Rpos3,R6Drot]` in METERS over the PAST window
  `[e-pl,e]` (`pl=min(WIN,e)`), **NO z-score / NO start-relative** (GR1+play both robocasa → same units). Match =
  mean-pose L2 SHORTLIST(150) → abs-traj DTW(SEQ24,band6), TOP-5, subgoal-gated + `prel≤WIN`. WIN=32/STRIDE=4/FUT=32,
  conditions on demo FUTURE `action57[m_w0:m_w0+FUT]`. OVERWRITES `gr1_future_pairs_{1x,2x}.npz` (75's z-scored output
  → `*.75bak.npz`). t=0 = same-task TaskDemo opening `[0:32]` (`77_gr1_t0_taskdemo.py --speed`). `build_gr1_posttrain_pairs`
  ingests unchanged. Video-verified smooth/forward (4 tasks × 1x/2x, `gr1_descfix_{1x,2x}.mp4`).
- **SUPERSEDED = `75_future_pairs.py`** (z-scored `01_desc_5s_si` wrist, SHORTLIST=300, ZW=5.0): dropped ABSOLUTE
  position/magnitude → t>0 matches 16–49cm off (active hand worst) + jumped backward (video-validated). 78 fixes this.
  (human/cup pairing still uses `01_desc_5s_si` z-scored — UNTOUCHED, only the GR1 path moved to 78.)
- **DEPRECATED = `67_gr1_desc.py` + `68_gr1_match.py`** (`gr1_play{1x,2x}_match.npz`, `play="training_1x"`):
  descriptor over a FORWARD WIN=100 window `[w, w+100]` (the robot's UNSEEN future = non-causal), no abs-height.
  Cannot be reproduced by a causal online retriever → train/eval conditioning mismatch.
- **PROOF the 1x ckpt is STALE (timeline 06-30):** ckpts trained 13:59 (iter_6000)/16:04 (iter_8000);
  `egoverse_pairs_gr1_1x` REBUILT from 75 at **17:07 — AFTER training**. CURRENT pairs ARE 75-causal
  (verified: 200 `w0=0` → same-task taskdemo opening; `PnPSinkToCabinet demo_0 w0=4 → 009@8`, not 68's `006@7640`).
  ⇒ that ckpt is invalid. Inference contract: `docs/gr1_inference_retrieval_spec.md`;
  memory [[gr1-future-pair-pipeline]], [[gr1-mimicdroid-online-eval]].
- Format (state57/action57/grip/env_action) independently verified bit-exact (round-trip 0 err) — the failure was
  retrieval, not format. Inference `gr1_online_retrieval.py` scoring was reconciled to 75 (`ZW=5.0`,`SHORTLIST=300`),
  but now must move again to the **78 abs-traj descriptor** to match the new pairs (tracked gap; don't edit per user).
- **REMAINING (in order): (1) §6c descriptor fix DONE via 78; (2) switch inference to the 78 abs-traj descriptor;
  (3) RETRAIN once on the 78 pairs; (4) wire server t=0=taskdemo→t>0=play pool switch.** Per-frame (t+k) dataloader DONE.

## EgoDex OFFLINE eval on the trained policy (2026-07-02) — held-out embodiment
**EgoDex** (Apple Vision Pro human-hand demos) is a NEW embodiment used ONLY for offline eval of the trained Cosmos 2.5 egoverse-ret policy (NOT training). Same egoverse zarr schema (`images.front_1` jpeg-object + `action57/action` + `state57/state`, `robot=human`, 30 Hz, `state57_abs`), converted by `egoverse_pair_builder/egodex_to_egoverse57.py` (+ `egodex_add_images.py`) → `dataset/egodex_egoverse/` (46,234 episodes).

**Pairs (self-contained, NOT the v9 windows/pairs manifest):** `dataset/egodex_pairs_top3.npz` — within-egodex WINDOW-level top-3 retrieval (`egodex_pair_v9.py`, win=150 stride=150 → 115,834 windows; keys `names/win_ep/win_w0/win_wl/top_idx/top_dtw`; avg 3.0 matches, 0 zero-match). Each query self-retrieves its 3 nearest egodex windows as the demo condition (same query+demo paradigm the policy trained on). Pair-viz `egodex_pair_render.py` (**emimic** env) → `[Q|#1|#2|#3]` front-cam mp4 @ real-time 30 Hz → `/mnt/ddn/tmp/egodex_pairs_render/` — verified semantically coherent (add_remove_lid → 3 cup-lid matches, dtw 2.1–2.6).

**Per-embodiment normalization (⚠️ MUST be 1/99 PERCENTILE, not min/max):** EgoDex is its own embodiment → own action range. `egoverse_pair_builder/egodex_gen_norm.py` (**emimic** env) → `dataset/egodex_{action57,state57}_stats.json`, per-emb `{"egodex":{lo,hi}, "_global":{...}}`. **BUG→FIX (2026-07-02):** the first version used RAW MIN/MAX, but a few ARKit/frame-drop outlier frames (0.1–0.25 m jumps) across 46k eps blew up the head/eef Δpos range **17–100× vs the egoverse training stats** (which use 1/99 pct in `egoverse_ret_dataset._load_or_compute_action_stats`). That compressed the conditioning demo action ~20× toward 0 AND inflated the de-normalized output ~20× → the eval showed `scale_ratio` 16–46× and cos≈0 for egodex. FIXED: the scanner now samples ~8000 eps and takes **1/99 percentile** (matches the trainer). egodex is genuinely near-static (|Δpos| head median 0.4 mm, hands p99 ~2 cm — units are FINE, meters/30 Hz); the pct range is now ≈ egoverse scale (head Δpos span 0.0035 vs old 0.22 vs aria 0.013). After the fix, iter-10k egodex `scale_ratio`→0.44/0.65/1.11 and `dtw_shape` 17–22 — i.e. in line with the in-distribution v9-test control (2.85/1.25/1.24, dtw_shape 13–21), proving the earlier "no transfer" result was the STATS bug, not an OOD gap. `*.minmax.bak.json` = old buggy stats. **Rule: any new-embodiment eval stats MUST use the SAME 1/99-pct method as training.**

**Eval infra (all in `/mnt/ddn/cosmos_policy`, py3.10 `--group egoverse`):**
- `cosmos_policy/datasets/egodex_ret_dataset.py` = `EgoDexRetDataset(EgoVerseRetDataset)` — subclass: overrides `__init__` (egodex_pairs_top3 → window tables, emb=`egodex`, cls=`other`, query=arange, match=top_idx) + `_grp` (egodex root); everything else inherited. Text: the model never saw an `[egodex]` tag, so it ALIASES `"<cmd> [egodex]"` → the trained `"<cmd> [human]"` t5 embedding (no T5 call).
- Experiment `cosmos_predict2p5_2b_egodex_ret_5s_eval` (same 5 s model/layout; dataset swapped) in `egoverse_ret_experiment_configs.py`.
- Eval scripts gained `--act_stats` (point at egodex stats): `eval_egoverse_ret.py` (viz: `[demo|GT|gen]` video + GT/gen 3D skeleton panels) + `eval_egoverse_ret_batch.py` (shape metrics). Launchers: `scripts/run_egodex_eval_viz.sh` (N viz samples); `scripts/run_batch_8gpu.sh` egodex mode = `EXP=cosmos_predict2p5_2b_egodex_ret_5s_eval ACT_STATS=…/egodex_action57_stats.json MANIFEST=` (empty MANIFEST skips the v9-only manifest_dir override).
- Checkpoint: `convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_5s --iter <N>` (with `IMAGINAIRE_OUTPUT_ROOT=<run dir>`) → `model_<N>.pt`, then `PT=<.pt> bash scripts/run_egodex_eval_viz.sh`.

**Scale-invariant metrics (2026-07-02):** EgoDex is UNSEEN, so the policy may emit a different ABSOLUTE motion scale than GT — that must NOT count as a shape error. `eval_egoverse_ret_batch.py` now splits metrics: PRIMARY = **scale-invariant** `dtw_shape` (DTW after re-scaling each trajectory to unit total path length), `cos` (per-step Δpos direction), `corr` (Pearson, affine-inv); REFERENCE = scale-dependent `dtw_norm`, `mse`; DIAGNOSTIC = `scale_ratio` (gen/gt |path|, surfaces the magnitude gap). Unit-tested: for gen = 5×·GT (same shape), `dtw_shape=0.0, cos=1.0, corr=1.0, scale_ratio=5.0` while `dtw_norm=4.6` wrongly ≈ its "different shape" value (5.4) — i.e. dtw_norm/mse are unusable on egodex, dtw_shape/cos/corr are correct. (merge_batch_metrics.py summary updated too.)

**Verified end-to-end (2026-07-02):** reader smoke OK (video (3,197,224,224)=8R+13, actions (92,57) mask 57, t5 (512,1024) via alias). Ran viz on the per-emb `iter_000001000` ckpt → 4 samples render fine (`/mnt/ddn/tmp/egodex_eval_viz_iter1000/`). ⚠️ iter 1000 is FAR too early (50k target): GEN action magnitude ~10–40× GT (head Δpos GEN ~0.032 vs GT ~0.0008), so quality is NOT yet meaningful — the run only proves the pipeline. Re-run viz + 8-GPU shape metrics on a much later per-emb checkpoint. See memory `egodex-eval` + `cosmos-policy-v9-action57`.

## Pendulum history (detail): see `docs/egoverse_pairing_history.md` (v5→v8) and memory `egoverse-pairing-v9-iso`.

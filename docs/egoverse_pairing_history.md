# EgoVerse retrieval-pairing — v5 → v8 history, root issues, and the v9 design question

This documents how the demo↔query **pairing** (which demo conditions which query) is
built for the EgoVerse conditioning policy, the approaches tried so far (v5–v8), **why
each was changed**, and the open design question for v9. Written because we keep
oscillating between two extremes; the goal is to stop reverting and find the middle.

- **Pipeline location:** `/mnt/ddn/dataset/egoverse_pair_builder/` (numbered scripts).
- **Run with:** `/mnt/ddn/EgoVerse/emimic/bin/python` (the `emimic` venv — has faiss +
  egomimic `process_egoverse`; the cosmos-framework `.venv` does NOT have faiss).
- **Inputs:** `egoverse_pairs_v2/desc_absdelta.npz` (per-window motion descriptor) and
  `ep_lang_act.npz` (per-episode text + T5 activity embedding `epemb`).
- **Output:** `windows.npz` (ep,w0,emb,cls,…) + `pairs.npz` (query_idx, match_idx(Nq,20),
  match_dist). Consumed by `EgoVersePairedDataset`.

---

## 1. The window descriptor (`01_desc_5s_si.py`) — the MOTION side

Per fixed **WIN=150** frame window (5 s @30), both hands, expressed relative to the
window start. Two consumers: a flat vector `M` (+`E`,`G`) for the L2/faiss prefilter, and
a subsampled sequence `SEQ` (wrist + fingers) for banded-DTW rerank.

- `blocks(fr,w0)` → **`M`** = `concat(iso(Mabs), iso(Mdel), Mhand)`:
  - `Mabs` = absolute wrist path (right/left/relative) rel. start,
  - `Mdel` = per-step delta path,
  - `Mhand` = 5-finger closeness (10-d).
  - **`E`** = per-hand **path length** `‖dR‖.sum(), ‖dL‖.sum()` ← the MOTION MAGNITUDE.
  - **`G`** = grip/grasp signal.
- `seq_parts(fr,w0)` → **`SEQ`** = wrist(abs+delta, 15) + finger(10), **per-window
  z-scored**, subsampled to SEQ_TS=30. Used by DTW.

### ⚠ The `iso` / z-score concern (user-flagged root issue)
```python
def iso(a):  a0=a-a.mean(0); return (a0/(a0.std()+1e-6)).flatten()   # scale-normalize
# seq_parts: wrist=(wrist-wrist.mean(0))/(wrist.std(0)+1e-6)          # per-window z-score
```
Both `M` (via `iso`) and `SEQ` (z-score) **divide out the magnitude** → they encode motion
SHAPE only. Consequences:
- A **big reaching** motion and a **small fidget** with similar shape look identical to the
  DTW reranker → big-reach query gets matched to small-fidget demo (the user's complaint).
  Measured: `corr(query motion, matched-demo motion) = 0.357`; 15 % of links have a >3×
  speed ratio.
- `E` (path length) DOES carry magnitude, but it is **whitened per-emb** in the prefilter
  `X` and is **absent from the DTW rerank order**, so it does not constrain the final pick.
- **Open question (user):** is the `iso` regularizer itself wrong for motion matching? It
  was added for cross-emb scale-invariance (different rigs span different absolute sizes),
  but it may be throwing away the very signal (motion magnitude) that distinguishes
  sub-activities. A middle ground (partial/anisotropic normalization, or magnitude as an
  explicit matched channel) likely beats both "full iso" and "no iso".

---

## 2. The semantic gate — the PENDULUM (v5 ⇄ v7 ⇄ v8)

The hard part is cross-embodiment semantics: which demos are the *same activity* as the
query, across rigs whose language differs (scale is terse "do dishes"; MECKA is verbose
"wash various dishes including plates and bowls"). Every version so far picks ONE of two
extremes and suffers its failure mode:

### v5 — hard keyword CLASS bucket
- `cls()` = 8-way keyword classifier; candidates must be in the query's class bucket.
- **Failure:** classes are **too coarse**. Long descriptions hit several keywords and
  collapse into broad buckets, so motion-similar but unrelated tasks match (the script's
  own example: "diamond painting" → "put cup on saucer").

### v6 / v7 — T5 activity-embedding gate (`03_match.py`, `MATCH_MODE=semantic`)
- Dropped the class bucket. A candidate must be **(1) motion-near** (faiss top-NPE per emb,
  L2 ≤ T) **AND (2) activity-near**: cosine of the **T5 activity embedding** (`epemb`,
  1024-d) ≥ **TAU=0.6** (adaptive: MARGIN relax, ABS_FLOOR=0.35). Pool size adapts; a
  semantically isolated query gets fewer/no matches instead of wrong ones.
- v7 motion mechanics (carried into v8): whitened-`M` L2 shortlist → wrist+grip banded-DTW
  rerank, per-emb whiten, ≤2 windows/episode.
- **Failure:** T5 cosine **cannot bridge the cross-emb vocab gap**. MEASURED on same-class
  legit cross pairs (scale × MECKA episode `epemb` cosine):
  | class | median act-cos | frac ≥ 0.5 |
  |---|---|---|
  | dishwash | **0.33** | **0.00** |
  | pack | 0.42 | 0.08 |
  | cup | 0.43 | 0.03 |
  | sort | 0.40 | 0.04 |
  A TAU=0.5–0.6 gate would drop **84–100 %** of legitimate cross-emb pairs (terse vs verbose
  phrasing → low cosine even for the SAME task). So T5 quietly kills cross-emb — fatal for a
  *cross-embodiment* conditioning policy.

### v8 — back to keyword CLASS, `activity()` (`61_build_v8_human.py`)
- Reverted the semantic gate to a keyword **activity class** (`activity()`: dishwash, cup,
  fold, sort, stack, pack, dishwasher, pickplace, other; CROSSABLE subset) precisely
  because T5 couldn't bridge vocab. Cross-emb matches only need to share the query's
  activity class (vocab-robust). Same-emb additionally uses an episode `epemb` cosine floor
  (SAME_FLOOR=0.80) — fine WITHIN an embodiment (same vocab) but NOT used cross-emb.
- **Failure (current):** the activity classes are **too coarse again** (v5's problem
  returns): `pack` lumps grocery-bagging + perfume-bottle-filling + party-favor-assembly +
  box-sealing + snack-packing. Within `pack`, cross-emb matches a perfume-fill query to a
  grocery-bag demo — same class, different sub-task (their T5 act-cos is only ~0.36).

### The pendulum, stated plainly
- **class only (v5/v8):** vocab-robust, bridges cross-emb ✓ — but too coarse, mixes
  sub-activities ✗.
- **T5 only (v6/v7):** fine semantic resolution ✓ — but vocab-fragile, cannot bridge
  cross-emb terse↔verbose ✗.
- **The v9 question = the MIDDLE.** We need a semantic signal that is (a) vocab-robust
  enough to bridge cross-emb (like the class) AND (b) finer than the 8 coarse classes (like
  T5). Do NOT just revert to either pole. Candidate directions (to explore, not yet chosen):
  - **Finer keyword classes / sub-classes** (split `pack`→bag/fill/seal, `pickplace`→
    pick/place by grip event + src/dst as `45_v8_human8.py` already does for MD queries).
    Still keyword-based ⇒ vocab-robust; finer ⇒ separates sub-tasks.
  - **Object/verb-level embedding** instead of full-sentence T5 (embed extracted
    object+action tokens; more robust to verbosity than whole-sentence cosine).
  - **Let MOTION do the sub-discrimination inside a class** — if `iso` is relaxed so
    magnitude survives, a big-reach query won't match a small-fidget demo even in the same
    coarse class; the motion descriptor itself separates many sub-activities, reducing how
    much the semantic gate must resolve.
  - A hybrid: coarse class as a hard gate (bridge vocab) + a vocab-robust finer score
    (object/motion) as a soft rerank — "somewhere in the middle".

---

## 3. The OUTPUT CONSTRAINTS — invariant across versions (do NOT change; only the RULE changes)

So results stay comparable across v5…v9, the per-query output STRUCTURE is fixed (from
`61_build_v8_human.py`):
- **K = MAXK = 20** matches per query.
- **same-emb: top SAME_N = 8** (banded-DTW ranked; episode-deduped; NPE=40 motion-rank
  shortlist; same-emb episode act-cos floor SAME_FLOOR=0.80; MINK=5 fallback).
- **cross-emb: DYNAMIC count** `clip((DTW ≤ best×SIM_RATIO).sum(), 0, CROSS_MAX)`,
  SIM_RATIO=1.6, **CROSS_MAX=12**; episode-deduped.
- **final = `sorted(same+cross by DTW)[:20]`**.
- windows WIN=150, STRIDE=30.

i.e. **same ≤8 + cross dynamic ≤12, total ≤20.** This is the *constraint*. The *matching
rule* (descriptor + scoring + gate) is the only thing a version is allowed to change.

## 4. Fixed evaluation set

Evaluate EVERY version on the **same fixed query set** — the `egoverse_gen` samples (the 20
queries already rendered under `/mnt/ddn/tmp/egoverse_gen/`), rendered as
`[query | demo]` so versions are directly comparable. Do NOT judge on fresh random scans
(that is the v7-style circling).

# GR1 post-train & eval with a 2×-sped demo pool (`v6_2x`)

> **Goal.** Post-train the GR1 retrieval policy where the retrieval **demo pool** (augmented
> DTH demos + t=0 taskdemo openings) is temporally **2× sped-up**, while the query and the
> `abs_action57` target stay unchanged 1× GR1. Question: does a 2×-compressed conditioning
> demo still transfer (ICL) as well as the 1× pool?

Everything is `_2x`-suffixed and the 1× `v6` artifacts are untouched — fully reversible
(delete the `__2x` demos, `*_2x_v6*` files, and `egoverse_pairs_gr1_v6_2x/`).

---

## 1 · What "2×" means  (verified — V1 video)

Each demo episode is temporally subsampled `idx = arange(0, n, 2)` (256 → 128 frames):

| array | 2× rule |
|---|---|
| `state57`, `images` | `[idx]` — **exact** (`state_2x == state_1x[even]`, max‖·‖ = 0) |
| `abs_action57` | `abs_action[idx[1:]-1]` — absolute pose target landing at the next kept frame |
| `action57` (delta) | translations summed over the 2 sub-steps; rot/grip from landing cmd |
| `meta/eef_speed` | `spd[idx] * 2` |

- Builder: `egoverse_pair_builder/make_2x_demos.py`
- Verify: `render_2x_compare.py` → `_viz_samples/{cmp2x,sheet2x}_*` (1×/2× identical trajectory)

## 2 · Matching descriptor — **no velocity**  (user decision)

Descriptor = **pos + rot6D(×0.25) + grip = 20-d**, velocity channel **dropped**.
Velocity (`diff-pos`) is the only speed-**dependent** channel — on a 2× demo it naturally
doubles and mismatches the 1× query; pos/rot/grip are absolute → fully speed-invariant.
`--no-vel` flag in the 2× matcher. V2 video confirmed lower DTW distances (0.02–0.21) and
same-task top-k retrieval.  (1× pool is 26-d with velocity; 2× pool is 20-d no-vel.)

## 3 · Approach-window fix  (user-flagged — verified by episode video)

`build_pool` started at `w0 = WIN(32)`, so the demo's first 32 frames (the **approach**)
were never a pool window. The query's early approach (t≈1…54) then all pinned to
`m_w0≈32` (post-approach) and the retrieved-demo pointer was **stuck**, never climbing from
t=0. On a 2× demo the approach is compressed to ~frames 0–27, so `w0=32` is strictly
*post*-approach — the approach was invisible as conditioning.

**Fix:** start the pool at `w0 = STRIDE(4)` (short past window `pl=min(WIN,w0)`, mirroring
the query side). Now the demo pointer climbs from t=0:

```
before:  t0:0 → t1:29 (JUMP) → stuck 31–35 for ~54 steps
after:   t0:0 → 1,2,…,7 → 12,16,24,51,…,90   (approach captured)
```

Check: `render_2x_match_episode.py` → `_viz_samples/ep2x_*.mp4` (query plays; matched 2×
demo advances in sync).

---

## 4 · Build pipeline  (emimic py3.11 · `/mnt/ddn/EgoVerse/emimic/bin/python`)

```bash
cd /mnt/ddn/dataset/egoverse_pair_builder
PY=/mnt/ddn/EgoVerse/emimic/bin/python

$PY make_2x_demos.py --glob 'dthdemo_*'            # 400 dthdemo_*__2x
$PY make_2x_demos.py --glob 'taskdemo_*'           #  37 taskdemo_*__2x
$PY 81c_2x_future_pairs_dth_grip_v6.py --no-vel --out-suffix 2x_v6
        # → gr1_dth_pairs_2x_v6.npz (61,686 rows) + dth_pool_2x_v6.npz (server pool, 20-d)
$PY 85c_2x_oneshot_taskdemo_pairs_v6.py            # → gr1_dth_pairs_2x_v6_os.npz (K 5→6)
$PY 82c_2x_build_gr1_v6_manifest.py                # → egoverse_pairs_gr1_v6_2x/{windows,pairs}.npz + t5
cp egoverse_pairs_gr1_v6/{state57,abs_action57,action57,cmd_delta57}_stats* \
   egoverse_pairs_gr1_v6_2x/     # query 'gr1'/target identical; demo 'mimicdroid' pctl差 ≤0.03
```

`egoverse_pairs_gr1_v6_2x`: **windows 75,758** (query gr1 61,686 + demo 14,072, **all `__2x`**;
dth 12,+ taskdemo), queries 61,686, K=6. Query side has **zero `__2x` leak**.

## 5 · Train

| | |
|---|---|
| config | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state` |
| yaml | `kube_yaml/cosmos-gr1-posttrain-v6-2x-448-state-3node.yaml` |
| job / out | `jeongeun-gr1-post-v62x-state-3n` · `/mnt/ddn/tmp/gr1_posttrain_v6_2x_448_state` |

Identical to `..._v6_448_state` (same mid ckpt, 8K schedule) — **only the pool data differs**,
so it is a clean 1× vs 2× A/B.

```bash
kubectl apply -f /mnt/ddn/kube_yaml/cosmos-gr1-posttrain-v6-2x-448-state-3node.yaml
```

---

## 6 · Inference — one-shot eval (2×-dedicated, base files untouched)

Two **new** files; `eval_gr1_server.py` / `gr1_online_retrieval.py` are NOT modified:

| file | what it does |
|---|---|
| `cosmos_policy/scripts/gr1_online_retrieval_2x.py` | no-vel (20-d) `traj_from_state57`/`traj_from_proprio`; `install()` rebinds them on the base module so `episode_windows` (pinned windows) **and** `OnlineRetriever.query` (live query) both go no-vel |
| `cosmos_policy/scripts/eval_gr1_server_2x.py` | subclass `GR1Policy` → one-shot pins the **2× taskdemos** (`taskdemo_<T>_<T>_D_demo_{1,2,3}__2x`); reuses base `main()` with 2× experiment/pairs-dir defaults |

The **client is unchanged** — the standard one-shot-pool3 client just hits the 2× server.

```bash
# ── server (2×) ──
uv run --extra cu128 --group egoverse python -m cosmos_policy.scripts.eval_gr1_server_2x \
    --checkpoint <2x_post_train_ckpt_dir> --port 8700
    # defaults injected: --experiment ..._v6_2x_448_state, --pairs-dir egoverse_pairs_gr1_v6_2x

# ── client (UNCHANGED, ICL one-shot) ──
uv run --extra cu128 --group egoverse python -m cosmos_policy.scripts.eval_gr1_client \
    --server http://127.0.0.1:8700 --oneshot-pool3 \
    --task PnPSinkToCabinet --demo-only demo_1 --episodes 10 --save-video ...
```

Why a dedicated path: the 2× pool was matched **no-vel (20-d)**; the base online descriptor
is **26-d with velocity**, so it is dimension-incompatible with `dth_pool_2x_v6.npz` and would
mis-align against a 2× demo. `install()` makes serve-time matching identical to training.

### ⚠️ `pinned-advance` / `mono_floor` on a 2× demo

`--pinned-advance` controls the pinned-demo pointer:

- **`match`** (default, **use this for 2×**) — unconstrained no-vel re-match each chunk; **no
  frame constants**, so it is 2×-length-agnostic.
- **`mono`** / **`lockstep`** — advance the pointer by FRAME-unit constants calibrated for
  **1× demos**: `--mono-floor 8/16`, a hardcoded band `[last+floor, last+48]`, and lockstep
  `+chunk_size(32)`. A 2× demo is **half as long**, so these step the pointer ~**2× too fast**
  and saturate the demo end early. **Do not reuse 8/16 (or +48/+32) on the 2× pool.** If mono
  is required on 2×, **halve** the constants (floor 4/8, band +24, lockstep +16) — which needs
  a code change; prefer `match`.

---

## Provenance
- Scripts: `dataset/egoverse_pair_builder/{make_2x_demos,render_2x_compare,render_2x_match_episode,81c_2x_*,85c_2x_*,82c_2x_*}.py`
- Videos: `/mnt/ddn/dataset/_viz_samples/{cmp2x,sheet2x,match2x,ep2x}_*`
- 1× counterpart config/data: `..._gr1_ret_v6_448_state` / `egoverse_pairs_gr1_v6`

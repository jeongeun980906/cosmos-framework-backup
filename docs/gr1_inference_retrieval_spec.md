# GR1 closed-loop inference retrieval — SPEC (must match training pairs exactly)

Authoritative contract for the **online demo-retrieval** the GR1 conditioning-policy
runs at eval. The trained model only behaves correctly if the LIVE retrieval produces
the SAME (descriptor, matching, conditioning) as the **training pairs**. The training
pairs are built by `egoverse_pair_builder/75_future_pairs.py` (PAST-window / causal
future-pairs). **This doc is the source of truth; build/verify the inference code to it.**
Do NOT align inference to the deprecated forward matcher (`67_gr1_desc.py` +
`68_gr1_match.py`) — see [[gr1-future-pair-pipeline]] memory and
`docs/egoverse_pairing_v9_log.md`.

> ✅ Scoring reconciled to 75 (2026-06-30): `gr1_online_retrieval.py` `ZW=5.0`, `SHORTLIST=300`.
> ✅ Done 2026-06-30: pairs rebuilt from 75 (t>0) + t=0→TaskDemo (1x+2x); per-frame t+k dataloader (§6b); 67/68 deleted.
> ✅ Done 2026-06-30: **DESCRIPTOR FIX (§6c)** — abs both-hand 6D-pose trajectory matcher `78_future_pairs_abstraj.py`
>   is now canonical (replaces 75 for the GR1 pairs). (Inference `gr1_online_retrieval.py` still on the OLD
>   z-scored desc — remaining train↔inference gap, §6c/§7.)
> ✅ Done 2026-07-01: **GR1 training episodes regenerated as EVAL-HEAVY** (§9) and the full data→pairs pipeline
>   re-run on them (73 convert → sidecams → 78 1x → 77 1x → build → `egoverse_pairs_gr1_1x`, windows=14159,
>   gr1=12475 over 200 eps). Ready to retrain (1x). See §9 for the data-side pipeline.
> ⚠️ Also unwired: server one `--pool-kind` per launch — t=0=taskdemo → t>0=play SWITCH (§2). Defer to post-retrain.

## 0. Constants (from 75_future_pairs.py — the training builder)
| knob | value | meaning |
|---|---|---|
| `WIN` | 32 | past window length (frames @ 20 Hz) used to build the query descriptor |
| `STRIDE` | 4 | boundary step (training); also the earliest boundary (no t=0 boundary) |
| `FUT` | 32 | future/conditioning chunk: demo clip = `action57[w0 : w0+FUT]` |
| `SHORTLIST` | **300** | L2-on-whitened-M shortlist size before DTW rerank |
| `ZW` | **5.0** | weight of the absolute-height (base-rel z) DTW term |
| `GRIP_W` (GW) | 1.0, **0 for `TurnOnFaucet`** | weight of the grip DTW term |
| `DTW_BAND` | 6 | Sakoe-Chiba band |
| `TOPK` | 5 | stored matches per query (≤1 per play demo) |

## 1. Query descriptor (live, from the robot's PAST)
Inputs per frame (robocasa GR1 obs, base/waist-relative):
`robot0_base_to_left_eef_pos/quat`, `robot0_base_to_right_eef_pos/quat`,
gripper `= (clip(actions[:,12|13],-1,1)+1)/2` (L,R), fingers = **zeros** (robot, no finger pose).
Build the src dict exactly as `75.src_dict` / `67.gr1_get` (rotations via `R.from_quat`,
validity = both eef quats finite & nonzero pos).

At boundary frame `e` (the CURRENT frame), the query window is the PAST:
```
pl = min(WIN, e)        # SHORTER than WIN near the episode start
w  = e - pl             # window = [w, e]  (ends at the current frame e)
m.WIN = pl              # temporarily set so the descriptor uses the short window
M, E, G  = blocks(fr, w)      # 01_desc_5s_si.blocks  -> M (flat L2 vec), G (grip seq)
Swrist, _ = seq_parts(fr, w)  # wrist DTW sequence
AB = sabs(fr, w, pl)          # (40,2) abs base-rel z for L,R hands
m.WIN = WIN
```
Descriptor module = `/mnt/ddn/dataset/egoverse_pair_builder/01_desc_5s_si.py` (import it,
do NOT re-implement — byte-identical to the trained pairs). `M` is whitened with the
**play pool's** mean/std: `Mw = (M - pmu)/psd` (pmu,psd from the precomputed play pool).

## 2. t=0 / early-frame handling (IMPORTANT — this is the answer to "how is t=0 handled")
**t=0 is a SPECIAL CASE, force-mapped to a same-task TaskDemo opening (NOT the play pool):**
- The very first query (`t=0`) is FORCED to map to the SAME-task **TaskDemo** clip `[0:FUT=32]`
  (its opening). Built by `77_gr1_t0_taskdemo.py` → `gr1_t0_taskdemo.npz`, merged by
  `build_gr1_posttrain_pairs.py`. Pool = the **12 canonical taskdemos** (3 per task × 4 GR1
  tasks), already converted to egoverse as `taskdemo_<task>_<task>_D_demo_{1,2,3}`. Conditioning
  = `taskdemo action57[0:32]`; GR1 target = its own `action57[0:32]`. This is what the online
  retrieval must do at the first frame (no past yet → condition on a same-task demo's start).
  Verified: 200/200 t=0 GR1 queries map to same-task taskdemo @ w0=0.
- **t>0**: the existing 75 future-pair PAST-window play matching (rest of this doc). Boundaries
  `range(STRIDE,n,STRIDE)` from `e=STRIDE=4`; past `[e-pl,e]`, `pl=min(WIN,e)` (shorter early),
  `m.WIN=pl`. Early GR1 windows (`e<WIN`) restrict the play pool to early-in-subgoal (`prel≤WIN`).
- Online: at the first frame retrieve from the TaskDemo pool (same task, opening). Once motion
  history accumulates (t>0), switch to the play-pool past-window retrieval. The server's
  `--pool-kind taskdemo` already loads the per-task demo pool; ensure it serves the SAME 12
  taskdemos at the start, then the play pool for t>0.

## 3. Play descriptor POOL (precomputed, not live)
Built by `tools/build_play_retrieval_pool.py` over `SinkEnvForPlay` play demos (1x and 2x),
identical windowing (`range(STRIDE,n,STRIDE)`, `pl=min(WIN,e)`). Each play window stores:
`M` (→ whitened `pMw`), `Swrist`, grip `G`, `AB`, `ep` (egoverse name e.g.
`003__SinkEnvForPlay_D_demo_1[__2x]`), `w0=e` (boundary), and the **subgoal** at the window
mid-frame: `(type, src, dst)` from `<demo>/subgoal.json`, plus `prel = e - subgoal_start`.
Whitening stats `pmu,psd = pM.mean(0), pM.std(0)+1e-6` come from the play pool and are applied
to BOTH play and GR1 M.

## 4. Subgoal gating (scenario pool per GR1 task)
`TASK2REQ` maps the GR1 task → required play subgoal `(type, src, dst)`:
- `TurnOnFaucet` → `("faucet_on", None, None)`
- `CloseLeftCabinetDoor` → `("door_close", None, None)`
- `PnPSinkToCabinet` → `("pick_place", "the sink", "the cabinet")`
- `PnPSinkToRightCounterPlate` → `("pick_place", "the sink", ("the plate","the counter"))`
Candidate pool = play windows whose mid-frame subgoal `type` matches and `src`/`dst` match
(dst may be a tuple of allowed values). For early GR1 windows also filter `prel ≤ WIN`.

## 5. Matching (shortlist → DTW rerank → top-K)
```
pool   = task_pool[task]                       # subgoal-gated (+ prel≤WIN if e<WIN)
cand   = pool[ argsort( ||pMw[pool] - gMw_query||_2 )[:SHORTLIST=300] ]
GW     = 0.0 if task == "TurnOnFaucet" else 1.0
score  = dtw(Swrist_q, Swrist_c)               # banded, band=6
       + GW * dtw(grip_q, grip_c)
       + ZW * dtw(AB_q, AB_c)                   # ZW = 5.0   (abs base-rel height)
order  = cand sorted by score ascending
# keep top TOPK=5, at most ONE window per play demo (dedup by ep)
```
For closed-loop conditioning use **top-1** (`order[0]`). The stored pairs keep top-5 for
training sampling.

## 6. Conditioning / target (the FUTURE)
The retrieved match is `(play_ep, m_w0)` where `m_w0` is the play boundary. The demo clip
used to condition the policy is the play's **FUTURE**: `action57[m_w0 : m_w0+FUT=32]` (+ the
matching video frames), read forward by the reader
(`egoverse_zarr_dataset.read_clip(ep, "mimicdroid", m_w0, span=FUT)` → `act[m_w0:m_w0+chunk]`).
The GR1 query target is likewise its own future `action57[e : e+FUT]`. This is the
future-pair setup: **match on PAST, condition/predict the FUTURE**.

NOTE (read length): the reader floors the FUT=32 span to a VAE-legal `4n+1` = **29 video
frames / 28 action steps** (`_frames_from_span` returns `(tgt, tgt-1)`), NOT 32. So 4 frames of
the 32-frame future are unused. To train full 32-step chunks the builder WIN/span would need to
be 36 (floors to 33). Pre-existing; not blocking.

## 6b. Per-frame (t+k) training — dataloader (2026-06-30, DONE for post-train)
75 emits query boundaries at STRIDE=4 (t=0,4,8…); to NOT skip the in-between frames, the
post-train dataloader trains EVERY frame. `egoverse_zarr_dataset.py`: `read_window`/`read_clip`
take `frame_offset=δ` (shifts the single `w0`→`w0+δ` so video, action, state0 all advance in
lockstep); `EgoVersePairedDataset.__getitem__` samples ONE `δ ∈ {0..min(STRIDE-1, room_q, room_c)}`
and passes the SAME δ to BOTH query and demo (so they advance together — at t=1 the condition is
the t=0 demo advanced by 1, target = GR1 `action[1:33]`, etc.). Config knob `per_frame_stride`
(=`_GR1_QUERY_STRIDE=4` for GR1; **default 1 = old behavior, all other paths unaffected**). δ is
clamped against `target_frames` (29 ≥ action 28 → action read always safe). At INFERENCE the
analogue is automatic (the robot is physically at frame t); no special handling needed.

## 6c. DESCRIPTOR FIX — proper trajectory matching (IMPLEMENTED 2026-06-30 → `78_future_pairs_abstraj.py`)
**Canonical GR1 matcher is now `egoverse_pair_builder/78_future_pairs_abstraj.py`** (replaces 75 for
the GR1 pairs). Descriptor = per-frame BOTH-hand ABSOLUTE 6D pose `[Lpos3, L6Drot, Rpos3, R6Drot]`
in METERS over the PAST window `[e-pl,e]`, `pl=min(WIN,e)` — **NO z-score, NO start-relative**
(GR1+play both robocasa → same units, so absolute trajectories compare directly). Match = mean-pose
L2 SHORTLIST(150) → abs-traj **DTW** rerank (band 6, SEQ=24), TOP-5, ≤1 per play demo. Subgoal-gated
(TASK2REQ §4) + `prel≤WIN` for early GR1. OVERWRITES `gr1_future_pairs_{1x,2x}.npz` (75's z-scored
output backed up as `*.75bak.npz`). `build_gr1_posttrain_pairs.py` consumes it unchanged. Verified by
video: consecutive query boundaries → CONSECUTIVE play positions (004@252→256→260; 006@503→…→531),
smooth/forward, NO jump, NO heuristic. ALL 4 tasks × 1x/2x render smooth (`gr1_descfix_{1x,2x}.mp4`).
Run: `78_future_pairs_abstraj.py --play {1x,2x}`. NOTE: inference (`gr1_online_retrieval.py`) still uses
the OLD z-scored `01_desc_5s_si` — it MUST be switched to the same abs-traj descriptor to stay
train↔inference consistent (do NOT edit the inference file per user; track here as the remaining gap).
The human/cup pairing (`01_desc_5s_si` z-scored) is UNTOUCHED — only the GR1 path changed (via 78).

### (historical) why the OLD desc jumped
The current `01_desc_5s_si` wrist seq is START-RELATIVE + per-channel z-scored → drops ABSOLUTE
position AND magnitude (only z height survives weakly via absZ). Consequence (validated by video
`/mnt/ddn/tmp/gr1_review2_*`, `closecab_descfix.mp4`): t>0 matches ignore absolute position
(both hands 16–49cm off, esp. the ACTIVE hand — e.g. CloseLeftCabinetDoor is a LEFT-hand task)
and JUMP discontinuously / backward between boundaries while the robot moves forward.
**Fix = a proper trajectory descriptor (NOT a sticky/monotonic heuristic):** per-frame BOTH-hand
ABSOLUTE 6D pose `[Lpos3, L6Drot, Rpos3, R6Drot]` in meters (GR1+play both robocasa, same units →
NO z-score, NO start-relative), DTW on the absolute trajectory. Prototype proof: consecutive
query boundaries then match CONSECUTIVE play positions (e.g. `004@252→256→260`, `006@503→…→531`),
smooth & forward, with ZERO heuristics. TODO: implement as a GR1 mode in `01_desc_5s_si`
(default-off so the human/cup path is unchanged), rebuild `build_play_retrieval_pool` + `75` +
`build_gr1_posttrain_pairs`, update §1/§5 here, retrain. Until then t>0 matching is low-quality.

## 7. Train↔inference consistency checklist (MUST hold)
- [ ] Same descriptor module (`01_desc_5s_si.py`) + same inputs (base-rel ee_pose L/R + cmd grip, fingers=0).
- [ ] `WIN=32`, `STRIDE=4`, `FUT=32` (NOT the deprecated 67's `WIN=100`).
- [ ] PAST window `[e-pl, e]`, `pl=min(WIN,e)`, `m.WIN=pl` for short windows (causal).
- [ ] Whiten M with the **play-pool** `pmu,psd`.
- [ ] `SHORTLIST=300`, DTW rerank `dtw(wrist) + GW·dtw(grip) + ZW·dtw(absZ)`, **`ZW=5.0`**, `GW=0` for TurnOnFaucet.
- [ ] Subgoal gating via `TASK2REQ`; early windows `prel≤WIN`.
- [ ] Condition on demo FUTURE `[m_w0:m_w0+FUT]`.
- [x] Scoring reconciled in `gr1_online_retrieval.py` (`ZW=5.0`, `SHORTLIST=300`) — 2026-06-30.
- [x] t=0 → same-task TaskDemo opening `[0:32]` (1x + 2x); per-frame t+k dataloader (§6b) — 2026-06-30.
- [ ] **DESCRIPTOR FIX (§6c): abs both-hand 6D-pose trajectory, no z-score — rebuild + retrain.** ← top priority for match quality.
- [ ] Wire the t=0=taskdemo → t>0=play pool SWITCH (server currently one `--pool-kind` per launch).
- [ ] RETRAIN: pairs already REBUILT from 75 at `egoverse_pairs_gr1{,_1x,_2x}` (old ckpts+wandb deleted; ready). Do the descriptor fix FIRST, then retrain once.

## 8. File map
- Training pairs (CANONICAL): **`egoverse_pair_builder/78_future_pairs_abstraj.py`** (t>0, PAST-causal, ABS-TRAJ desc — replaces 75) + `77_gr1_t0_taskdemo.py --speed {1x,2x}` (t=0 → taskdemo) → `tools/build_gr1_posttrain_pairs.py` → `egoverse_pairs_gr1{,_1x,_2x}`. (`75_future_pairs.py` = OLD z-scored matcher, superseded; its output saved as `gr1_future_pairs_*.75bak.npz`.)
- TaskDemo conversion: `tools/convert_taskdemos.py` (1x), `tools/convert_taskdemos_2x.py` (2x = raw [::2] + delta re-derive → `taskdemo_*__2x`).
- Play pool: `tools/build_play_retrieval_pool.py`. Descriptor (SHARED with human path — gate any GR1 change behind a default-off flag): `egoverse_pair_builder/01_desc_5s_si.py`.
- Per-frame dataloader: `cosmos_framework/data/vfm/action/datasets/egoverse_zarr_dataset.py` (`frame_offset`, `per_frame_stride`).
- Inference (DO NOT EDIT here): `cosmos_framework/scripts/gr1_online_retrieval.py`, `action_policy_server_gr1.py`, `eval_gr1_client.py`.
- DELETED (were the forward/non-causal chain): `67_gr1_desc.py`, `68_gr1_match.py` (removed 2026-06-30). Stale data `gr1_play{1x,2x}_match.npz` is dead — do not consume.

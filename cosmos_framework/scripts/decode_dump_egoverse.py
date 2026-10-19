# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage 1 (cosmos-framework .venv): WiLoR-eval decode dump for the EgoVerse policy (Cosmos 3).

For a checkpoint x eval split, generate + decode the GEN/GT query video + gen action,
FK the gen action to 3D head-frame wrists, and dump ONE compressed npz per query window.
The per-sample npz format is the EXACT contract of cosmos_policy's (2.5)
``cosmos_policy/scripts/decode_dump.py``::

    np.savez(out, gen=jpg(gen_frames), gt=jpg(gt_frames), h=H, w=W,
             fkL=(T,3) float32, fkR=(T,3) float32, emb=str, ret=str, idx=int)

so stage 2 (``/mnt/ddn/tmp/wilor_metric.py``, run in wilor_env) is reused VERBATIM and the
resulting SSIM/PSNR/LPIPS/FID/pose_action_corr are directly comparable by construction.
The ``jpg`` (PIL JPEG q=88), ``_vid_uint8`` (truncating byte cast) and FK/de-norm helpers
are line-level ports of the 2.5 code (decode_dump.py + eval_egoverse_ret._fk_skeleton);
fkL/fkR = per-frame HEAD-FRAME wrist positions in cm (== 2.5's ``HR^T (P_world - HP)``).

Query enumeration mirrors 2.5's decode_dump: per-episode time-strided subsample
(--per-episode, default 5 like PER_EPISODE=5 in tools/wilor_eval_all.sh), optional
--episodes cap, then --shard/--num-shards round-robin. The retrieved demo per query is a
seeded-random pick among the stored matches (2.5's dataset picks a random match).

Run (defaults = the v10 3s repro run) — NOTE ``-m``: this dir has a local ``hydra.py``
that would shadow the hydra package if the script were run by file path::

    .venv/bin/python -m cosmos_framework.scripts.decode_dump_egoverse \
        --iter 20000 --out_dir /mnt/ddn/tmp/wilor_eval_cosmos3/iter20000 \
        --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import io
import os
import time

import json
import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.data.generator.action.datasets.egoverse_zarr_dataset import (
    EgoVersePairedDataset,
    EgoVersePairedTransform,
    EgoVerseWindowReader,
)
from cosmos_framework.data.generator.action.action_processing import ActionProcessingRecord
from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint

_D = "cosmos_framework/data/generator/action/datasets"
_DEFAULT_RUN_ROOT = (
    "/mnt/ddn/tmp/cosmos_egoverse_3s_v10/cosmos3_action/action_sft/action_policy_egoverse_3s_repro"
)


# ---------------------------------------------------------------------------
# EXACT ports from cosmos_policy (2.5) — decode_dump.py: jpg(); eval_egoverse_ret.py:
# _vid_uint8(). Keep byte-for-byte identical semantics (JPEG q=88 via PIL; TRUNCATING
# uint8 cast, not round()) so the image metrics see the same quantization.
# ---------------------------------------------------------------------------
def jpg(frames: np.ndarray, q: int = 88) -> np.ndarray:
    out = []
    for f in frames:
        b = io.BytesIO()
        Image.fromarray(f).save(b, "JPEG", quality=q)
        out.append(b.getvalue())
    return np.array(out, object)


def _vid_uint8(x_c_t_h_w: torch.Tensor) -> np.ndarray:
    """(C,T,H,W) in [-1,1] -> (T,H,W,3) uint8 (2.5's truncating cast)."""
    v = ((x_c_t_h_w.float().clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    return np.transpose(v, (1, 2, 3, 0))


# ---------------------------------------------------------------------------
# FK port — same math as 2.5's eval_egoverse_ret._fk_skeleton (== this repo's
# tools/eval_egoverse_generate._fk_skeleton), reduced to what the dump needs:
# de-norm the 57-D action and integrate each wrist in the HEAD frame (cm).
# 2.5's decode_dump stores wl = HR^T (L_P_world - HP_world), and since
# world = head-pose ∘ head-frame, that is exactly the head-frame wrist path P.
# action57/state57 layout: head9 | Leef9 | Lhand15 | Reef9 | Rhand15.
# ---------------------------------------------------------------------------
_SEG = {"L": (slice(9, 12), slice(12, 18)), "R": (slice(33, 36), slice(36, 42))}


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """6-D rotation [col0(3), col1(3)] -> 3x3 matrix (columns) via Gram-Schmidt."""
    c0, c1 = d6[..., :3], d6[..., 3:6]
    b0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1p = c1 - (b0 * c1).sum(-1, keepdims=True) * b0
    b1 = c1p / (np.linalg.norm(c1p, axis=-1, keepdims=True) + 1e-8)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=-1)


def _fk_head_wrists(
    action: np.ndarray, q01: np.ndarray, q99: np.ndarray, state0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """De-normalize (per-emb quantile, the stats the model was trained with) and
    forward-integrate each wrist in the head frame: ``p_{t+1} = p_t + R_t·Δpos``,
    ``R_{t+1} = R_t·ΔR``, anchored at the real ``state0`` head-frame pose. Returns
    (fkL, fkR) each ``(T+1, 3)`` float32 in cm — the 2.5 decode_dump convention."""
    denom = np.clip(q99 - q01, 1e-8, None)
    a = (action[:, :57] + 1.0) / 2.0 * denom + q01  # de-quantile -> native units (m)
    T = a.shape[0]
    M = 100.0  # m -> cm (matches 2.5)
    out = {}
    for side in ("L", "R"):
        ps, rs = _SEG[side]
        P = np.zeros((T + 1, 3))
        R = np.zeros((T + 1, 3, 3))
        P[0] = state0[ps] * M
        R[0] = _rot6d_to_mat(state0[rs])
        eef = a[:, ps.start : ps.start + 9]  # [Δpos(3), Δrot6d(6)]
        for t in range(T):
            P[t + 1] = P[t] + R[t] @ (eef[t, :3] * M)
            R[t + 1] = R[t] @ _rot6d_to_mat(eef[t, 3:9])
        out[side] = P.astype(np.float32)
    return out["L"], out["R"]


def _to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, list):
        return [_to_cuda(v) for v in x]
    return x


def _enumerate_rows(ds, reader, per_episode: int, episodes: int, shard: int, num_shards: int):
    """Per-episode time-strided subsample + episode cap + shard — EXACT mirror of 2.5's
    decode_dump enumeration (and of tools/eval_egoverse_metrics.py's --per-episode path)."""
    from collections import OrderedDict

    N = len(ds._query_idx)
    ep_of_row = [str(reader._ep[int(q)]) for q in ds._query_idx]
    by_ep = OrderedDict()
    for r in range(N):
        by_ep.setdefault(ep_of_row[r], []).append(r)
    K = int(per_episode)
    rows: list[int] = []
    for ridx in by_ep.values():
        if K <= 0 or K >= len(ridx):
            rows.extend(ridx)
        else:
            st = len(ridx) / float(K)
            rows.extend(ridx[int(j * st)] for j in range(K))
    if episodes > 0 and episodes < len(rows):
        step = len(rows) / float(episodes)
        rows = [rows[int(i * step)] for i in range(episodes)]
    return rows[shard::num_shards], len(by_ep)


_ABC_CACHE = {}


def _abc_demo(ds, reader, r, q, ms, mode):
    """A/B/C 셀의 데모 윈도를 고른다 (cosmos_policy eval_egoverse_ret_batch.py --abc 와 동일 정의).

    A 매치 데모 중 같은 embodiment,  B 매치 데모 중 다른 embodiment,
    C 비매치 + 같은 embodiment + 다른 task 무작위.
    저장된 매치 순서가 거리순이라 A/B 는 조건을 만족하는 첫 번째를 쓴다.
    해당 셀이 없는 행은 None -> 호출부에서 건너뛴다."""
    # emb/cls/ep 는 **reader**(EgoVerseWindowReader) 에 있다. ds 에 있는 줄 알고 ds._emb 를
    # 참조했다가 전 행이 AttributeError 로 스킵돼 덤프가 0개 나왔다.
    emb, cls, eps = reader._emb, reader._cls, reader._ep
    if not ms:
        return None
    if mode in ("A", "B"):
        want_same = mode == "A"
        for m in ms:
            if (str(emb[m]) == str(emb[q])) == want_same:
                return m
        return None
    key = str(emb[q])
    if key not in _ABC_CACHE:                       # 같은 emb 풀은 한 번만 만든다
        _ABC_CACHE[key] = np.where(np.asarray([str(x) for x in emb]) == key)[0]
    pool, mset = _ABC_CACHE[key], set(ms)
    rng = np.random.default_rng(10_000 + r)
    for _ in range(200):
        w = int(pool[rng.integers(len(pool))])
        if w not in mset and str(cls[w]) != str(cls[q]) and str(eps[w]) != str(eps[q]):
            return w
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="action_policy_egoverse_3s_nano")
    ap.add_argument("--run-root", default=_DEFAULT_RUN_ROOT,
                    help="training run dir containing checkpoints/iter_<9d> (used with --iter)")
    ap.add_argument("--iter", type=int, default=20000, help="checkpoint iteration under --run-root")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit DCP checkpoint dir (overrides --run-root/--iter)")
    ap.add_argument("--data-root", default="/mnt/ddn/dataset")
    ap.add_argument("--pairs-dir", default="/mnt/ddn/dataset/egoverse_pairs_v10_test")
    ap.add_argument("--norm", default=f"{_D}/egoverse_normalization.json")
    ap.add_argument("--speed", default=f"{_D}/egoverse_speed_stats.json")
    ap.add_argument("--chunk-length", type=int, default=88, help="88 for 3s (89 frames), 148 for 5s")
    # The reader defaults (delta / no state) match the v10 3s repro run. Runs trained with
    # action_source="state" (abs NEXT-state57 chunk, e.g. action_policy_egoverse_v10_icl_edge)
    # MUST pass --action-source state --use-state, else the demo conditioning is handed to the
    # model in the wrong space and the FK below integrates rows that are not deltas.
    ap.add_argument("--action-source", dest="action_source", default="delta",
                    choices=["delta", "abs", "state"])
    ap.add_argument("--use-state", dest="use_state", action="store_true",
                    help="prepend the clean state0 conditioning row (train-time use_state=True)")
    ap.add_argument("--per-episode", dest="per_episode", type=int, default=5,
                    help="query windows per episode (time-strided); matches 2.5 PER_EPISODE=5")
    ap.add_argument("--episodes", type=int, default=0, help="cap total rows (0 = no cap)")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    # A/B/C 셀 선택 (cosmos_policy 의 eval_egoverse_ret_batch.py --abc 와 같은 정의):
    #   A 매치 데모 중 같은 embodiment 로 가장 가까운 것
    #   B 매치 데모 중 다른 embodiment 로 가장 가까운 것
    #   C 비매치 + 같은 embodiment + 다른 task 에서 무작위
    # random(기본) 은 기존 동작(저장된 매치 중 무작위) 그대로 둔다.
    ap.add_argument("--subset", default=None,
                    help="183_fixed_eval_subset.py 가 만든 eval_fixed200.json. "
                         "쿼리+데모 쌍을 고정한다 (--episodes/--per-episode/--demo-mode 무시)")
    ap.add_argument("--demo-mode", dest="demo_mode", default="random",
                    choices=["random", "A", "B", "C"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", dest="num_shards", type=int, default=1)
    ap.add_argument("--vae-path", default="/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth")
    ap.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    args = ap.parse_args()

    ckpt = args.checkpoint or os.path.join(args.run_root, "checkpoints", f"iter_{args.iter:09d}")
    assert os.path.isdir(ckpt), f"checkpoint dir not found: {ckpt}"
    os.makedirs(args.out_dir, exist_ok=True)

    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=ckpt,
        parallelism_config={"data_parallel_shard_degree": 1, "data_parallel_replicate_degree": 1},
        experiment_opts=[f"model.config.tokenizer.vae_path={args.vae_path}"],
    )
    model = model.cuda().eval()
    tokenizer_config = config.model.config.vlm_config.tokenizer

    reader = EgoVerseWindowReader(
        data_root=args.data_root, pairs_dir=args.pairs_dir,
        normalization_path=args.norm, speed_stats_path=args.speed,
        chunk_length=args.chunk_length,
        action_source=args.action_source, use_state=args.use_state,
    )
    transform = EgoVersePairedTransform(
        resolution="256", max_action_dim=config.model.config.max_action_dim,
        share_temporal=True, keep_aspect_ratio=False, tokenizer_config=tokenizer_config,
    )
    ds = EgoVersePairedDataset(reader=reader, pairs_dir=args.pairs_dir, transform=transform)

    subset = None
    if args.subset:
        # 고정 평가 부분집합(183_fixed_eval_subset.py). 쿼리 행과 데모 행을 **둘 다** 파일에서
        # 받아 두 모델(cosmos_policy / edge)이 완전히 같은 조건으로 평가되게 한다.
        # 이게 없으면 --episodes 로 잘린 서로 다른 윈도우를 비교하게 된다(2026-08-14).
        subset = json.load(open(args.subset))
        allr = list(range(len(ds._query_idx)))
        q2r = {int(ds._query_idx[r]): r for r in allr}
        rows, demo_of = [], {}
        for qi, di in zip(subset["query"], subset["demo"]):
            r = q2r.get(int(qi))
            if r is None:
                continue
            rows.append(r); demo_of[r] = int(di)
        rows = rows[args.shard::args.num_shards]
        n_ep = len(set(subset.get("ep", [])))
        print(f"[dump] 고정 부분집합 {args.subset}: {len(subset['query'])}개 중 "
              f"{len(rows)}개가 이 샤드 담당", flush=True)
    else:
        demo_of = {}
        rows, n_ep = _enumerate_rows(ds, reader, args.per_episode, args.episodes,
                                     args.shard, args.num_shards)
    print(f"[dump] shard {args.shard}/{args.num_shards}: {len(rows)} query windows "
          f"({n_ep} episodes, per_ep={args.per_episode}, ckpt={ckpt})", flush=True)

    t0 = time.time()
    done = 0
    for k, r in enumerate(rows):
        try:
            q = int(ds._query_idx[r])
            ms = [int(m) for m in ds._match_idx[r] if m >= 0]
            if r in demo_of:
                m = demo_of[r]                     # 고정 부분집합: 데모까지 파일이 정한다
            elif args.demo_mode == "random":
                # 2.5's dataset picks a RANDOM stored match as the demo; do the same but
                # seeded by the row index so reruns/shards are deterministic.
                m = q if not ms else ms[int(np.random.default_rng(r).integers(len(ms)))]
            else:
                m = _abc_demo(ds, reader, r, q, ms, args.demo_mode)
                if m is None:
                    continue                       # 이 행엔 그 셀이 없다 (예: 교차 emb 매치 없음)
            emb, ret = reader.emb_of(q), reader.emb_of(m)
            sample = transform({"query": reader.read_window(q), "condition": reader.read_window(m)})
            if int(sample["raw_action_dim"][1].item()) != 57:  # FK needs the 57-D hand-pose rig
                continue
            data_batch = {k2: _to_cuda(v) for k2, v in custom_collate_fn([sample]).items()}
            # Upstream action extraction refuses to externalize (unpad) the generated action
            # without a per-sample processing record — the training dataloader attaches one,
            # the manual batch here does not. normalizer=None: we invert the quantile
            # normalization ourselves below with the same q01/q99 the run trained on.
            # (Same fix as scripts/action_policy_server_gr1.py.)
            data_batch["action_processing_record"] = [
                ActionProcessingRecord(raw_action_dim=57, action_normalizer=None)
            ]
            if isinstance(data_batch.get("image_size"), list):  # flatten per FLAT vision item
                data_batch["image_size"] = [
                    it for samp in data_batch["image_size"]
                    for it in (samp if isinstance(samp, list) else [samp])
                ]
            with torch.no_grad():
                result = model.generate_samples_from_batch(
                    data_batch, guidance=args.guidance, seed=[args.seed],
                    n_sample=1, num_steps=args.num_steps, shift=args.shift)
                if not result.get("action") or not result.get("vision"):
                    continue
                gen_video = model.decode(result["vision"][-1])  # (B,C,T,H,W) in [-1,1]
            gen_q = _vid_uint8(gen_video[0])
            gt_q = _vid_uint8(sample["video"][1])  # query GT pixels (demo is item 0)
            gen_a = result["action"][-1].detach().float().cpu().numpy()[:, :57]

            qs = reader._stats[emb]  # SAME per-emb quantile stats the model trained with
            q01, q99 = qs["q01"].numpy()[:57], qs["q99"].numpy()[:57]
            try:
                state0 = reader.read_state0(q)
            except Exception as e:  # noqa: BLE001 — no state zarr -> anchor at origin
                print(f"  [r={r}] read_state0 failed ({e}); anchoring at origin", flush=True)
                state0 = np.zeros(57, np.float32)
            def _wrists(a_norm):
                """rows -> (L,R) head-frame wrist positions in cm.

                action_source="state": each row IS the absolute state57, so the head-frame
                wrist is read straight off it (state57 = head9|Labs9|Lhand15|Rabs9|Rhand15,
                abs9 = [pos, R0, R1]) — NO integration, which would be wrong here.
                Otherwise: forward-integrate the deltas from state0 (original behavior)."""
                if args.action_source != "state":
                    return _fk_head_wrists(a_norm, q01, q99, state0)
                d = np.clip(q99 - q01, 1e-8, None)
                a = (a_norm[:, :57] + 1.0) / 2.0 * d + q01
                return (a[:, 9:12] * 100.0).astype(np.float32), (a[:, 33:36] * 100.0).astype(np.float32)

            wl, wr = _wrists(gen_a)
            # GT twin from the query's own (normalized) chunk -> trajectory comparison is
            # self-contained; row 0 is the clean state0 conditioning row when --use-state.
            gt_a = sample["action"][1].detach().float().cpu().numpy()[:, :57]
            if len(gt_a) == len(gen_a) + 1:
                gt_a = gt_a[1:]
            gl, gr = _wrists(gt_a)

            np.savez(f"{args.out_dir}/s{args.shard}_{k:06d}.npz",
                     gen=jpg(gen_q), gt=jpg(gt_q), h=gen_q.shape[1], w=gen_q.shape[2],
                     fkL=wl, fkR=wr, fkL_gt=gl, fkR_gt=gr,
                     gen_a=gen_a.astype(np.float32), gt_a=gt_a.astype(np.float32),
                     emb=str(emb), ret=str(ret), idx=r)
            done += 1
        except Exception as ex:  # noqa: BLE001 — one bad query shouldn't kill the shard
            print(f"  [skip r={r}] {type(ex).__name__}: {ex}", flush=True)
            continue
        if k % 20 == 0:
            print(f"  {k}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[dump] shard {args.shard} DONE {done}/{len(rows)} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

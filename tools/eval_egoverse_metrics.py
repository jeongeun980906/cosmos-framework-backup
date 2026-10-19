"""Batch offline SHAPE-METRIC eval for the EgoVerse conditioning policy (Cosmos 3 / OmniMoT).

Mirror of cosmos_policy's ``eval_egoverse_ret_batch.py`` but for THIS repo's model+data:
loads the trained policy ONCE, iterates one query per held-out v9-TEST episode, generates
the query action57, de-normalizes gen+GT with the QUERY embodiment's q01/q99 (same per-emb
normalization the reader/skeleton use), and scores gen-vs-GT end-effector motion SHAPE.

Metrics split by scale-sensitivity — the generated ABSOLUTE motion scale can differ from GT
(esp. for held-out embodiments), which must NOT be counted as a shape error, so the PRIMARY
metrics are SCALE-INVARIANT:
  * dtw_shape — DTW after re-scaling each trajectory to unit total path length (pure shape)
  * cos       — per-step Δpos direction cosine (magnitude-free)
  * corr      — Pearson corr of the integrated xyz trajectory (affine-invariant)
SCALE-DEPENDENT reference: dtw_norm, mse; DIAGNOSTIC: scale_ratio (gen/gt |path|). Per hand
(L/R) + head, aggregated to mean/median. Writes eval_metrics[_shard*].csv + _summary.json.

Run (1 GPU) on the held-out TEST split:
  python tools/eval_egoverse_metrics.py --experiment action_policy_egoverse_5s_nano \
      --checkpoint <.../checkpoints/iter_000050000> --chunk-length 148 \
      --pairs-dir /mnt/ddn/dataset/egoverse_pairs_v9_test --out-dir /mnt/ddn/tmp/egoverse_metrics
8-GPU sharded: launch 8 procs with --shard i --num-shards 8 (CUDA_VISIBLE_DEVICES=i), then
one `--merge --num-shards 8` pass to aggregate.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn
from cosmos_framework.data.vfm.action.datasets.egoverse_zarr_dataset import (
    EgoVersePairedDataset,
    EgoVersePairedTransform,
    EgoVerseWindowReader,
)
from cosmos_framework.utils.vfm.model_loader import load_model_from_checkpoint

_D = "cosmos_framework/data/vfm/action/datasets"

# action57 Δpos blocks (head9 | Leef9 | Lhand15 | Reef9 | Rhand15): head=0:3, L-eef=9:12, R-eef=33:36
_BLOCKS = {"head": slice(0, 3), "L": slice(9, 12), "R": slice(33, 36)}


# ----------------------------- metrics (scale-invariant primary) -----------------------------
def _banded_dtw(a: np.ndarray, b: np.ndarray, band: int = 8) -> float:
    """Sakoe-Chiba banded DTW between two 3-D point sequences (Euclidean local cost)."""
    n, m = len(a), len(b)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF, np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = max(1, i - band), min(m, i + band)
        ai = a[i - 1]
        for j in range(jlo, jhi + 1):
            c = float(np.linalg.norm(ai - b[j - 1]))
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def _shape_metrics(dgt: np.ndarray, dgen: np.ndarray, band: int = 8) -> dict:
    """dgt/dgen: (T,3) per-step Δpos of one eef/head. Scale-invariant (dtw_shape/cos/corr),
    TIMING-robust (cos_lag best-lag cosine, spec_cos |FFT| cosine, DTW band widened to ±25%
    of length), + scale-dependent ref (dtw_norm/mse) and scale_ratio diagnostic."""
    T = len(dgt)
    band = max(band, T // 4)                                  # TIMING-robust: warp up to ±25% of length
    tgt = np.cumsum(dgt, 0)
    tgen = np.cumsum(dgen, 0)
    plen_gt = float(np.linalg.norm(dgt, axis=1).sum())
    plen_gen = float(np.linalg.norm(dgen, axis=1).sum())
    dtw = _banded_dtw(tgt, tgen, band=band)
    dtw_norm = dtw / (plen_gt + plen_gen + 1e-6)
    sgt = np.cumsum(dgt / (plen_gt + 1e-9), 0)
    sgen = np.cumsum(dgen / (plen_gen + 1e-9), 0)
    dtw_shape = _banded_dtw(sgt, sgen, band=band)
    scale_ratio = float(plen_gen / (plen_gt + 1e-9))
    ng = np.linalg.norm(dgt, axis=1)
    nn = np.linalg.norm(dgen, axis=1)
    msk = (ng > 1e-6) & (nn > 1e-6)
    cos = float(((dgt[msk] * dgen[msk]).sum(1) / (ng[msk] * nn[msk])).mean()) if msk.any() else float("nan")
    L = max(1, T // 4)                                        # best-lag direction cosine (±25%)
    best = -2.0
    for lag in range(-L, L + 1):
        a, b = (dgt[lag:], dgen[:T - lag]) if lag >= 0 else (dgt[:T + lag], dgen[-lag:])
        na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
        m2 = (na > 1e-6) & (nb > 1e-6)
        if m2.sum() >= 3:
            best = max(best, float(((a[m2] * b[m2]).sum(1) / (na[m2] * nb[m2])).mean()))
    cos_lag = best if best > -2 else float("nan")
    sc = []                                                   # spectral |FFT| cosine (phase-invariant)
    for d in range(3):
        fa = np.abs(np.fft.rfft(dgt[:, d])); fb = np.abs(np.fft.rfft(dgen[:, d]))
        na_, nb_ = np.linalg.norm(fa), np.linalg.norm(fb)
        if na_ > 1e-9 and nb_ > 1e-9:
            sc.append(float((fa * fb).sum() / (na_ * nb_)))
    spec_cos = float(np.mean(sc)) if sc else float("nan")
    corrs = []
    for d in range(3):
        x, y = tgt[:, d], tgen[:, d]
        if x.std() > 1e-9 and y.std() > 1e-9:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
    corr = float(np.mean(corrs)) if corrs else float("nan")
    mse = float(((dgt - dgen) ** 2).mean())
    return {"dtw_shape": dtw_shape, "cos": cos, "cos_lag": cos_lag, "spec_cos": spec_cos, "corr": corr,
            "dtw_norm": dtw_norm, "mse": mse, "scale_ratio": scale_ratio}


def _agg_by_episode(rows, col, kind=None):
    """EPISODE-LEVEL macro-average: per episode mean over ALL its evals (windows × demos),
    then mean/median over episodes (equal weight per episode). `kind` filters same/cross."""
    from collections import defaultdict
    by_ep = defaultdict(list)
    for r in rows:
        if kind is not None and r.get("kind") != kind:
            continue
        try:
            v = float(r.get(col))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            by_ep[r.get("episode", "?")].append(v)
    ep_means = np.asarray([float(np.mean(vs)) for vs in by_ep.values() if vs], np.float64)
    if ep_means.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n_ep": 0, "n_eval": 0}
    return {"mean": float(ep_means.mean()), "median": float(np.median(ep_means)),
            "n_ep": int(ep_means.size), "n_eval": int(sum(len(v) for v in by_ep.values()))}


# metric families: PHYSICAL (de-norm meters) + NORMALIZED ([-1,1], stats-free) + timing + ref
_MET_LR = ("dtws", "cos", "cosl", "spec", "dtw", "sr", "dtwsn", "cosn")
_MET_ALL = ("dtws", "cos", "cosl", "spec", "corr", "dtw", "mse", "sr", "dtwsn", "cosn", "corrn")
_COLS = (["idx", "episode", "emb", "ret_emb", "kind"]
         + [f"{m}_{k}" for m in ("dtws", "cos", "corr", "cosl", "spec") for k in ("head", "L", "R")]
         + [f"{m}_LR" for m in ("dtws", "cos", "cosl", "spec")]
         + [f"{m}_{k}" for m in ("dtwsn", "cosn", "corrn") for k in ("head", "L", "R")]
         + [f"{m}_LR" for m in ("dtwsn", "cosn")]
         + [f"{m}_{k}" for m in ("dtw", "mse", "sr") for k in ("head", "L", "R")]
         + ["dtw_LR", "sr_LR"])


def _print_summary(summary: dict, header: str) -> None:
    g = lambda k: summary.get(k, {}).get("mean", float("nan"))
    ne = summary.get("dtws_L", {}).get("n_ep", "?"); nv = summary.get("dtws_L", {}).get("n_eval", "?")
    print(f"\n==== {header}  (episodes={ne}, evals={nv}) ====", flush=True)
    print("  -- PHYSICAL space (meters) --")
    print(f"  DTW_shape↓: L={g('dtws_L'):.2f} R={g('dtws_R'):.2f}   cos↑: L={g('cos_L'):+.3f} R={g('cos_R'):+.3f}   corr↑: L={g('corr_L'):+.3f} R={g('corr_R'):+.3f}")
    print("  -- NORMALIZED space ([-1,1], stats-free) --")
    print(f"  DTW_shape↓: L={g('dtwsn_L'):.2f} R={g('dtwsn_R'):.2f}   cos↑: L={g('cosn_L'):+.3f} R={g('cosn_R'):+.3f}   corr↑: L={g('corrn_L'):+.3f} R={g('corrn_R'):+.3f}")
    print("  -- TIMING-robust (physical) --")
    print(f"  best-lag cos↑: L={g('cosl_L'):+.3f} R={g('cosl_R'):+.3f}   spectral cos↑: L={g('spec_L'):+.3f} R={g('spec_R'):+.3f}")
    print(f"  scale_ratio(med) L={summary.get('sr_L',{}).get('median',float('nan')):.2f} R={summary.get('sr_R',{}).get('median',float('nan')):.2f}   (mean unreliable: near-0 GT)")


def _summarize(rows: list[dict]) -> dict:
    summary = {"n_samples": len(rows)}
    for metric in _MET_ALL:
        for key in ("head", "L", "R"):
            summary[f"{metric}_{key}"] = _agg_by_episode(rows, f"{metric}_{key}")
            for kd in ("same", "cross"):
                summary[f"{metric}_{key}__{kd}"] = _agg_by_episode(rows, f"{metric}_{key}", kind=kd)
    for m in _MET_LR:
        summary[f"{m}_LR"] = _agg_by_episode(rows, f"{m}_LR")
    return summary


def _merge(out_dir: str, num_shards: int) -> None:
    rows = []
    _of = f"of{num_shards}" if num_shards > 0 else "of*"
    for fp in sorted(glob.glob(os.path.join(out_dir, f"eval_metrics_shard*{_of}.csv"))):
        with open(fp) as f:
            rows.extend(list(csv.DictReader(f)))
    comb = os.path.join(out_dir, "eval_metrics.csv")
    with open(comb, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else _COLS)
        w.writeheader(); w.writerows(rows)
    summary = _summarize(rows)
    summary["n_shards"] = num_shards
    json.dump(summary, open(os.path.join(out_dir, "eval_metrics_summary.json"), "w"), indent=2)
    _print_summary(summary, "SHAPE-METRIC SUMMARY (merged, episode-macro)")
    print(f"  -> {comb}", flush=True)


def _to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, list):
        return [_to_cuda(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="action_policy_egoverse_5s_nano")
    ap.add_argument("--checkpoint", default=None, help="trained DCP checkpoint dir (required unless --merge)")
    ap.add_argument("--data-root", default="/mnt/ddn/dataset")
    ap.add_argument("--pairs-dir", default="/mnt/ddn/dataset/egoverse_pairs_v9_test")
    ap.add_argument("--norm", default=f"{_D}/egoverse_normalization.json")
    ap.add_argument("--speed", default=f"{_D}/egoverse_speed_stats.json")
    ap.add_argument("--chunk-length", type=int, default=148)
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtw-band", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=0, help="cap #episodes (0=all held-out); even stride")
    ap.add_argument("--per-episode", dest="per_episode", type=int, default=0,
                    help="windows/episode: 0=ALL (every ~1s window); K>0=K time-strided")
    ap.add_argument("--topk", type=int, default=1,
                    help="demos/query = same-top-k + cross-top-k (matches stored best-first); 0=ALL")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="merge shard CSVs -> summary and exit")
    ap.add_argument("--vae-path", default="/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth")
    ap.add_argument("--out-dir", default="/mnt/ddn/tmp/egoverse_metrics")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.merge:
        _merge(args.out_dir, args.num_shards)
        return
    assert args.checkpoint, "--checkpoint required (unless --merge)"

    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint,
        parallelism_config={"data_parallel_shard_degree": 1, "data_parallel_replicate_degree": 1},
        experiment_opts=[f"model.config.tokenizer.vae_path={args.vae_path}"],
    )
    model = model.cuda().eval()
    tokenizer_config = config.model.config.vlm_config.tokenizer

    reader = EgoVerseWindowReader(
        data_root=args.data_root, pairs_dir=args.pairs_dir,
        normalization_path=args.norm, speed_stats_path=args.speed,
        chunk_length=args.chunk_length,
    )
    transform = EgoVersePairedTransform(
        resolution="256", max_action_dim=config.model.config.max_action_dim,
        share_temporal=True, keep_aspect_ratio=False, tokenizer_config=tokenizer_config,
    )
    ds = EgoVersePairedDataset(reader=reader, pairs_dir=args.pairs_dir, transform=transform)

    # ---- selection: EVERY query window (per_episode=0), or K time-strided per episode ----
    from collections import OrderedDict
    emb_arr = reader._emb                                       # embodiment per WINDOW index
    ep_of_row = np.array([str(reader._ep[int(q)]) for q in ds._query_idx])
    if int(args.per_episode) <= 0:
        rows = list(range(len(ds._query_idx)))
    else:
        by_ep = OrderedDict()
        for r in range(len(ds._query_idx)):
            by_ep.setdefault(ep_of_row[r], []).append(r)
        K = max(1, int(args.per_episode)); rows = []
        for ridx in by_ep.values():
            rows.extend(ridx if K >= len(ridx) else [ridx[int(j * len(ridx) / K)] for j in range(K)])
    if args.episodes and 0 < args.episodes < len(rows):
        rows = [rows[int(i * len(rows) / args.episodes)] for i in range(args.episodes)]
    total = len(rows)
    if args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
    print(f"[metrics] query_rows={len(ds._query_idx)} total={total} this_shard={len(rows)} "
          f"(shard {args.shard}/{args.num_shards}, topk={args.topk})", flush=True)

    def _demos(r):
        """same-top-k + cross-top-k retrieval demos (best-first), labeled same/cross."""
        q = int(ds._query_idx[r]); e = str(emb_arr[q])
        ms = [int(m) for m in ds._match_idx[r] if m >= 0]
        same = [m for m in ms if str(emb_arr[m]) == e]
        cross = [m for m in ms if str(emb_arr[m]) != e]
        if args.topk > 0:
            same, cross = same[:args.topk], cross[:args.topk]
        return q, ([(m, "same") for m in same] + [(m, "cross") for m in cross]) or [(q, "self")]

    per_sample, t0 = [], time.time()
    for i, r in enumerate(rows):
        try:
            q, demos = _demos(r)
            emb = reader.emb_of(q)
            qs = reader._stats[emb]
            q01 = qs["q01"].numpy()[:57]; q99 = qs["q99"].numpy()[:57]
            denom = np.clip(q99 - q01, 1e-8, None)
            de = lambda a: (a + 1.0) / 2.0 * denom + q01           # per-emb q01/q99 de-norm
            for m_win, kind in demos:
                sample = transform({"query": reader.read_window(q), "condition": reader.read_window(m_win)})
                data_batch = {k: _to_cuda(v) for k, v in custom_collate_fn([sample]).items()}
                if isinstance(data_batch.get("image_size"), list):
                    data_batch["image_size"] = [
                        it for samp in data_batch["image_size"] for it in (samp if isinstance(samp, list) else [samp])]
                if int(sample["raw_action_dim"][1].item()) != 57:   # metrics need the 57-D rig
                    continue
                with torch.no_grad():
                    result = model.generate_samples_from_batch(
                        data_batch, guidance=args.guidance, seed=[args.seed],
                        n_sample=1, num_steps=args.num_steps, shift=args.shift)
                if not result.get("action"):
                    continue
                gen_n = result["action"][-1].detach().float().cpu().numpy()[:, :57]   # NORMALIZED
                gt_n = sample["action"][1].detach().float().cpu().numpy()[:, :57]
                gt_a, gen_a = de(gt_n), de(gen_n)                                     # PHYSICAL
                row = {"idx": r, "episode": str(reader._ep[q]), "emb": emb,
                       "ret_emb": str(emb_arr[m_win]), "kind": kind}
                for key in ("head", "L", "R"):
                    sl = _BLOCKS[key]
                    p = _shape_metrics(gt_a[:, sl], gen_a[:, sl], band=args.dtw_band)  # physical
                    row[f"dtws_{key}"] = p["dtw_shape"]; row[f"cos_{key}"] = p["cos"]; row[f"corr_{key}"] = p["corr"]
                    row[f"cosl_{key}"] = p["cos_lag"]; row[f"spec_{key}"] = p["spec_cos"]
                    row[f"dtw_{key}"] = p["dtw_norm"]; row[f"mse_{key}"] = p["mse"]; row[f"sr_{key}"] = p["scale_ratio"]
                    nrm = _shape_metrics(gt_n[:, sl], gen_n[:, sl], band=args.dtw_band)  # normalized
                    row[f"dtwsn_{key}"] = nrm["dtw_shape"]; row[f"cosn_{key}"] = nrm["cos"]; row[f"corrn_{key}"] = nrm["corr"]
                for mm in _MET_LR:
                    row[f"{mm}_LR"] = float(np.nanmean([row[f"{mm}_L"], row[f"{mm}_R"]]))
                per_sample.append(row)
        except Exception as ex:  # noqa: BLE001 — one bad query shouldn't kill the sweep
            print(f"  [skip r={r}] {type(ex).__name__}: {ex}", flush=True)
            continue
        if (i + 1) % 5 == 0 or (i + 1) == len(rows):
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)} windows, {len(per_sample)} evals  "
                  f"cosn_LR={np.nanmean([x['cosn_LR'] for x in per_sample]):+.3f}  ({el:.0f}s)", flush=True)

    suffix = f"_shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    csv_fp = os.path.join(args.out_dir, f"eval_metrics{suffix}.csv")
    with open(csv_fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_sample)
    print(f"[metrics] wrote {len(per_sample)} rows -> {csv_fp}", flush=True)
    if args.num_shards > 1:
        return
    summary = _summarize(per_sample)
    json.dump(summary, open(os.path.join(args.out_dir, "eval_metrics_summary.json"), "w"), indent=2)
    _print_summary(summary, "SHAPE-METRIC SUMMARY (v9 TEST)")


if __name__ == "__main__":
    main()

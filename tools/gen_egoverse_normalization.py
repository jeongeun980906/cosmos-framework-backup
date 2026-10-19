# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate per-embodiment action/state normalization stats for EgoVerse.

EgoVerse stores the egomimic-processed unified ``action{D}/action`` and
``state{D}/state`` arrays in each raw episode zarr (``D=57`` for the human rigs
MECKA/scale/aria, ``D=20`` for the ``eva`` bimanual robot). The values are
already bounded to roughly ``[-1, 1]`` but are **not per-dim normalized** — some
dims barely use the negative range and a few are near-constant — so a single
global normalize would squash most channels. This tool computes per-embodiment,
per-dim stats so the policy can map each channel cleanly into ``[-1, 1]``.

For each embodiment we emit two blocks — ``"<emb>"`` (action) and
``"<emb>_state"`` (state) — each with ``min / max / mean / std / q01 / q99`` in
the exact format consumed by
:func:`cosmos_framework.data.vfm.action.action_normalization.load_action_stats`
(``stats_key=<block>``) and ``normalize_action`` (``"minmax"`` /
``"quantile"`` / ``"meanstd"``). The same file is passed to the eval server so
train-time normalization and eval-time denormalization stay identical.

``min/max/mean/std`` are exact (streaming over every frame). ``q01/q99`` are
computed from a per-embodiment frame subsample (``--quantile-sample`` frames,
reservoir-free uniform stride) — robust enough for the recommended
``quantile`` method while keeping memory bounded.

Usage::

    python tools/gen_egoverse_normalization.py \\
        --data-root /mnt/ddn/dataset \\
        --pairs-dir /mnt/ddn/dataset/egoverse_pairs_v5 \\
        --out cosmos_framework/data/vfm/action/datasets/egoverse_normalization.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

# emb tag -> (raw-data root dir name, unified action/state dim).
# MECKA/scale/aria are the human rigs (hand_pose, 57D); eva is the bimanual robot (20D).
EMB_REGISTRY: dict[str, dict] = {
    "MECKA": {"root": "egoverse", "dim": 57},
    "scale": {"root": "egoverse", "dim": 57},
    "aria": {"root": "egoverse", "dim": 57},
    "mimicdroid": {"root": "egoverse", "dim": 57},   # robocasa/MimicDroid → action57, under egoverse/
    "gr1": {"root": "egoverse", "dim": 57},          # GR1 humanoid → action57, under egoverse/
    "eva": {"root": "egoverse_eva", "dim": 20},
}


class _StreamStats:
    """Exact streaming min/max/mean/std + a strided frame subsample for quantiles."""

    def __init__(self, dim: int, quantile_sample: int) -> None:
        self.dim = dim
        self.n = 0
        self.vmin = np.full(dim, np.inf, dtype=np.float64)
        self.vmax = np.full(dim, -np.inf, dtype=np.float64)
        self.vsum = np.zeros(dim, dtype=np.float64)
        self.vsumsq = np.zeros(dim, dtype=np.float64)
        self._q_cap = quantile_sample
        self._q_buf: list[np.ndarray] = []
        self._q_rows = 0

    def update(self, arr: np.ndarray) -> None:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, self.dim)
        if a.size == 0:
            return
        self.n += a.shape[0]
        self.vmin = np.minimum(self.vmin, a.min(0))
        self.vmax = np.maximum(self.vmax, a.max(0))
        self.vsum += a.sum(0)
        self.vsumsq += (a * a).sum(0)
        # Keep a bounded uniform subsample for quantiles (every k-th frame).
        if self._q_rows < self._q_cap:
            step = max(1, a.shape[0] // 256)
            self._q_buf.append(a[::step].astype(np.float32))
            self._q_rows += self._q_buf[-1].shape[0]

    def finalize(self) -> dict[str, list[float]]:
        mean = self.vsum / max(self.n, 1)
        var = np.maximum(self.vsumsq / max(self.n, 1) - mean**2, 0.0)
        std = np.sqrt(var)
        if self._q_buf:
            q = np.concatenate(self._q_buf, axis=0)
            q01 = np.quantile(q, 0.01, axis=0)
            q99 = np.quantile(q, 0.99, axis=0)
        else:  # pragma: no cover - empty embodiment
            q01, q99 = self.vmin, self.vmax
        return {
            "min": _round(self.vmin),
            "max": _round(self.vmax),
            "mean": _round(mean),
            "std": _round(std),
            "q01": _round(q01),
            "q99": _round(q99),
        }


def _round(v: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in v]


def _episodes_for_emb(pairs_dir: Path, emb: str, max_eps: int | None) -> list[str]:
    """Unique episode keys for *emb*, taken from the v5 windows index."""
    w = np.load(pairs_dir / "windows.npz", allow_pickle=True)
    eps = np.unique(w["ep"][w["emb"] == emb]).tolist()
    if max_eps is not None:
        eps = eps[:max_eps]
    return eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path, help="Parent of egoverse/ and egoverse_eva/.")
    ap.add_argument("--pairs-dir", required=True, type=Path, help="egoverse_pairs_v5 dir (for windows.npz ep list).")
    ap.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    ap.add_argument("--embodiments", nargs="+", default=list(EMB_REGISTRY), help="Subset of embodiments.")
    ap.add_argument("--max-episodes-per-emb", type=int, default=None, help="Cap episodes scanned per emb (smoke).")
    ap.add_argument("--quantile-sample", type=int, default=300_000, help="Frames kept per emb for q01/q99.")
    args = ap.parse_args()

    payload: dict[str, dict] = {}
    for emb in args.embodiments:
        spec = EMB_REGISTRY[emb]
        root, dim = spec["root"], spec["dim"]
        eps = _episodes_for_emb(args.pairs_dir, emb, args.max_episodes_per_emb)
        act = _StreamStats(dim, args.quantile_sample)
        sta = _StreamStats(dim, args.quantile_sample)
        n_ok = 0
        for ep in eps:
            d = args.data_root / root / ep
            try:
                act.update(np.asarray(zarr.open(f"{d}/action{dim}/action", mode="r")[:]))
                sta.update(np.asarray(zarr.open(f"{d}/state{dim}/state", mode="r")[:]))
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — skip unreadable episodes, keep scanning
                print(f"  [skip] {emb}/{ep}: {type(e).__name__}: {str(e)[:80]}")
        print(f"{emb}: {n_ok}/{len(eps)} eps, {act.n} action frames, {sta.n} state frames (dim={dim})")
        payload[emb] = act.finalize()
        payload[f"{emb}_state"] = sta.finalize()

    # MERGE into any existing stats file: only the computed embodiments' blocks are
    # updated, every other block is preserved. (A plain overwrite here previously
    # wiped the whole file when run for a single embodiment.)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}
    if args.out.exists():
        merged = json.loads(args.out.read_text())
    merged.update(payload)
    args.out.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"Wrote {args.out}  ({len(payload)} new/updated blocks, {len(merged)} total)")


if __name__ == "__main__":
    main()

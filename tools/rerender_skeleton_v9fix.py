"""Re-render a saved egoverse_gen skeleton from its *_action.npz (gen/gt/state0) using the
CURRENT (head-integrating) _fk_skeleton — NO model / NO GPU. Proves the head-pose fix: the
old *_skeleton.mp4 (pre-fix) pinned the head at state0; this re-render moves it per predicted
head9 deltas. Usage: python tools/rerender_skeleton_v9fix.py aria2MECKA_527034 [more labels...]"""
import sys, numpy as np
from pathlib import Path
from tools.eval_egoverse_generate import _save_skeleton_video, EgoVerseWindowReader

GEN = Path("/mnt/ddn/tmp/egoverse_gen")
_D = "cosmos_framework/data/vfm/action/datasets"
reader = EgoVerseWindowReader(
    data_root="/mnt/ddn/dataset", pairs_dir="/mnt/ddn/dataset/egoverse_pairs_v9",
    normalization_path=f"{_D}/egoverse_normalization.json",
    speed_stats_path=f"{_D}/egoverse_speed_stats.json", chunk_length=148,
)
labels = sys.argv[1:] or ["aria2MECKA_527034"]
for label in labels:
    widx = int(label.rsplit("_", 1)[1])
    npz = np.load(GEN / f"{label}_action.npz")
    gen, gt, state0 = npz["gen"], npz["gt"], npz["state0"]
    q_emb = reader.emb_of(widx)
    qs = reader._stats[q_emb]; q01 = qs["q01"].numpy(); q99 = qs["q99"].numpy()
    # head9 dpos motion sanity (denormalized cm)
    denom = np.clip(q99 - q01, 1e-8, None)
    ga = (gen[:, :57] + 1.0) / 2.0 * denom + q01; gh = ga[:, 0:3]
    hp = np.cumsum(gh, 0) * 100.0
    print(f"[{label}] q_emb={q_emb} gen head path span(cm)={np.round(hp.max(0)-hp.min(0),2)}", flush=True)
    _save_skeleton_video(
        gen, gt, q01, q99, state0,
        GEN / f"{label}_skeleton.mp4", GEN / f"{label}_skeleton.png",
        caption=f"[{label}] v9fix", frame="world", emb=q_emb,
    )
    print(f"[{label}] re-rendered -> {label}_skeleton.mp4", flush=True)

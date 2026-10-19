# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Report-style offline generation for the EgoVerse conditioning policy.

Renders ONE retrieval pair (a forced demo + query window) into a single
``*_report.mp4`` matching the ``eval_*_<demo>2<query>_report.mp4`` layout:

  * Top strip  : retrieved [DEMO] | query [QUERY] GT | query [QUERY] GEN videos.
  * Bottom pair: 3D action quiver — GT action vs generated action (normalized,
    cumulative dim 0-2 trajectory + current-frame arrow), animated per frame.
  * Suptitle   : task text + ``[demo] -> [query]`` + frame counter.

Unlike ``eval_egoverse_generate.py`` (which samples conditions at random), the
demo and query windows are pinned via ``--query-widx`` / ``--demo-widx`` so a
specific cross-embodiment example (e.g. aria cup demo -> scale) can be shown.

    python tools/eval_egoverse_report.py \\
        --experiment action_policy_egoverse_5s_nano \\
        --checkpoint .../checkpoints/iter_000010000 --chunk-length 148 \\
        --query-widx 98376 --demo-widx 65731 \\
        --model-tag cosmos3_nano --idx 0 \\
        --out-dir /mnt/ddn/dataset/_viz_samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn
from cosmos_framework.data.vfm.action.datasets.egoverse_zarr_dataset import (
    EgoVersePairedTransform,
    EgoVerseWindowReader,
)
from cosmos_framework.utils.vfm.model_loader import load_model_from_checkpoint

_D = "cosmos_framework/data/vfm/action/datasets"

# --- LEFT / RIGHT hand translation dim indices in the 57-D egoverse action ----
# The 57-D action is an egomimic delta-pose vector laid out as repeating pose
# blocks; per-step delta-translation triples were identified empirically by
# correlating each candidate action triple against np.diff(...xyz) of the raw
# component pose arrays (left/right.obs_ee_pose, *_wrist_pose, obs_head_pose) in
# the episode zarrs. Best-linear-fit R^2 over 6 episodes (scale + MECKA):
#   act[0:3]   -> obs_head_pose         (head translation)
#   act[8:11]  -> left.obs_ee_pose      (6/6 episodes, meanR2 .62)  -> LEFT hand
#   act[33:36] -> right.obs_ee_pose     (6/6 episodes, meanR2 .76)  -> RIGHT hand
# Other triples (4-6, 13-15, 37-39, ...) are head/quaternion/keypoint dims with
# low/ambiguous R^2. The dataset loader quantile-normalizes the action but
# preserves column indices, so these same indices slice the normalized actions.
LEFT_XYZ = (8, 9, 10)
RIGHT_XYZ = (33, 34, 35)
LEFT_COLOR = "tab:blue"
RIGHT_COLOR = "tab:orange"


def _to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, list):
        return [_to_cuda(e) for e in x]
    return x


def _to_uint8_thwc(video_cthw: torch.Tensor) -> np.ndarray:
    """(C,T,H,W) in [-1,1] → (T,H,W,3) uint8."""
    v = video_cthw.detach().float().clamp(-1, 1).cpu()
    v = ((v + 1.0) * 127.5).round().to(torch.uint8)
    return v.permute(1, 2, 3, 0).numpy()


def _quiver_one(ax, traj: np.ndarray, t: int, color: str, label: str) -> None:
    """Draw one cumulative 3D trajectory up to frame t + current-step arrow."""
    pts = traj[: t + 1]
    if len(pts) >= 2:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=1.2, alpha=0.5)
    p = traj[t]
    # current motion direction = delta into this frame
    p0 = traj[t - 1] if t > 0 else np.zeros(3)
    ax.quiver(p0[0], p0[1], p0[2], p[0] - p0[0], p[1] - p0[1], p[2] - p0[2],
              color=color, lw=2.0, arrow_length_ratio=0.3, label=label)
    ax.scatter([p[0]], [p[1]], [p[2]], color=color, s=18)


def _quiver_axes(ax, trajs: dict[str, tuple[np.ndarray, str]], t: int, title: str, lim) -> None:
    """Draw multiple cumulative 3D trajectories (e.g. left + right hand) on ``ax``.

    ``trajs`` maps a legend label -> ``(traj (T,3), color)``.
    """
    ax.clear()
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.tick_params(labelsize=6)
    for label, (traj, color) in trajs.items():
        _quiver_one(ax, traj, t, color, label)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="action_policy_egoverse_5s_nano")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/mnt/ddn/dataset")
    ap.add_argument("--pairs-dir", default="/mnt/ddn/dataset/egoverse_pairs_v9_test")  # offline eval = held-out TEST split only
    ap.add_argument("--norm", default=f"{_D}/egoverse_normalization.json")
    ap.add_argument("--speed", default=f"{_D}/egoverse_speed_stats.json")
    ap.add_argument("--chunk-length", type=int, default=148)
    ap.add_argument("--query-widx", type=int, required=True, help="window index used as the GENERATION target")
    ap.add_argument("--demo-widx", type=int, required=True, help="window index used as the DEMO condition")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-tag", default="cosmos3_nano", help="model name for the output filename")
    ap.add_argument("--idx", type=int, default=0, help="sample index label in the filename")
    ap.add_argument("--out-dir", default="/mnt/ddn/dataset/_viz_samples")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--vae-path", default="/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    # Pinned pair (forced demo + query) — NOT the random-condition dataset path.
    q_raw = reader.read_window(args.query_widx)
    c_raw = reader.read_window(args.demo_widx)
    demo_emb, query_emb = c_raw["emb"], q_raw["emb"]
    task_txt = q_raw["ai_caption"].split(". speed")[0]
    sample = transform({"query": q_raw, "condition": c_raw})

    data_batch = {k: _to_cuda(v) for k, v in custom_collate_fn([sample]).items()}
    if isinstance(data_batch.get("image_size"), list):
        data_batch["image_size"] = [
            it for samp in data_batch["image_size"] for it in (samp if isinstance(samp, list) else [samp])
        ]

    with torch.no_grad():
        result = model.generate_samples_from_batch(
            data_batch, guidance=args.guidance, seed=[args.seed],
            n_sample=1, num_steps=args.num_steps, shift=args.shift,
        )

    gen_video = model.decode(result["vision"][-1])
    gen_thwc = _to_uint8_thwc(gen_video[0])
    demo_thwc = _to_uint8_thwc(sample["video"][0])
    gt_thwc = _to_uint8_thwc(sample["video"][1])

    # Action: cumulative 3D LEFT- and RIGHT-hand translation trajectories in
    # normalized space (LEFT_XYZ / RIGHT_XYZ dims identified by correlation).
    gen_action = result["action"][-1].detach().float().cpu().numpy()
    gt_action = sample["action"][1].detach().float().cpu().numpy()
    gt_l = np.cumsum(gt_action[:, LEFT_XYZ], axis=0)
    gt_r = np.cumsum(gt_action[:, RIGHT_XYZ], axis=0)
    gen_l = np.cumsum(gen_action[:, LEFT_XYZ], axis=0)
    gen_r = np.cumsum(gen_action[:, RIGHT_XYZ], axis=0)
    lim = float(max(np.abs(gt_l).max(), np.abs(gt_r).max(),
                    np.abs(gen_l).max(), np.abs(gen_r).max(), 1e-3)) * 1.1

    T = min(demo_thwc.shape[0], gt_thwc.shape[0], gen_thwc.shape[0],
            gt_l.shape[0], gen_l.shape[0])

    # --- Compact renderer: PIL video strip (native res, no axis margins) stacked
    #     on a tight two-panel 3D action figure. Both are W=3*panel_w wide. --------
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager

    ph, pw = demo_thwc.shape[1], demo_thwc.shape[2]  # per-panel H,W
    W = pw * 3
    TITLE_H = 22  # main title band above the strip
    try:
        _fp = font_manager.findfont("DejaVu Sans")
        font = ImageFont.truetype(_fp, 13); font_lab = ImageFont.truetype(_fp, 12)
    except Exception:
        font = ImageFont.load_default(); font_lab = font

    def _make_top(t: int) -> np.ndarray:
        strip = np.concatenate([demo_thwc[t], gt_thwc[t], gen_thwc[t]], axis=1)  # (ph, W, 3)
        img = Image.new("RGB", (W, ph + TITLE_H), (12, 12, 16))
        img.paste(Image.fromarray(strip), (0, TITLE_H))
        dr = ImageDraw.Draw(img)
        dr.text((6, 4), f"task: {task_txt} [{query_emb}]   retrieved [{demo_emb}] → "
                        f"query [{query_emb}]   frame {t + 1}/{T}", fill=(240, 220, 90), font=font)
        for k, lab in enumerate((f"retrieved [{demo_emb}]", f"query [{query_emb}] GT",
                                 f"query [{query_emb}] GEN")):
            dr.text((k * pw + 5, TITLE_H + 3), lab, fill=(70, 235, 90), font=font_lab)
        return np.asarray(img)

    # Bottom: tight 3D action figure, width matched to the strip.
    BOT_H = max(280, W // 3)
    figB = plt.figure(figsize=(W / 100.0, BOT_H / 100.0), dpi=100)
    ax_a_gt = figB.add_subplot(1, 2, 1, projection="3d")
    ax_a_gen = figB.add_subplot(1, 2, 2, projection="3d")
    figB.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.06)
    canvasB = FigureCanvasAgg(figB)

    def _make_bot(t: int) -> np.ndarray:
        _quiver_axes(ax_a_gt, {"L": (gt_l, LEFT_COLOR), "R": (gt_r, RIGHT_COLOR)},
                     t, "GT action (L/R hands)", lim)
        _quiver_axes(ax_a_gen, {"L": (gen_l, LEFT_COLOR), "R": (gen_r, RIGHT_COLOR)},
                     t, "gen action (L/R hands)", lim)
        canvasB.draw()
        b = np.asarray(canvasB.buffer_rgba())[..., :3]
        return b[:, :W].copy() if b.shape[1] >= W else np.pad(b, ((0, 0), (0, W - b.shape[1]), (0, 0)))

    frames = [np.concatenate([_make_top(t), _make_bot(t)], axis=0) for t in range(T)]
    plt.close(figB)

    name = f"eval_{args.model_tag}_egoverse_ret_5s_idx{args.idx}_{demo_emb}2{query_emb}_report.mp4"
    imageio.mimwrite(str(out / name), frames, fps=args.fps, macro_block_size=1)
    np.savez(out / name.replace("_report.mp4", "_actions.npz"),
             gen=gen_action, gt=gt_action, raw_dim=57,
             left_xyz=np.array(LEFT_XYZ), right_xyz=np.array(RIGHT_XYZ))
    print(f"saved {out / name}  ({len(frames)} frames, {frames[0].shape})")
    print(f"  demo[{demo_emb}] widx={args.demo_widx} '{c_raw['ai_caption'][:60]}'")
    print(f"  query[{query_emb}] widx={args.query_widx} '{q_raw['ai_caption'][:60]}'")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Closed-loop PushT evaluation client for the Cosmos3 action-policy server.

Simulator-side counterpart to ``cosmos_framework.scripts.action_policy_server_libero``
— which is domain-generic, so the SAME server serves PushT when launched with
``--raw-action-dim 2 --action-chunk-size 8 --fps 10`` and a PushT stats JSON.
The client and server run as separate processes (the client needs only
``gym-pusht`` + numpy + requests, NOT torch):

    [gym-pusht sim + this client]  --POST /predict {image, prompt}-->  [policy server]
                                   <--{action: [[2], ...], video}-----

PushT is a planar pushing task: the policy outputs 2D ``[pos_x, pos_y]`` target
positions (the server denormalizes them back to PushT pixel coordinates using the
same min/max the policy was trained with — pass it
``pusht_lerobot_normalization.json`` via ``--action-stats-path``). The client
executes the action chunk open-loop and scores an episode as a success when the
T-block coverage reaches ``--success-threshold`` (the standard gym-pusht / LeRobot
PushT success criterion).

Example::

    # 1) start the (domain-generic) policy server on a GPU:
    PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_libero \\
        --checkpoint-path /path/to/pusht-train-output/model \\
        --action-stats-path cosmos_framework/data/generator/action/datasets/pusht_lerobot_normalization.json \\
        --action-normalization minmax --raw-action-dim 2 \\
        --action-chunk-size 8 --fps 10 --port 8000

    # 2) run this client (in a venv with gym-pusht):
    python -m cosmos_framework.scripts.eval_pusht_client \\
        --server-url http://127.0.0.1:8000 \\
        --num-episodes 50 --num-open-loop-steps 8 \\
        --out-dir outputs/pusht_eval
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

_PUSHT_PROMPT = "push the T block to the target"


def _encode_png_b64(img_hwc_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_hwc_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_policy(server_url: str, image_hwc_uint8: np.ndarray, image_size: int) -> np.ndarray:
    payload = {
        "image": _encode_png_b64(image_hwc_uint8),
        "prompt": _PUSHT_PROMPT,
        "domain_name": "pusht",
        "image_size": int(image_size),
    }
    resp = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    action = np.asarray(data.get("action", []), dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.size == 0:
        raise RuntimeError(f"Policy server returned empty action: {data.get('error', '<no error field>')}")
    return action[:, :2]  # [chunk, 2]


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)


def run(args: argparse.Namespace) -> dict:
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the env)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=args.image_size,
        observation_height=args.image_size,
    )

    successes = 0
    coverages = []
    out_dir = Path(args.out_dir)

    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        t = 0
        max_coverage = 0.0
        action_queue: list[np.ndarray] = []
        frames: list[np.ndarray] = []

        while not done and t < args.max_steps:
            img = np.ascontiguousarray(obs["pixels"], dtype=np.uint8)
            if len(frames) < args.max_video_frames:
                frames.append(img)

            if not action_queue:
                chunk = query_policy(args.server_url, img, args.image_size)
                action_queue = list(chunk[: args.num_open_loop_steps])

            action = action_queue.pop(0)
            obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            # gym-pusht reward IS the normalized block coverage in [0, 1].
            max_coverage = max(max_coverage, float(reward))
            done = bool(terminated or truncated)
            t += 1

        success = max_coverage >= args.success_threshold
        successes += int(success)
        coverages.append(max_coverage)
        if args.save_videos and (ep < args.save_first_n_videos or not success):
            _save_video(frames, out_dir / f"ep{ep}_{'success' if success else 'fail'}_cov{max_coverage:.2f}.mp4")
        print(
            f"[pusht] ep {ep+1}/{args.num_episodes} -> {'SUCCESS' if success else 'fail'} "
            f"max_coverage={max_coverage:.3f} (running {successes}/{ep+1} = {successes/(ep+1):.1%})",
            flush=True,
        )

    env.close()
    return {
        "num_episodes": args.num_episodes,
        "successes": successes,
        "success_rate": successes / args.num_episodes if args.num_episodes else 0.0,
        "mean_max_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "success_threshold": args.success_threshold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-url", default="http://127.0.0.1:8000")
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--num-open-loop-steps", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=96, help="Render size & server image_size (gym-pusht native 96).")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--success-threshold", type=float, default=0.95, help="Min T-block coverage counted as success.")
    ap.add_argument("--out-dir", default="outputs/pusht_eval")
    ap.add_argument("--save-videos", action="store_true", default=True)
    ap.add_argument("--no-save-videos", dest="save_videos", action="store_false")
    ap.add_argument("--save-first-n-videos", type=int, default=3)
    ap.add_argument("--max-video-frames", type=int, default=400)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    result = run(args)
    result["timestamp"] = stamp
    result["server_url"] = args.server_url
    report_path = out_dir / f"pusht_eval_{stamp}.json"
    report_path.write_text(json.dumps(result, indent=2))

    print("\n========== PUSHT EVAL SUMMARY ==========")
    print(f"  success_rate     : {result['success_rate']:.1%}  ({result['successes']}/{result['num_episodes']})")
    print(f"  mean max coverage: {result['mean_max_coverage']:.3f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

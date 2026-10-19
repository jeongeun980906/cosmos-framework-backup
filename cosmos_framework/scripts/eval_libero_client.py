# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Closed-loop LIBERO evaluation client for the Cosmos3 action-policy server.

This is the simulator-side counterpart to
``cosmos_framework.scripts.action_policy_server_libero`` (the HTTP policy
server). The two run as separate processes (and may live in separate virtualenvs
— the client needs only the LIBERO simulator + numpy + requests, NOT torch or
the model), talking over HTTP:

    [LIBERO sim + this client]  --POST /predict {image, prompt}-->  [policy server]
                                <--{action: [[7], ...], video}-----

For each task suite (``libero_spatial``, ``libero_object``, ``libero_goal``,
``libero_10``) it rolls out ``num_trials_per_task`` episodes per task, executing
the policy's 7D delta-EE action chunks open-loop, and reports the success rate.
Conventions (NUM_STEPS_WAIT, per-suite max steps, vertical image flip, dummy
no-op action, success = env ``done``) mirror the OpenVLA-OFT / Cosmos-Policy
LIBERO harness so numbers are comparable.

The policy server already DENORMALIZES actions (pass it the same
``libero_lerobot_normalization.json`` via ``--action-stats-path``), so the 7D
action this client receives is in the dataset's raw action space
``[dpos(3), daxang(3), gripper]``. The only remaining transform is the gripper
convention between the LeRobot dataset (gripper ∈ [0, 1]) and the robosuite
OSC_POSE controller the LIBERO env expects (gripper ∈ [-1, 1], −1 = open). See
``--gripper-map``; inspect the first-episode rollout video to confirm grasping
behaves, and flip the mapping if the gripper acts inverted.

Requires the LIBERO simulator (see docs/action_policy_eval.md):
    MUJOCO_GL=egl  # headless rendering

Example::

    # 1) start the policy server (in the cosmos-framework venv, on a GPU):
    PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_libero \\
        --checkpoint-path /path/to/train-output/model \\
        --action-stats-path cosmos_framework/data/generator/action/datasets/libero_lerobot_normalization.json \\
        --action-chunk-size 16 --raw-action-dim 7 --fps 20 --port 8000

    # 2) run this client (in the libero-client venv):
    MUJOCO_GL=egl python -m cosmos_framework.scripts.eval_libero_client \\
        --server-url http://127.0.0.1:8000 \\
        --task-suites libero_spatial libero_object libero_goal libero_10 \\
        --num-trials-per-task 50 --num-open-loop-steps 8 \\
        --out-dir outputs/libero_eval
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

# --- LIBERO conventions (match OpenVLA-OFT / Cosmos-Policy run_libero_eval) ---
NUM_STEPS_WAIT = 10  # let objects settle before the policy acts
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # no-op (gripper open)
TASK_MAX_STEPS = {  # match LeRobot official LiberoEnv.TASK_SUITE_MAX_STEPS
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


# ---------------------------------------------------------------------------
# Server I/O
# ---------------------------------------------------------------------------


def _encode_png_b64(img_hwc_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_hwc_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """robosuite-convention quaternion (x, y, z, w) → axis-angle (3,). Matches the
    nvidia/LIBERO_LeRobot_v3 ``observation.state`` axis-angle channels."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-8)
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    axis = q[:3] / den
    return (axis * angle).astype(np.float32)


def obs_to_state7(obs: dict) -> np.ndarray:
    """Build the 7D proprio state ``[eef_pos(3), axisangle(3), gripper]`` from a
    LIBERO/robosuite obs — the same channels as ``observation.state[:7]`` the policy
    was trained to condition on (use_state). gripper = first finger qpos."""
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)[:3]
    aa = _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
    grip = float(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[0])
    return np.concatenate([pos, aa, [grip]]).astype(np.float32)  # [7]


def _apply_flip(im: np.ndarray, mode: str) -> np.ndarray:
    """mode: none / ud (vertical) / lr (horizontal) / 180 (both)."""
    if mode == "ud":
        im = im[::-1, :]
    elif mode == "lr":
        im = im[:, ::-1]
    elif mode == "180":
        im = im[::-1, ::-1]
    elif mode != "none":
        raise ValueError(f"bad flip mode {mode!r}")
    return np.ascontiguousarray(im, dtype=np.uint8)


def build_policy_image(obs: dict, flip_mode: str, two_cam: bool) -> np.ndarray:
    """Build the policy frame. ``two_cam`` vertically concats agent-view (top) +
    eye-in-hand wrist (bottom) into (2H, W, 3), matching training. ``flip_mode``
    is applied to EACH camera to match the dataset orientation."""
    agent = _apply_flip(np.asarray(obs["agentview_image"], dtype=np.uint8), flip_mode)
    if not two_cam:
        return agent
    wrist = _apply_flip(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8), flip_mode)
    return np.ascontiguousarray(np.concatenate([agent, wrist], axis=0), dtype=np.uint8)  # (2H, W, 3)


def query_policy(
    server_url: str, image_hwc_uint8: np.ndarray, prompt: str, image_size: int, state: np.ndarray | None = None
) -> np.ndarray:
    """POST one observation to the policy server, return the (chunk, 7) action array."""
    payload = {
        "image": _encode_png_b64(image_hwc_uint8),
        "prompt": prompt,
        "domain_name": "libero",
        "image_size": int(image_size),
    }
    if state is not None:
        # use_state server conditions on the current absolute proprio state[:7].
        payload["state"] = [float(x) for x in np.asarray(state).reshape(-1)[:7]]
    resp = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    action = np.asarray(data.get("action", []), dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.size == 0:
        raise RuntimeError(f"Policy server returned empty action: {data.get('error', '<no error field>')}")
    return action  # [chunk, 7]


def map_gripper(action_chunk: np.ndarray, mode: str) -> np.ndarray:
    """Map the policy gripper (dataset space) to the robosuite env gripper space.

    The LeRobot LIBERO dataset stores gripper ∈ [0, 1]; the robosuite OSC_POSE
    controller expects gripper ∈ [-1, 1] with −1 = open, +1 = close.
      * ``rescale`` (default): env_g = 2*g − 1   (dataset 0→−1 open, 1→+1 close)
      * ``identity``: pass through unchanged (dataset already in env space)
      * ``invert``: env_g = 1 − 2*g            (flip if grasping is inverted)
    """
    out = action_chunk.copy()
    g = out[:, 6]
    if mode == "rescale":
        out[:, 6] = 2.0 * g - 1.0
    elif mode == "invert":
        out[:, 6] = 1.0 - 2.0 * g
    elif mode != "identity":
        raise ValueError(f"Unknown gripper map: {mode!r}")
    return np.clip(out, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


@dataclass
class SuiteResult:
    suite: str
    episodes: int = 0
    successes: int = 0
    per_task: dict[str, list[int]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)


def run_suite(args: argparse.Namespace, suite_name: str) -> SuiteResult:
    # Imported here so `--help` and code review work without the simulator installed.
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    n_tasks = suite.n_tasks
    max_steps = TASK_MAX_STEPS[suite_name]
    result = SuiteResult(suite=suite_name)
    out_dir = Path(args.out_dir) / suite_name

    # Optional task-id slice (for splitting a suite across GPUs/workers).
    task_list = list(range(n_tasks)) if not args.task_ids else [t for t in args.task_ids if 0 <= t < n_tasks]
    for task_id in task_list:
        task = suite.get_task(task_id)
        task_desc = task.language
        env, _ = _make_env(task, args.resolution)
        init_states = suite.get_task_init_states(task_id)
        result.per_task[task_desc] = []

        for ep in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            t, done, success = 0, False, False
            action_queue: list[np.ndarray] = []
            frames: list[np.ndarray] = []

            while t < max_steps + NUM_STEPS_WAIT:
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                    t += 1
                    continue

                # 180° flip (both axes) — raw mujoco is upside-down vs the dataset
                # (arm top). two_cam → agent(top)+wrist(bottom) concat like training.
                img = build_policy_image(obs, args.flip_mode, args.two_cam)
                if len(frames) < args.max_video_frames:
                    frames.append(img)

                if not action_queue:
                    state = obs_to_state7(obs) if args.use_state else None
                    chunk = query_policy(args.server_url, img, task_desc, args.image_size, state)
                    chunk = map_gripper(chunk, args.gripper_map)
                    action_queue = list(chunk[: args.num_open_loop_steps])

                action = action_queue.pop(0)
                obs, _, done, _ = env.step(action.tolist())
                t += 1
                if done:
                    success = True
                    break

            result.episodes += 1
            result.successes += int(success)
            result.per_task[task_desc].append(int(success))
            if args.save_videos and (ep < args.save_first_n_videos or not success):
                tag = f"task{task_id}_ep{ep}_{'success' if success else 'fail'}"
                _save_video(frames, out_dir / f"{tag}.mp4")
            print(
                f"[{suite_name}] task {task_id+1}/{n_tasks} ep {ep+1}/{args.num_trials_per_task} "
                f"-> {'SUCCESS' if success else 'fail'} "
                f"(running {result.successes}/{result.episodes} = {result.success_rate:.1%})",
                flush=True,
            )
        env.close()

    return result


def _make_env(task: Any, resolution: int):
    import os as _os

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = _os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    # Match LeRobot official LiberoEnv relative control: the 7D action is a DELTA
    # EE command (OSC_POSE), so force use_delta=True rather than rely on the env
    # default. (OSC_POSE defaults to delta, but the official sets it explicitly.)
    for robot in env.robots:
        robot.controller.use_delta = True
    return env, task.language


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-url", default="http://127.0.0.1:8000", help="Policy server base URL.")
    ap.add_argument("--task-suites", nargs="+", default=list(DEFAULT_SUITES), help="LIBERO suites to evaluate.")
    ap.add_argument("--task-ids", type=int, nargs="*", default=None,
                    help="Restrict to these task indices within each suite (for splitting work across GPUs).")
    ap.add_argument("--num-trials-per-task", type=int, default=50)
    ap.add_argument("--num-open-loop-steps", type=int, default=8, help="Actions executed per policy query (<= chunk).")
    ap.add_argument("--resolution", type=int, default=256, help="Per-camera env render size.")
    ap.add_argument("--image-size", type=int, default=256,
                    help="Server image_size. Single-view: 256. Two-cam concat (512x256): pass 512 so "
                    "the server skips its height pre-resize and buckets the stack to training's 320x192.")
    ap.add_argument("--two-cam", action="store_true", default=False,
                    help="Concat agent-view (top) + eye-in-hand wrist (bottom) — the lerobot/libero 2-cam recipe.")
    ap.add_argument("--flip-mode", choices=["none", "ud", "lr", "180"], default="180",
                    help="Per-camera flip to match dataset orientation. nvidia/LIBERO_v3=180; "
                    "lerobot/libero may be 'ud' (raw env is upside-down but not L-R mirrored).")
    ap.add_argument(
        "--gripper-map",
        choices=["rescale", "identity", "invert"],
        default="invert",
        help="Map policy gripper [0,1] to env gripper [-1,1]. 'invert' (1-2g) is correct "
        "for nvidia/LIBERO_LeRobot_v3 — verified by a solved rollout (rescale grasps inverted).",
    )
    ap.add_argument(
        "--use-state",
        action="store_true",
        default=True,
        help="Send the current proprio state[:7] so a use_state-trained policy conditions on it "
        "(the server must run with --use-state). Matches the use_state=True training recipe.",
    )
    ap.add_argument("--no-use-state", dest="use_state", action="store_false")
    ap.add_argument("--out-dir", default="outputs/libero_eval")
    ap.add_argument("--save-videos", action="store_true", default=True)
    ap.add_argument("--no-save-videos", dest="save_videos", action="store_false")
    ap.add_argument("--save-first-n-videos", type=int, default=2, help="Always save the first N episodes per task.")
    ap.add_argument("--max-video-frames", type=int, default=600)
    args = ap.parse_args()

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"  # default to headless rendering

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results: dict[str, SuiteResult] = {}
    for suite_name in args.task_suites:
        if suite_name not in TASK_MAX_STEPS:
            raise SystemExit(f"Unknown suite {suite_name!r}; choices: {sorted(TASK_MAX_STEPS)}")
        print(f"\n=== Evaluating {suite_name} ===", flush=True)
        results[suite_name] = run_suite(args, suite_name)

    # Summary + JSON report.
    summary = {
        "timestamp": stamp,
        "server_url": args.server_url,
        "num_trials_per_task": args.num_trials_per_task,
        "num_open_loop_steps": args.num_open_loop_steps,
        "gripper_map": args.gripper_map,
        "suites": {
            s: {
                "success_rate": r.success_rate,
                "successes": r.successes,
                "episodes": r.episodes,
                "per_task": {k: (sum(v), len(v)) for k, v in r.per_task.items()},
            }
            for s, r in results.items()
        },
    }
    overall_ep = sum(r.episodes for r in results.values())
    overall_succ = sum(r.successes for r in results.values())
    summary["overall"] = {
        "success_rate": overall_succ / overall_ep if overall_ep else 0.0,
        "successes": overall_succ,
        "episodes": overall_ep,
    }
    report_path = out_dir / f"libero_eval_{stamp}.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print("\n========== LIBERO EVAL SUMMARY ==========")
    for s, r in results.items():
        print(f"  {s:16s}: {r.success_rate:6.1%}  ({r.successes}/{r.episodes})")
    print(f"  {'OVERALL':16s}: {summary['overall']['success_rate']:6.1%}  ({overall_succ}/{overall_ep})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

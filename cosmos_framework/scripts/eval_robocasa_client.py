# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Closed-loop RoboCasa (24-task kitchen) evaluation client for the Cosmos3 action-policy server.

Simulator-side counterpart to ``cosmos_framework.scripts.action_policy_server_libero``
(which is domain-generic — it reads ``domain_name`` per request). The two run as
separate processes, and normally separate virtualenvs: this client needs only
robosuite/robocasa + numpy + requests, NOT torch or the model.

    [RoboCasa sim + this client]  --POST /predict {image, prompt, state}-->  [policy server]
                                  <--{action: [[7], ...]}------------------

Rolls out ``--num-trials-per-task`` episodes per task and reports success rate.
The env/scene conventions (PandaMobile robot, ``obj_instance_split=B`` held-out
objects, the 5 test layout/style scenes selected deterministically by episode
index, ``NUM_STEPS_WAIT=10``, per-task max steps, success = ``env._check_success()``)
mirror the Cosmos-Policy ``run_robocasa_eval.py`` harness so numbers are comparable.

Observation contract — must match ``RoboCasaHDF5Dataset`` exactly:

  * **Frame**: the three cameras vertically concatenated in the order
    ``[agentview_left, agentview_right, eye_in_hand]``, each flipped vertically
    (the env renders upside-down; the dataset was recorded with ``np.flipud``).
    At ``--env-img-res 224`` that is a 672x224 stack, so start the server with
    ``--resolution 480`` — otherwise it derives the tier from the shape, gets
    ``"256"``, and silently feeds the policy a different canvas than training.
  * **Prompt**: ``CAPTION_PREFIX + env.get_ep_meta()["lang"]`` — the same string the
    dataset builds from the demo's ``task_description`` attribute, and nothing else.
    The recipe turns every optional appender off (viewpoint / duration-fps /
    resolution / idle-frames), so start the server with ``--append-duration-fps False
    --append-resolution-info False`` or it will append sentences training never had.
    (Unlike the T5-conditioned Cosmos-Policy runs, an unseen ``B``-split phrasing is
    not a cache miss here: Cosmos3-Edge conditions on the VLM prompt directly.)
  * **State** (``--use-state``, on by default): ``[eef_pos(3), axisangle(3),
    gripper_qpos[0]]``, exactly the channels the dataset reads from
    ``obs/ee_pos`` / ``obs/ee_ori`` / ``obs/gripper_states``.

The server denormalizes the action (pass it the same
``robocasa_hdf5_normalization.json`` via ``--action-stats-path``
``--action-stats-key robocasa``), so what arrives here is already the raw 7D
RoboCasa command ``[dpos(3), daxang(3), gripper]`` the OSC_POSE controller
consumes — including the gripper, which RoboCasa stores in env space ``[-1, 1]``
already (hence ``--gripper-map identity``; LIBERO needs a rescale, RoboCasa does
not). The env takes 12 dims, so a no-op mobile-base suffix ``[0,0,0,0,-1]`` is
appended.

Example::

    # 1) policy server (cosmos-framework venv, on a GPU):
    PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_libero \\
        --experiment action_policy_robocasa_edge \\
        --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \\
        --checkpoint-path /mnt/ddn/tmp/robocasa24_edge/.../checkpoints/iter_000020000 \\
        --action-stats-path cosmos_framework/data/generator/action/datasets/robocasa_hdf5_normalization.json \\
        --action-stats-key robocasa --action-normalization minmax \\
        --use-state --state-stats-key robocasa_state \\
        --append-duration-fps False --append-resolution-info False \\
        --resolution 480 --raw-action-dim 7 --action-chunk-size 32 --fps 20 --port 8000

    # 2) this client (a venv with robocasa installed, e.g. /mnt/ddn/cosmos_policy/.venv):
    MUJOCO_GL=egl PYTHONPATH=/mnt/ddn/cosmos-framework \\
      /mnt/ddn/cosmos_policy/.venv/bin/python -m cosmos_framework.scripts.eval_robocasa_client \\
        --server-url http://127.0.0.1:8000 \\
        --num-trials-per-task 50 --num-open-loop-steps 16 \\
        --out-dir /mnt/ddn/tmp/robocasa24_edge_eval
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

# --- RoboCasa conventions (match Cosmos-Policy run_robocasa_eval.py) ---------
NUM_STEPS_WAIT = 10  # let objects settle before the policy acts
MOBILE_BASE_NOOP = (0.0, 0.0, 0.0, 0.0, -1.0)  # env action is 12D = 7 arm + 5 base

# Cameras, top-to-bottom in the concatenated frame. MUST match
# ``robocasa_hdf5_dataset.DEFAULT_CAMERAS`` (obs keys drop the ``_rgb`` suffix and
# gain ``_image`` in the live env).
CAMERA_OBS_KEYS = (
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
)
# MUST match ``robocasa_hdf5_dataset.CAPTION_PREFIX`` (duplicated rather than
# imported so this client stays torch-free).
CAPTION_PREFIX = "Top: left third-person camera. Middle: right third-person camera. Bottom: wrist camera. Task: "

# The OSC_POSE controller config the Cosmos-Policy RoboCasa evals use (identical to
# ``robosuite.load_controller_config(default_controller="OSC_POSE")`` defaults;
# inlined so this client needs no cross-repo pickle). Override with
# ``--controller-config-pkl`` for byte-exact parity with a specific harness.
DEFAULT_CONTROLLER_CONFIG: dict[str, Any] = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping_ratio": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_ratio_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}

TASK_MAX_STEPS: dict[str, int] = {
    # Pick and place
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    # Doors
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    # Drawers
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    # Stove
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    # Sink
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    # Coffee
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    # Microwave
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}
ALL_TASKS = tuple(TASK_MAX_STEPS.keys())  # the 24 tasks of the release
DEFAULT_LAYOUT_STYLE_IDS = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))  # 5 test scenes


# ---------------------------------------------------------------------------
# Observation -> request
# ---------------------------------------------------------------------------


def _encode_png_b64(img_hwc_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_hwc_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """robosuite quaternion (x, y, z, w) -> axis-angle (3,).

    Reproduces ``robosuite.utils.transform_utils.quat2axisangle``, which is what
    wrote ``obs/ee_ori`` into the dataset (regenerate_robocasa_dataset.py:490).
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-8)
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] / den) * (2.0 * np.arccos(w))).astype(np.float32)


def obs_to_state7(obs: dict) -> np.ndarray:
    """``[eef_pos(3), axisangle(3), gripper]`` — the dataset's ``use_state`` row.

    Mirrors ``RoboCasaHDF5Dataset._read_state``: ``obs/ee_pos`` = ``robot0_eef_pos``,
    ``obs/ee_ori`` = ``quat2axisangle(robot0_eef_quat)``, and the gripper channel is
    the FIRST finger qpos (``obs/gripper_states[0]``), dropping the duplicate.
    """
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)[:3]
    aa = _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
    grip = float(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[0])
    return np.concatenate([pos, aa, [grip]]).astype(np.float32)


def build_policy_image(obs: dict, flip: bool) -> np.ndarray:
    """The 3-camera vertical stack the policy was trained on -> ``(3H, W, 3)`` uint8.

    ``flip`` applies ``np.flipud`` per camera: the env renders upside-down and the
    dataset was recorded flipped (regenerate_robocasa_dataset.py), so leaving it off
    hands the policy inverted scenes.
    """
    views = []
    for key in CAMERA_OBS_KEYS:
        if key not in obs:
            raise KeyError(f"Env obs is missing {key!r}; got {sorted(k for k in obs if k.endswith('_image'))}")
        im = np.asarray(obs[key], dtype=np.uint8)
        views.append(np.flipud(im) if flip else im)
    return np.ascontiguousarray(np.concatenate(views, axis=0), dtype=np.uint8)


def query_policy(
    server_url: str,
    image_hwc_uint8: np.ndarray,
    prompt: str,
    image_size: int,
    state: np.ndarray | None,
    timeout: float = 600.0,
    want_generated_video: bool = False,
    domain_name: str = "robocasa",
) -> np.ndarray:
    """POST one observation, return the ``(chunk, 7)`` denormalized action array.

    Uses ``/predict_batch`` with a single item, NOT ``/predict``. Both run the same
    diffusion forward and return byte-identical actions, but ``/predict`` also VAE-
    decodes the 33-frame generated video and base64s it back — which this client
    throws away. Measured on one H100 at ``num_steps=15``: 3.60 s vs 0.61 s per
    request, a **5.9x** difference, and eval wall-clock is entirely request-bound.
    Pass ``want_generated_video`` to take the slow path when you actually want to look
    at what the world model produced (``--save-videos`` does NOT need it — that saves
    the env's own frames).
    """
    payload: dict[str, Any] = {
        "image": _encode_png_b64(image_hwc_uint8),
        "prompt": prompt,
        "domain_name": domain_name,
        "image_size": int(image_size),
    }
    if state is not None:
        payload["state"] = [float(x) for x in np.asarray(state).reshape(-1)[:7]]
    base = server_url.rstrip("/")
    if want_generated_video:
        resp = requests.post(f"{base}/predict", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("action", [])
    else:
        resp = requests.post(f"{base}/predict_batch", json={"items": [payload]}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        raw = (data.get("actions") or [[]])[0]
    action = np.asarray(raw, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.size == 0:
        raise RuntimeError(f"Policy server returned empty action: {data.get('error', '<no error field>')}")
    return action


def map_gripper(action_chunk: np.ndarray, mode: str) -> np.ndarray:
    """Map the policy gripper channel into the env's ``[-1, 1]`` (-1 = open).

    RoboCasa stores the raw env command, so ``identity`` (the default) is correct.
    The other modes exist for datasets that store ``[0, 1]`` (LIBERO/LeRobot) or if
    the gripper turns out inverted in the rollout videos.
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
class TaskResult:
    task: str
    episodes: int = 0
    successes: int = 0
    lengths: list[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


def _make_env(args: argparse.Namespace, task: str, seed: int, episode_idx: int):
    """One RoboCasa env, matching the Cosmos-Policy eval kwargs.

    The scene is picked deterministically from the 5 test layout/style pairs by
    ``episode_idx // 10`` so a given trial index always lands in the same scene.
    """
    import robocasa  # noqa: F401 — importing it REGISTERS the kitchen envs with robosuite
    import robosuite

    if args.controller_config_pkl:
        with open(args.controller_config_pkl, "rb") as f:
            controller_configs = pickle.load(f)
    else:
        controller_configs = DEFAULT_CONTROLLER_CONFIG

    scenes = DEFAULT_LAYOUT_STYLE_IDS
    layout_and_style_ids = (scenes[(episode_idx // 10) % len(scenes)],)
    env = robosuite.make(
        env_name=task,
        robots=args.robots,
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=args.env_img_res,
        camera_heights=args.env_img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=args.obj_instance_split,
        generative_textures=None,
        randomize_cameras=args.randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    return env


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)


def run_task(args: argparse.Namespace, task: str) -> TaskResult:
    max_steps = args.max_steps if args.max_steps > 0 else TASK_MAX_STEPS.get(task, 500)
    result = TaskResult(task=task)
    out_dir = Path(args.out_dir) / task
    image_size = args.image_size if args.image_size else args.env_img_res * len(CAMERA_OBS_KEYS)

    for ep in range(args.num_trials_per_task):
        seed = args.seed if not args.randomize_seed else args.seed + ep
        env = _make_env(args, task, seed=seed, episode_idx=ep)
        obs = env.reset()
        task_description = env.get_ep_meta()["lang"]
        prompt = CAPTION_PREFIX + task_description

        # Let the scene settle before the policy acts.
        zero_action = np.zeros(env.action_spec[0].shape)
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(zero_action)

        success = False
        steps = 0
        action_queue: list[np.ndarray] = []
        frames: list[np.ndarray] = []

        for _ in range(max_steps):
            img = build_policy_image(obs, args.flip_images)
            if len(frames) < args.max_video_frames:
                frames.append(img)

            if not action_queue:
                state = obs_to_state7(obs) if args.use_state else None
                chunk = query_policy(
                    args.server_url,
                    img,
                    prompt,
                    image_size,
                    state,
                    want_generated_video=args.want_generated_video,
                    domain_name=args.domain_name,
                )
                chunk = map_gripper(chunk, args.gripper_map)
                action_queue = list(chunk[: args.num_open_loop_steps])

            action = action_queue.pop(0)
            if action.shape[-1] == 7 and env.action_dim == 12:
                action = np.concatenate([action, np.asarray(MOBILE_BASE_NOOP, dtype=np.float32)])
            obs, _, _, _ = env.step(action)
            steps += 1
            if env._check_success():
                success = True
                break

        result.episodes += 1
        result.successes += int(success)
        result.lengths.append(steps)
        if args.save_videos and (ep < args.save_first_n_videos or not success):
            _save_video(frames, out_dir / f"ep{ep}_{'success' if success else 'fail'}.mp4", fps=args.fps)
        print(
            f"[{task}] ep {ep + 1}/{args.num_trials_per_task} -> {'SUCCESS' if success else 'fail'} "
            f"({steps} steps; running {result.successes}/{result.episodes} = {result.success_rate:.1%}) "
            f"| '{task_description}'",
            flush=True,
        )
        env.close()

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-url", default="http://127.0.0.1:8000", help="Policy server base URL.")
    ap.add_argument("--tasks", nargs="+", default=list(ALL_TASKS), help="Tasks to evaluate (default: all 24).")
    ap.add_argument("--num-trials-per-task", type=int, default=50)
    ap.add_argument(
        "--num-open-loop-steps",
        type=int,
        default=16,
        help="Steps of each predicted chunk to execute before re-querying (Cosmos-Policy uses 16 of 32).",
    )
    ap.add_argument("--env-img-res", type=int, default=224, help="Per-camera render resolution (dataset is 224).")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Override the per-task episode cap. 0 (default) uses TASK_MAX_STEPS. Set a small "
        "value to smoke-test the server/client plumbing without waiting out full episodes.",
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Height the server scales the stack to. 0 = env_img_res * 3 (no resize). "
        "Start the server with --resolution 480 so the canvas matches training.",
    )
    ap.add_argument("--robots", default="PandaMobile", help="PandaMobile is the PandaOmron alias.")
    ap.add_argument("--obj-instance-split", default="B", help="'B' = held-out test objects.")
    ap.add_argument("--randomize-cameras", action="store_true")
    ap.add_argument("--seed", type=int, default=195, help="Cosmos-Policy paper seeds: 195, 196, 197.")
    ap.add_argument("--randomize-seed", action="store_true", help="Offset the seed per episode instead of fixing it.")
    ap.add_argument("--fps", type=int, default=20, help="Rollout video fps (RoboCasa control rate).")
    ap.add_argument(
        "--no-flip-images",
        dest="flip_images",
        action="store_false",
        help="Skip the per-camera vertical flip. The dataset IS flipped, so only use this to debug.",
    )
    ap.add_argument(
        "--no-use-state",
        dest="use_state",
        action="store_false",
        help="Do not send the proprio state (only for a policy trained with use_state=False).",
    )
    ap.add_argument("--gripper-map", default="identity", choices=("identity", "rescale", "invert"))
    # MUST match the domain the checkpoint was trained under: action2llm/llm2action are
    # per-domain weight tables, so querying a row the run never trained returns garbage.
    # Default "robocasa" -> id 5. Use "hand_pose" (id 3) only for a checkpoint trained
    # with the dataset's matching domain_name override.
    ap.add_argument("--domain-name", default="robocasa")
    ap.add_argument(
        "--want-generated-video",
        action="store_true",
        help="Query /predict instead of /predict_batch so the server also returns the model's "
        "generated video. ~5.9x slower per request and the client discards it — only for "
        "inspecting the world model. (--save-videos records the ENV frames and does not need this.)",
    )
    ap.add_argument("--controller-config-pkl", default=None, help="Pickled robosuite controller config to use.")
    ap.add_argument("--out-dir", default="outputs/robocasa_eval")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--save-first-n-videos", type=int, default=2)
    ap.add_argument("--max-video-frames", type=int, default=600)
    args = ap.parse_args()

    unknown = [t for t in args.tasks if t not in TASK_MAX_STEPS]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Available: {list(ALL_TASKS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_task(args, task) for task in args.tasks]

    total_eps = sum(r.episodes for r in results)
    total_succ = sum(r.successes for r in results)
    report = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "server_url": args.server_url,
        "seed": args.seed,
        "num_trials_per_task": args.num_trials_per_task,
        "num_open_loop_steps": args.num_open_loop_steps,
        "use_state": args.use_state,
        "flip_images": args.flip_images,
        "max_steps_override": args.max_steps or None,
        "gripper_map": args.gripper_map,
        "env_img_res": args.env_img_res,
        "overall_success_rate": (total_succ / total_eps) if total_eps else 0.0,
        "per_task": {
            r.task: {
                "success_rate": r.success_rate,
                "successes": r.successes,
                "episodes": r.episodes,
                "mean_length": float(np.mean(r.lengths)) if r.lengths else 0.0,
            }
            for r in results
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== RoboCasa eval ===")
    for r in results:
        print(f"  {r.task:<24} {r.successes:>3}/{r.episodes:<3} = {r.success_rate:.1%}")
    print(f"  {'OVERALL':<24} {total_succ:>3}/{total_eps:<3} = {report['overall_success_rate']:.1%}")
    print(f"Report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()

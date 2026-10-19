# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""robocasa GR1 env-side client for the conditioning-policy online eval.

Runs the MimicDroid robocasa **GR1** humanoid env, streams proprio + the 3 agentview
images to ``action_policy_server_gr1`` every chunk, and executes the returned 32-step
``env_action`` (already integrated to ABSOLUTE base-frame eef poses by the server).
The server does the online demo retrieval + generation + delta→abs integration; this
client owns the env, the obs→proprio extraction and the rollout/success bookkeeping.

Run this in the **mimicdroid-robocasa venv** (has robocasa/robosuite); the GPU model
server runs in the cosmos-framework venv. They talk over HTTP.

Verified format conventions (must match the training converters
``egoverse_pair_builder/73_gr1_to_egoverse57.py`` + ``10_mimicdroid_convert.py``):
  * proprio = ``robot0_base_to_{left,right}_eef_{pos,quat}`` (BASE frame, quat xyzw)
    + per-arm grip command g∈[0,1] (1=closed) tracked from the last executed action.
  * images = ``robot0_agentview_{center,sideleft,sideright}_image``, **flipped ``[::-1]``**
    (env obs is bottom-origin; training renders were flipped to upright), 128×128.
  * env_action (server output) = 14-dim ``[Rpos3, Raa3, Lpos3, Laa3, grip12(=L), grip13(=R)]``
    OSC_POSE absolute / base frame / axis-angle ori; gripper cmd ∈ [-1,1] (+1=close).

Usage (on the robocasa machine)::

    MUJOCO_GL=egl python cosmos_framework/scripts/eval_gr1_client.py \\
        --server http://127.0.0.1:8000 --task PnPSinkToCabinet --episodes 6

Manifest mode (Phase-3/4 of docs/gr1_data_eval_redesign.md) — iterate the frozen
scenes of a Phase-1 eval manifest (make_eval_manifest.py output) instead of the 3
legacy teleop inits::

    MUJOCO_GL=egl python cosmos_framework/scripts/eval_gr1_client.py \\
        --server http://127.0.0.1:8000 \\
        --manifest /mnt/ddn/dataset/gr1_eval_manifests/CloseLeftCabinetDoorL3__L3.npz \\
        --oneshot   # L2/L3: pin the manifest's one-shot conditioning demo on the server
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from robocasa.utils.mimicdroid_utils import EnvArgs, make_env, reset_to

CHUNK = 32
TASKDEMOS = "/mnt/ddn/dataset/MimicDroidDataset/TaskDemos"
CAMS = ("center", "sideleft", "sideright", "center_top")   # order = server grid TL,TR,BL,BR; old 3-view servers slice to the first 3 (grid4 servers use all 4; 2026-07-16)

# GR1 v2 training data grip convention (VERIFIED on egoverse zarr state57, all 4 tasks,
# both hands): the ACTIVE arm's gripper starts OPEN (cmd -1 -> g=0), while the INACTIVE
# arm's gripper rests at NEUTRAL cmd 0.0 -> g=0.5 (hand15 = midpoint of OPEN15/CLOSED15)
# for the WHOLE episode. Seeding both grips at -1 puts the inactive hand at the opposite
# end of its (tiny) normalized state57 range -> the proprio-conditioned policy is
# off-manifold from chunk 1 (live-batch bisect: proprio swap alone moved gen-vs-GT
# action L1 from 0.64 to 0.06).
ACTIVE_ARM = {"TurnOnFaucet": "right", "PnPSinkToRightCounterPlate": "right",
              "PnPSinkToCabinet": "right", "CloseLeftCabinetDoor": "left"}


def init_grip(task: str) -> np.ndarray:
    """Initial executed grip cmds [cmd_L, cmd_R] matching the training-data rest state."""
    return (np.array([-1.0, 0.0]) if ACTIVE_ARM.get(task, "right") == "left"
            else np.array([0.0, -1.0]))


def _post(server: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        server.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:      # surface the server-side traceback (in the 500 body)
        body = e.read().decode(errors="replace")
        print(f"[server {e.code} on {path}] {body[:2000]}", flush=True)
        raise


def _demo_src(task: str) -> str:
    """The TaskDemos hdf5 carrying the env metadata + init states for ``task``."""
    import glob
    hits = sorted(glob.glob(f"{TASKDEMOS}/{task}/*/demo_im128_notp.hdf5"))
    if not hits:
        raise FileNotFoundError(f"no TaskDemos hdf5 for task {task!r} under {TASKDEMOS}")
    return hits[0]


def make_gr1_env(task: str):
    """Build the GR1 humanoid eval env for ``task`` from its TaskDemos hdf5."""
    src = _demo_src(task)
    env, _ = make_env(src, EnvArgs(
        robots="GR1TwoHand", render=False, control_freq=20,
        controller="WHOLE_BODY_MINK_IK", use_camera_obs=True, camera_segmentations="instance",
    ))
    return env, src


def settle_env(env, steps: int = 3) -> dict:
    """Unrecorded closed-loop hold steps after reset_to — the SAME 3 hold steps
    gen_unified.settle_env runs before recording training states[0] (train t0 =
    spawn + 3 settles). Without them the first eval obs sits outside every train
    episode's start manifold (arms ~0.1 rad, fingers ~0.22 rad out-of-span) and
    the t=0 render misses dynamically-loaded objects. Hold row: current both-arm
    eef pose re-read each step (robot stays put), grips 0 — matches gen_unified
    HOLD_TMPL. Uses the SITE quat (the WHOLE_BODY_MINK_IK action frame; the plain
    body-eef quat is ~180deg flipped and makes the hold demand a bogus rotation)."""
    import robosuite.utils.transform_utils as T
    obs = env._get_observations()
    for _ in range(steps):
        row = np.zeros(14)
        for arm, b in (("right", 0), ("left", 6)):
            key = f"robot0_base_to_{arm}_eef"
            row[b:b + 3] = np.asarray(obs[key + "_pos"], float)
            row[b + 3:b + 6] = T.quat2axisangle(np.asarray(obs[key + "_quat_site"], float))
        obs, _, _, _ = env.step(row)
    return obs


def jitter_object_spawn(env, name_hint: str, jitter_cm: float,
                        off_x_cm: float = 0.0, off_y_cm: float = 0.0) -> str:
    """Nudge the task object's FREE joint xy by uniform +-jitter_cm (one-shot eval
    convention: demo-anchored scene, tiny spawn variance). Returns a log string."""
    sim = env.sim
    names = [sim.model.joint_id2name(j) for j in range(sim.model.njnt) if sim.model.jnt_type[j] == 0]
    # exact "<hint>_joint0" first — substring alone grabs e.g. vegetable_CONTAINER_joint0
    exact = [n for n in names if n and n.lower() == f"{name_hint.lower()}_joint0"]
    subs = [n for n in names if n and name_hint.lower() in n.lower() and "container" not in n.lower()]
    target = (exact or subs or [None])[0]
    for jid in range(sim.model.njnt):
        if sim.model.jnt_type[jid] != 0:            # mjJNT_FREE only
            continue
        jname = sim.model.joint_id2name(jid)
        if jname and jname == target:
            adr = sim.model.jnt_qposadr[jid]
            dx, dy = np.random.uniform(-jitter_cm, jitter_cm, 2) / 100.0 if jitter_cm > 0 else (0.0, 0.0)
            dx += off_x_cm / 100.0   # world +x = robot RIGHT (base xmat measured 2026-07-15)
            dy += off_y_cm / 100.0   # world +y = robot FORWARD
            sim.data.qpos[adr] += dx
            sim.data.qpos[adr + 1] += dy
            sim.forward()
            return f"{jname} dx={dx*100:+.1f}cm dy={dy*100:+.1f}cm"
    return f"no free joint matching '{name_hint}'"


def reset_episode(env, src: str, demo_key: str) -> dict:
    """Reset to a TaskDemos demo's initial scene (swapped onto GR1) → first obs dict."""
    g = h5py.File(src, "r")["data"][demo_key]
    reset_to(env, {
        "states": g["states"][0],
        "ep_meta": json.loads(g.attrs["ep_meta"]),
        "model": g.attrs["model_file"],
        "non_robot_qpos_idx": g.attrs["non_robot_qpos_idx"],
        "qpos_state": g.attrs["initial_qpos"],
    }, replace_robot_joints=True, change_to_gr1=True)
    return settle_env(env)


# ---------------------------------------------------------------------------
# Manifest mode (frozen Phase-1 eval scenes; docs/gr1_data_eval_redesign.md §2)
# ---------------------------------------------------------------------------

def load_manifest(npz_path: str) -> tuple[dict, dict]:
    """Phase-1 manifest -> (npz arrays, json meta). Field names per make_eval_manifest.py:
    npz: seeds/states/initial_qpos/non_robot_qpos_idx/object_state/renders/model_files/
    ep_metas/cond_demo_keys; json: env/tier/base_task/demo_hdf5/cond_demo/..."""
    arrs = np.load(npz_path, allow_pickle=True)
    with open(os.path.splitext(npz_path)[0] + ".json") as f:
        meta = json.load(f)
    if "demo_hdf5" not in meta and meta.get("derived_from"):
        # v2 refined manifests omit demo_hdf5; inherit it from the frozen v1 json
        # they were derived from (same directory, name recorded in derived_from).
        v1_path = os.path.join(os.path.dirname(npz_path), meta["derived_from"].split(" ")[0])
        with open(v1_path) as f:
            meta["demo_hdf5"] = json.load(f)["demo_hdf5"]
    return arrs, meta


def reset_to_manifest_scene(env, arrs, i: int) -> dict:
    """Restore frozen scene ``i`` onto GR1 via the SAME reset_to path as reset_episode
    (Phase-1 verified: model xml + ep_meta + non-robot qpos restores with identical
    object t0 under change_to_gr1=True). Returns the PRE-settle obs — caller must
    run settle_env() after any restore checks, before the first policy obs."""
    reset_to(env, {
        "states": np.asarray(arrs["states"][i], np.float64),
        "ep_meta": json.loads(str(arrs["ep_metas"][i])),
        "model": str(arrs["model_files"][i]),
        "non_robot_qpos_idx": np.asarray(arrs["non_robot_qpos_idx"][i], np.int64),
        "qpos_state": np.asarray(arrs["initial_qpos"][i], np.float64),
    }, replace_robot_joints=True, change_to_gr1=True)
    return env._get_observations()


def manifest_restore_err(env, arrs, i: int) -> float:
    """Max |qpos error| of the restored SCENE joints (objects/doors/fixtures) vs the
    manifest record. Mirrors reset_to's index mapping (manifest non_robot_qpos_idx ->
    sorted non-robot indices of the CURRENT GR1 model, in order), then EXCLUDES the
    GR1 ``gripper0_*`` finger joints: they count as "non-robot" under the
    _ref_joint_pos_indexes definition but are re-initialized by the robot swap
    (verified: they are the ONLY mismatching joints; all scene joints restore exactly)."""
    mj = env.sim.model
    grip_addr = set()
    for j in range(mj.njnt):
        nm = mj.joint_id2name(j)
        if nm and nm.startswith("gripper0"):
            a = mj.get_joint_qpos_addr(nm)
            grip_addr |= {int(a)} if np.isscalar(a) else set(range(a[0], a[1]))
    robot_idx = set(int(j) for j in env.robots[0]._ref_joint_pos_indexes)
    other = sorted(set(range(mj.nq)) - robot_idx)
    want = np.asarray(arrs["initial_qpos"][i], np.float64)[
        np.asarray(arrs["non_robot_qpos_idx"][i], np.int64)]
    have = np.asarray(env.sim.data.qpos, np.float64)[other]
    keep = np.array([a not in grip_addr for a in other])
    return float(np.abs(have[keep] - want[keep]).max())


def oneshot_demo_ep(meta: dict, seed: int) -> str:
    """Manifest cond_demo (TaskDemos hdf5 + demo key) -> converted egoverse zarr episode
    name (taskdemo_<TaskDemosDir>_<demoKey>, the existing taskdemo_* convention)."""
    cd = meta["cond_demo"][str(seed)]
    task_dir = cd["hdf5"].split("/TaskDemos/")[1].split("/")[0]
    return f"taskdemo_{task_dir}_{cd['key']}"


# ---------------------------------------------------------------------------
# v5 init convention: unrecorded approach bridge to the retargeted demo t0
# ---------------------------------------------------------------------------

def _rc_scripts():
    import robocasa
    d = os.path.join(os.path.dirname(robocasa.__file__), "scripts")
    if d not in os.sys.path:
        os.sys.path.insert(0, d)


class DemoT0Aligner:
    """gen_unified v5 records GR1 training episodes from the retargeted demo t0
    (the approach bridge is UNRECORDED: attrs approach_recorded=0). Eval must
    therefore start each episode at that same pose. This reuses the training
    machinery verbatim — a DemoTwoHand env for src_constants/retarget_onto_drawn
    (exactly the DTH stage), so the bridge target equals the training DTH
    episode's actions[0] for the same (scene, demo)."""

    def __init__(self, task: str):
        _rc_scripts()
        import make_eval_manifest as MEM
        self.MEM = MEM
        self.spec = MEM.SPECS[task]
        self.env = MEM.build_env(self.spec)      # DemoTwoHand, same as the gen stages
        self.sources = {s["key"]: s for s in MEM.load_sources(self.spec)}
        self._const = {}

    def demo_t0_row(self, demo_key: str, arrs, i: int) -> np.ndarray | None:
        """actions[0] of ``demo_key`` retargeted onto frozen scene ``i`` (post-settle
        drawn poses, same discipline as gen_unified run_episodes)."""
        src = self.sources.get(demo_key)
        if src is None:
            print(f"    [approach] unknown cond demo {demo_key}; skipping bridge", flush=True)
            return None
        if demo_key not in self._const:
            self._const[demo_key] = self.MEM.src_constants(self.env, self.spec, src)
        reset_to(self.env, {
            "states": np.asarray(arrs["states"][i], np.float64),
            "ep_meta": json.loads(str(arrs["ep_metas"][i])),
            "model": str(arrs["model_files"][i]),
            "non_robot_qpos_idx": np.asarray(arrs["non_robot_qpos_idx"][i], np.int64),
            "qpos_state": np.asarray(arrs["initial_qpos"][i], np.float64),
        }, replace_robot_joints=True, change_to_gr1=False)
        settle_env(self.env)
        try:
            traj, _ = self.MEM.retarget_onto_drawn(self.env, self.spec, src, self._const[demo_key])
        except Exception as e:
            print(f"    [approach] retarget failed ({type(e).__name__}: {e}); skipping bridge",
                  flush=True)
            return None
        return np.asarray(traj[0], np.float64)


def _aligner_worker(task: str, manifest_npz: str, conn) -> None:
    """DemoT0Aligner in its OWN process. The aligner's DTH env shares no EGL/mujoco
    offscreen context with the eval env — in-process it clobbered the eval env's
    renderer (all-gray obs/video the moment the DTH env reset; found 2026-07-13)."""
    al = DemoT0Aligner(task)
    arrs = np.load(manifest_npz, allow_pickle=True)
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        cond_key, i = msg
        try:
            a0 = al.demo_t0_row(cond_key, arrs, i)
            conn.send(None if a0 is None else np.asarray(a0, np.float64))
        except Exception as e:
            print(f"    [approach] worker error {type(e).__name__}: {e}", flush=True)
            conn.send(None)


class AlignerProc:
    """Process-isolated DemoT0Aligner (see _aligner_worker)."""

    def __init__(self, task: str, manifest_npz: str):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_aligner_worker, args=(task, manifest_npz, child),
                                 daemon=True)
        self._proc.start()
        assert self._conn.recv() == "ready"

    def demo_t0_row(self, cond_key: str, i: int):
        self._conn.send((cond_key, i))
        return self._conn.recv()

    def close(self):
        try:
            self._conn.send(None)
        except Exception:
            pass


def approach_to_demo_t0(env, obs: dict, a0: np.ndarray, vel: float = 0.01,
                        rot_vel: float = 0.06, max_steps: int = 60,
                        min_steps: int = 20, frac: float = 1.0) -> dict:
    """UNRECORDED velocity-capped bridge to the retargeted demo t0 — the exact rows
    gen_unified.run_gr1 executes (and does not record) before training frame 0.
    Nothing is pushed to the server during the bridge; the same --approach_vel /
    --approach_rot_vel / --approach_max defaults as gen_unified.

    GATED (user rule 2026-07-16): bridge ONLY when the demo t0 is far from home —
    fewer than ``min_steps`` bridge rows means the demo starts near the home pose
    and the bridge just moves t0 OFF the training start manifold (PlateL2 A/B:
    with bridge 0/6, without 4/6; Door L2/L3 need 29-39 rows and DO need it)."""
    _rc_scripts()
    import replay_to_gr1 as RG
    a_cur = RG._gr1_current_action(obs, a0)
    rows = RG._approach_rows(a_cur, a0, vel, rot_vel, max_steps)
    if len(rows) < min_steps:
        print(f"    [approach] SKIP bridge ({len(rows)} rows < {min_steps}: demo t0 ~ home)", flush=True)
        return obs
    # Partial bridge (--approach-frac < 1.0): execute only the FIRST frac of the
    # velocity-capped rows — the robot is carried partway toward demo t0 and the
    # POLICY must close the remaining approach gap itself. frac=1.0 (default) is
    # the certified full bridge, bit-identical to before.
    n_exec = len(rows) if frac >= 1.0 else max(0, int(round(len(rows) * max(0.0, frac))))
    for a in rows[:n_exec]:
        obs, _, _, _ = env.step(a)
    print(
        f"    [approach] bridged {n_exec}/{len(rows)} unrecorded steps to demo t0"
        + (f" (frac={frac:.2f})" if frac < 1.0 else ""), flush=True,
    )
    return obs


def get_proprio(obs: dict, last_grip: np.ndarray) -> dict:
    """Env obs → server proprio dict. ``last_grip`` = last executed [cmd_L, cmd_R] ∈ [-1,1]."""
    return {
        "pL": np.asarray(obs["robot0_base_to_left_eef_pos"]).tolist(),
        "pR": np.asarray(obs["robot0_base_to_right_eef_pos"]).tolist(),
        "quatL": np.asarray(obs["robot0_base_to_left_eef_quat"]).tolist(),   # xyzw
        "quatR": np.asarray(obs["robot0_base_to_right_eef_quat"]).tolist(),
        "gripL": float((last_grip[0] + 1.0) / 2.0),   # cmd[-1,1] → g[0,1] (1=closed)
        "gripR": float((last_grip[1] + 1.0) / 2.0),
    }


def _images(obs: dict) -> list:
    """3 agentview images, flipped to upright (training convention), as uint8 lists."""
    return [np.asarray(obs[f"robot0_agentview_{c}_image"], np.uint8)[::-1].tolist() for c in CAMS]


def run_episode(server: str, env, obs: dict, task: str, max_steps: int,
                video_path: str | None = None, reset_payload: dict | None = None,
                video_3cam: bool = False) -> bool:
    """One closed-loop episode. Caller has already reset ``env`` (legacy demo-key or
    manifest scene) and passes the first ``obs``; ``reset_payload`` extends the server
    /reset body (e.g. one-shot demo pinning)."""
    _post(server, "/reset", {"task": task, **(reset_payload or {})})
    last_grip = init_grip(task)          # per-arm rest cmds (see ACTIVE_ARM note above)
    # Seed the retrieval buffer with the t=0 frame; thereafter stream per-step proprio (below) so the
    # server's PAST window is a real 32-frame trajectory and the t>0 play match advances with the robot.
    _post(server, "/push", {"proprio": get_proprio(obs, last_grip)})
    vw = imageio.get_writer(video_path, fps=20) if video_path else None
    cur_demo = None      # latest retrieved demo frame (for the [env | demo] video)
    success = False
    try:
        for t in range(0, max_steps, CHUNK):
            resp = _post(server, "/predict",
                         {"task": task, "images": _images(obs), "proprio": get_proprio(obs, last_grip),
                          "skip_push": True})
            if resp.get("demo"):
                d = resp["demo"]
                print(f"    t={t} demo={d['ep']}@{d['w0']} dist={d['dist']:.2f}", flush=True)
            if resp.get("demo_frame") is not None:
                cur_demo = np.asarray(resp["demo_frame"], np.uint8)
            env_action = resp.get("env_action")
            if not env_action:    # warmup / no demo yet → server returns a hold; skip if absent
                continue
            step_props = []   # per-step proprio for this chunk → streamed to /push after the loop
            for a in env_action:
                a = np.asarray(a, np.float32)
                obs, _r, done, _info = env.step(a)
                last_grip = a[12:14]
                step_props.append(get_proprio(obs, last_grip))
                if vw is not None:
                    if video_3cam:
                        # cam strip [center | sideleft | sideright | center_top] — grasp
                        # geometry is invisible from center alone (user 2026-07-15);
                        # center_top = zoomed-out overhead for whole-scene context
                        # (user 2026-07-16)
                        frame = np.concatenate([
                            env.sim.render(height=256, width=256,
                                           camera_name=f"robot0_agentview_{c}")[::-1]
                            for c in ("center", "sideleft", "sideright", "center_top")], axis=1)
                    else:
                        frame = env.sim.render(height=256, width=256,
                                               camera_name="robot0_agentview_center")[::-1]
                    if cur_demo is not None:   # append the retrieved demo frame
                        demo256 = np.asarray(Image.fromarray(cur_demo).resize((256, 256)))
                        frame = np.concatenate([frame, demo256], axis=1)
                    vw.append_data(np.ascontiguousarray(frame))
                if env._check_success():
                    success = True
                    break
            if step_props:    # stream this chunk's per-step proprio → real 32-frame past window
                _post(server, "/push", {"proprios": step_props})
            if success or done:
                break
    finally:
        if vw is not None:
            vw.close()
    return success


def run_legacy(args) -> None:
    """Legacy mode: cycle the task's ~3 TaskDemos teleop inits (unchanged behavior)."""
    if args.episodes is None:
        args.episodes = 6
    env, src = make_gr1_env(args.task)
    demo_keys = list(h5py.File(src, "r")["data"].keys())   # ~3 demos → cycle for init variety
    if getattr(args, "demo_only", None):
        demo_keys = [k for k in demo_keys if k.endswith(args.demo_only)]
        assert demo_keys, f"--demo-only {args.demo_only!r}: no matching demo key"
        print(f"[{args.task}] demo-only: {demo_keys[0]}", flush=True)
    succ = 0
    for ep in range(args.episodes):
        dk = demo_keys[ep % len(demo_keys)]
        t0 = time.time()
        vp = os.path.join(args.video_dir, f"{args.task}_ep{ep:02d}_{dk}.mp4") if args.video_dir else None
        obs = reset_episode(env, src, dk)
        if getattr(args, "approach_demo_t0", False):
            # drive the eef to THIS demo's first COMMANDED pose (actions[0]) before the
            # policy starts. Command (not recorded obs[0]) is deliberate: the whole-body
            # IK stops short of any target by the same posture-cost bias for the demo
            # robot and for us, so aiming at the same command lands both at the same
            # physical pose (probe v4/v5 2026-07-16: cmd target -> 1-2cm vs demo;
            # recorded-pose target -> systematically shorter, RightL2 d1 4.7->5.5cm).
            a0 = np.asarray(h5py.File(src, "r")["data"][dk]["actions"][0], np.float64)
            obs = approach_to_demo_t0(env, obs, a0, min_steps=args.approach_min_steps, frac=getattr(args, "approach_frac", 1.0))
        if getattr(args, "spawn_jitter_cm", 0.0) > 0 or getattr(args, "spawn_offset_x_cm", 0.0) or getattr(args, "spawn_offset_y_cm", 0.0):
            msg = jitter_object_spawn(env, args.jitter_object, args.spawn_jitter_cm,
                                      args.spawn_offset_x_cm, args.spawn_offset_y_cm)
            obs = settle_env(env)
            print(f"[{args.task}] ep {ep} spawn-jitter: {msg}", flush=True)
        payload = {}
        if getattr(args, "mono_floor", None):
            payload["mono_floor"] = args.mono_floor
        if args.oneshot_pool3:
            # v6 default one-shot (설정 A): pin ALL 3 taskdemos of the task — the server
            # matches within their combined windows (train-space 26-d). This branch was
            # missing from the default run() (manifest mode only) — found 2026-07-14
            # when hits showed dist>0/dthdemo instead of the pinned taskdemo pool.
            payload["oneshot_task"] = args.task
            print(f"[{args.task}] ep {ep} oneshot_task={args.task} (3 demos)", flush=True)
        elif args.oneshot:
            # legacy+oneshot: pin the SAME taskdemo the scene came from — "start near
            # demo_k, conditioned one-shot on demo_k" (within-episode matching on server).
            # hdf5 demo keys are already "<Task>_D_demo_k", so prefix only the task dir.
            payload["oneshot_demo"] = {"ep": f"taskdemo_{args.task}_{dk}", "w0": 0}
        ok = run_episode(args.server, env, obs, args.task, args.max_steps, video_path=vp,
                         reset_payload=payload, video_3cam=args.video_3cam)
        succ += int(ok)
        print(f"[{args.task}] ep {ep + 1}/{args.episodes} init={dk} -> {'SUCCESS' if ok else 'fail'} "
              f"({time.time() - t0:.0f}s)  running {succ}/{ep + 1}", flush=True)
    print(f"\nGR1 {args.task}: {succ}/{args.episodes} success ({100 * succ / args.episodes:.0f}%)", flush=True)


def run_manifest(args) -> None:
    """Manifest mode: iterate the frozen eval scenes of a Phase-1 manifest npz."""
    arrs, meta = load_manifest(args.manifest)
    tag, task = meta["tag"], meta["base_task"]     # base task drives grip init + server pools
    seeds = [int(s) for s in arrs["seeds"]]
    n = len(seeds) if args.episodes is None else min(args.episodes, len(seeds))
    # the manifest's own TaskDemos hdf5 carries the env metadata for make_env
    env, _ = make_env(meta["demo_hdf5"], EnvArgs(
        robots="GR1TwoHand", render=False, control_freq=20,
        controller="WHOLE_BODY_MINK_IK", use_camera_obs=True, camera_segmentations="instance",
    ))
    # v6: NO bridge/aligner — training episodes RECORD the approach from home+settle,
    # so eval t0 == training t0 by construction (AlignerProc retired with v5).
    aligner = None
    succ = 0
    for i in range(n):
        seed = seeds[i]
        t0 = time.time()
        reset_to_manifest_scene(env, arrs, i)
        rerr = manifest_restore_err(env, arrs, i)   # object-t0 restore check (pre-settle: bit-exact)
        obs = settle_env(env)                       # then the training-convention 3 hold steps
        if aligner is not None:
            # v5: bridge (unrecorded) to the retargeted t0 of the SAME demo used as
            # the conditioning demo; default = the manifest's per-scene cond demo.
            if args.oneshot_demo:
                cond_key = f"{task}_D_{args.oneshot_demo}"
            elif args.oneshot:
                cond_key = meta["cond_demo"][str(seed)]["key"]
            else:
                cond_key = str(arrs["cond_demo_keys"][i])
            a0 = aligner.demo_t0_row(cond_key, i)
            if a0 is not None:
                obs = approach_to_demo_t0(env, obs, a0, min_steps=args.approach_min_steps, frac=getattr(args, "approach_frac", 1.0))
        payload = {}
        if args.oneshot_demo:
            # force ONE specific taskdemo for every scene (seen-tier one-shot sweep:
            # measure demo_1/2/3 separately vs the retrieval protocol)
            ep_name = f"taskdemo_{task}_{task}_D_{args.oneshot_demo}"
            payload["oneshot_demo"] = {"ep": ep_name, "w0": 0}
            print(f"[{tag}] scene {i + 1}/{n} seed={seed} oneshot={ep_name}", flush=True)
        elif args.oneshot_pool3:
            # v6 default one-shot: pin ALL 3 taskdemos of the task — the server
            # matches within their combined windows (train-space 26-d).
            payload["oneshot_task"] = task
        if getattr(args, "mono_floor", None):
            payload["mono_floor"] = args.mono_floor
            print(f"[{tag}] scene {i + 1}/{n} seed={seed} oneshot_task={task} (3 demos)", flush=True)
        elif args.oneshot:
            ep_name = oneshot_demo_ep(meta, seed)
            payload["oneshot_demo"] = {"ep": ep_name, "w0": 0}
            print(f"[{tag}] scene {i + 1}/{n} seed={seed} oneshot={ep_name}", flush=True)
        vp = os.path.join(args.video_dir, f"{tag}_seed{seed}.mp4") if args.video_dir else None
        ok = run_episode(args.server, env, obs, task, args.max_steps,
                         video_path=vp, reset_payload=payload)
        succ += int(ok)
        print(f"[{tag}] scene {i + 1}/{n} seed={seed} restore_err={rerr:.2e} -> "
              f"{'SUCCESS' if ok else 'fail'} ({time.time() - t0:.0f}s)  running {succ}/{i + 1}",
              flush=True)
    print(f"\nGR1 {tag}: {succ}/{n} success ({100 * succ / max(n, 1):.0f}%)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--approach-demo-t0", action="store_true",
                    help="bridge the eef to the init demo's actions[0] pose before the rollout")
    ap.add_argument("--approach-frac", type=float, default=1.0,
                    help="fraction of the approach-bridge rows to execute (1.0=full certified bridge; "
                         "e.g. 0.5 carries the robot HALFWAY to demo t0 — the policy must close the rest)")
    ap.add_argument("--approach-min-steps", type=int, default=20,
                    help="skip the bridge when it needs fewer rows than this (demo t0 ~ home); 0 = always bridge")
    ap.add_argument("--demo-only", default=None,
                    help="run ONLY the demo key ending with this (e.g. 'demo_2') — lets N workers "
                         "split one task's episodes per-demo for parallel eval")
    ap.add_argument("--mono-floor", type=int, default=None,
                    help="per-episode server mono-floor override (sent in /reset); "
                         "task rule 2026-07-16: TurnOnFaucetL3=16, others server default (8)")
    ap.add_argument("--video-3cam", action="store_true",
                    help="record [center|sideleft|sideright] strip instead of center-only")
    ap.add_argument("--spawn-jitter-cm", type=float, default=0.0,
                    help="after reset, nudge the task object xy by uniform +-cm (one-shot spawn variance)")
    ap.add_argument("--spawn-offset-x-cm", type=float, default=0.0,
                    help="deterministic spawn shift, world +x = robot RIGHT")
    ap.add_argument("--spawn-offset-y-cm", type=float, default=0.0,
                    help="deterministic spawn shift, world +y = robot FORWARD")
    ap.add_argument("--jitter-object", default="tomato",
                    help="substring of the object FREE joint to jitter (Cabinet: tomato / Plate: corn)")
    ap.add_argument("--raw-spawn", action="store_true",
                    help="skip the PnPSinkToCabinet* default anchor/jitter -> raw demo spawn (offset OFF)")
    ap.add_argument("--oneshot-pool3", action="store_true",
                    help="v6 one-shot: pin the task's 3 taskdemos (server matches across all)")
    ap.add_argument("--task", default="PnPSinkToCabinet")
    ap.add_argument("--episodes", type=int, default=None,
                    help="legacy: episode count (default 6); manifest: scene cap (default all)")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--video-dir", default=None, help="save per-episode center-cam rollout mp4 here")
    ap.add_argument("--manifest", default=None,
                    help="Phase-1 eval manifest npz (gr1_eval_manifests/<tag>.npz); "
                         "iterates its frozen scenes instead of the legacy demo inits")
    ap.add_argument("--oneshot", action="store_true",
                    help="manifest mode: pin the manifest's per-scene one-shot conditioning "
                         "demo on the server (L2/L3 protocol; requires the demos converted "
                         "to egoverse zarr episodes)")
    ap.add_argument("--no-approach", action="store_true",
                    help="manifest mode: skip the v5 unrecorded approach bridge to the "
                         "retargeted demo t0 (use when evaluating pre-v5 checkpoints, "
                         "whose training episodes recorded the bridge)")
    ap.add_argument("--oneshot-demo", default=None,
                    help="manifest mode: pin ONE specific taskdemo key (e.g. demo_1) for "
                         "every scene — seen-tier one-shot sweep vs the retrieval protocol")
    args = ap.parse_args()
    # PnPSinkToCabinet* 디폴트 스폰 (user 2026-07-15): 학습 밀도 중심(F+3/R+3)을 앵커로
    # 두고 그 주변 ±1cm jitter — 명시 플래그가 오면 그대로 존중.
    # Plate는 jitter 안 씀 (user 2026-07-16): corn이 싱크 안 container 위에 얹혀 있어
    # xy 오프셋이 corn을 container 가장자리로 밀어 굴러떨어짐(스폰 튕김) — raw demo 스폰 사용.
    if args.task.startswith("PnPSinkToCabinet") and not args.raw_spawn \
            and args.spawn_offset_x_cm == 0.0 \
            and args.spawn_offset_y_cm == 0.0 and args.spawn_jitter_cm == 0.0:
        args.spawn_offset_x_cm = 2.0   # RIGHT (user 2026-07-15: 3cm은 과함)
        args.spawn_offset_y_cm = 3.0   # FORWARD
        args.spawn_jitter_cm = 1.0
        args.jitter_object = "vegetable"
        print(f"[{args.task}] default spawn: anchor F+3/R+2 cm + jitter ±1.0 cm", flush=True)


    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
    if args.manifest:
        run_manifest(args)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()

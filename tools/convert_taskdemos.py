# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert MimicDroidDataset/TaskDemos → egoverse ``action57`` zarr (emb=mimicdroid).

TaskDemos are per-task FloatingTwoHand/robocasa demonstrations (``TaskDemos/<task>/
<num>/demo_im128_notp.hdf5``, ~3 demos/task, single task each, NO subgoal.json).
They are the **online-eval demo pool** for the GR1 conditioning policy: at inference
the GR1 query is conditioned on a task-matched TaskDemo (instead of the long
SinkEnvForPlay play sequence). They are the same robot family as the play demos, so
they get ``embodiment="mimicdroid"`` (matches the training demo embodiment), and
each demo is one whole task (one subgoal spanning the clip).

Reuses the action57 construction from ``10_mimicdroid_convert.py`` (head9 | Leef9 |
Lhand15 | Reef9 | Rhand15). Output: ``egoverse/taskdemo_<task>_<demo>`` so the
server's ``EgoVerseWindowReader.read_clip`` can read the demo clip.

Usage::

    python tools/convert_taskdemos.py \\
        --src /mnt/ddn/dataset/MimicDroidDataset/TaskDemos \\
        --dst /mnt/ddn/dataset/egoverse
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os

import h5py
import numpy as np
import zarr
from PIL import Image
from scipy.spatial.transform import Rotation as R

FPS = 20
IMG_KEY = "robot0_agentview_center_image"
GRIP_CMD = (12, 13)
JPEG_Q = 92

HEAD9 = np.array([-0.60, 0.0, 1.17, 0.033, -0.999, -0.003, 0.818, 0.028, -0.574], float)
IDENT9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], float)
OPEN15 = np.array([0.0005, 0.0958, -0.034, 0.0285, 0.0222, -0.1263, 0.0085, 0.0217, -0.1285,
                   -0.0116, 0.0222, -0.1282, -0.0314, 0.0229, -0.1259], float)
CLOSED15 = np.array([0.0052, 0.0775, -0.0701, 0.0266, 0.044, -0.0916, 0.0085, 0.0441, -0.1016,
                     -0.0097, 0.044, -0.0937, -0.0278, 0.044, -0.0919], float)

# GR1 query tasks → a detailed instruction (for the demo's single subgoal lang).
TASK_LANG = {
    "CloseLeftCabinetDoor": "close the left cabinet door",
    "CloseRightCabinetDoor": "close the right cabinet door",
    "PnPSinkToCabinet": "pick the object from the sink and place it in the cabinet",
    "PnPSinkToRightCounterPlate": "pick the object from the sink and place it on the plate",
    "PnPSinkToMicrowaveTop": "pick the object from the sink and place it on the microwave",
    "TurnOnFaucet": "turn on the sink faucet",
}


def _rmat(quat_xyzw):
    q = np.asarray(quat_xyzw, float)
    n = np.linalg.norm(q, axis=1)
    bad = (n < 1e-6) | ~np.isfinite(q).all(1)
    qn = q.copy()
    qn[bad] = [0, 0, 0, 1]
    qn /= np.linalg.norm(qn, axis=1, keepdims=True)
    return R.from_quat(qn).as_matrix(), ~bad


def _eef9(pos, quat):
    Rm, vq = _rmat(quat)
    vf = vq & ~np.all(pos == 0, 1)
    RtT = np.transpose(Rm[:-1], (0, 2, 1))
    dpos = np.einsum("nij,nj->ni", RtT, pos[1:] - pos[:-1])
    dR = np.einsum("nij,njk->nik", RtT, Rm[1:])
    return np.concatenate([dpos, dR[:, :, 0], dR[:, :, 1]], axis=1), vf[:-1] & vf[1:]


def _grip(actions, dim):
    return (np.clip(actions[:, dim], -1, 1) + 1) / 2


def _hand15(g):
    g = np.asarray(g, float)[:, None]
    return (1.0 - g) * OPEN15 + g * CLOSED15


def _abs9(pos, quat):
    Rm, _ = _rmat(quat)
    return np.concatenate([pos, Rm[:, :, 0], Rm[:, :, 1]], axis=1)


def _base_task(folder: str) -> str:
    """Strip the L2/L3 difficulty suffix → the GR1 query task name."""
    for suf in ("L3", "L2"):
        if folder.endswith(suf):
            return folder[: -len(suf)]
    return folder


def convert(h5_im: str, key_im: str, task_folder: str, out_dir: str) -> int:
    dg = h5py.File(h5_im, "r")["data"][key_im]
    obs = dg["obs"]
    n = obs["robot0_base_to_left_eef_pos"].shape[0]
    Lp = np.asarray(obs["robot0_base_to_left_eef_pos"]); Lq = np.asarray(obs["robot0_base_to_left_eef_quat"])
    Rp = np.asarray(obs["robot0_base_to_right_eef_pos"]); Rq = np.asarray(obs["robot0_base_to_right_eef_quat"])
    actions = np.asarray(dg["actions"])
    Lg, Rg = _grip(actions, GRIP_CMD[0]), _grip(actions, GRIP_CMD[1])
    le9, lv = _eef9(Lp, Lq); re9, rv = _eef9(Rp, Rq)
    act = np.concatenate([np.broadcast_to(IDENT9, (n - 1, 9)), le9, _hand15(Lg[1:]),
                          re9, _hand15(Rg[1:])], axis=1).astype(np.float64)
    valid = np.stack([np.ones(n - 1, bool), lv, rv], axis=1)
    st = np.concatenate([np.broadcast_to(HEAD9, (n, 9)), _abs9(Lp, Lq), _hand15(Lg),
                         _abs9(Rp, Rq), _hand15(Rg)], axis=1).astype(np.float64)
    spd = np.stack([np.r_[0, np.linalg.norm(np.diff(Lp, axis=0), axis=1)] * FPS,
                    np.r_[0, np.linalg.norm(np.diff(Rp, axis=0), axis=1)] * FPS], axis=1)

    base = _base_task(task_folder)
    lang = TASK_LANG.get(base, base)
    subgoals = [{"start": 0, "end": n, "lang": lang, "type": base}]

    os.makedirs(out_dir, exist_ok=True)
    g = zarr.open_group(out_dir, mode="w")
    imgs = np.asarray(obs[IMG_KEY])
    enc = np.empty(n, dtype=object)
    for i in range(n):
        bf = io.BytesIO()
        Image.fromarray(imgs[i]).save(bf, "JPEG", quality=JPEG_Q)
        enc[i] = bf.getvalue()
    g.create_array(name="images.front_1", shape=(n,), chunks=(1,),
                   dtype="variable_length_bytes", overwrite=True)[:] = enc
    a = g.create_group("action57")
    a.create_array(name="action", shape=act.shape, chunks=(min(200, len(act)), 57), dtype="float64", overwrite=True)[:] = act
    a.create_array(name="valid", shape=valid.shape, chunks=(min(200, len(valid)), 3), dtype="bool", overwrite=True)[:] = valid
    a.attrs.update(dict(action57_hz=FPS, layout="head9|Leef9|Lhand15|Reef9|Rhand15", domain="hand_pose"))
    s = g.create_group("state57")
    s.create_array(name="state", shape=st.shape, chunks=(min(200, n), 57), dtype="float64", overwrite=True)[:] = st
    s.create_array(name="valid", shape=(n, 3), chunks=(min(200, n), 3), dtype="bool", overwrite=True)[:] = np.ones((n, 3), bool)
    m = g.create_group("meta")
    m.create_array(name="eef_speed", shape=spd.shape, chunks=(min(200, n), 2), dtype="float64", overwrite=True)[:] = spd
    g.attrs.update(dict(embodiment="mimicdroid", robot="FloatingTwoHand", env="robocasa",
                        task_name=base, task_description=lang, fps=FPS, total_frames=n,
                        source=f"TaskDemos/{task_folder}/{key_im}", action57_done=True, state57_abs=True, dim=57,
                        speed_meta_done=True, taskdemo=True, task_folder=task_folder,
                        subgoals=json.dumps(subgoals), n_subgoals=1))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/mnt/ddn/dataset/MimicDroidDataset/TaskDemos")
    ap.add_argument("--dst", default="/mnt/ddn/dataset/egoverse")
    args = ap.parse_args()
    total = 0
    for task_folder in sorted(os.listdir(args.src)):
        for h5_im in sorted(glob.glob(f"{args.src}/{task_folder}/*/demo_im128_notp.hdf5")):
            for key_im in h5py.File(h5_im, "r")["data"].keys():
                out = f"{args.dst}/taskdemo_{task_folder}_{key_im}"
                n = convert(h5_im, key_im, task_folder, out)
                total += 1
                print(f"  {task_folder}/{key_im} -> {out}  ({n} frames)", flush=True)
    print(f"Converted {total} TaskDemos → {args.dst}/taskdemo_*")


if __name__ == "__main__":
    main()

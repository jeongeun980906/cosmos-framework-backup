"""Convert the 12 canonical GR1 TaskDemos to 2x-SPEED egoverse zarr (emb=mimicdroid, __2x).

2x = subsample the raw obs by 2 (every 2nd frame, half length, plays 2x faster), then run the
SAME action57/state57 derivation as tools/convert_taskdemos.py (eef9 body-frame deltas
RE-DERIVED on the subsampled poses; abs9; grip→hand15 template). Output egoverse zarr name
``taskdemo_<task>_<task>_D_demo_{i}__2x`` so build_gr1_posttrain_pairs / the reader can read it
for the 2x post-train t=0 conditioning. Mirrors the SinkEnvForPlay play __2x (half-length) zarrs.
"""
from __future__ import annotations
import io, json, os, importlib.util
import numpy as np, h5py, zarr
from PIL import Image

# reuse the exact 1x converter helpers + constants
_p = "/mnt/ddn/cosmos-framework/tools/convert_taskdemos.py"
_s = importlib.util.spec_from_file_location("ct", _p); ct = importlib.util.module_from_spec(_s); _s.loader.exec_module(ct)

SRC = "/mnt/ddn/dataset/MimicDroidDataset/TaskDemos"
DST = "/mnt/ddn/dataset/egoverse"
TASKS = ["TurnOnFaucet", "CloseLeftCabinetDoor", "PnPSinkToCabinet", "PnPSinkToRightCounterPlate"]
STEP = 2  # 2x


def convert_2x(task: str, key: str) -> int:
    h5 = f"{SRC}/{task}/003/demo_im128_notp.hdf5"
    dg = h5py.File(h5, "r")["data"][key]
    obs = dg["obs"]
    sl = slice(None, None, STEP)
    Lp = np.asarray(obs["robot0_base_to_left_eef_pos"])[sl]; Lq = np.asarray(obs["robot0_base_to_left_eef_quat"])[sl]
    Rp = np.asarray(obs["robot0_base_to_right_eef_pos"])[sl]; Rq = np.asarray(obs["robot0_base_to_right_eef_quat"])[sl]
    actions = np.asarray(dg["actions"])[sl]
    imgs = np.asarray(obs[ct.IMG_KEY])[sl]
    n = Lp.shape[0]
    Lg, Rg = ct._grip(actions, ct.GRIP_CMD[0]), ct._grip(actions, ct.GRIP_CMD[1])
    le9, lv = ct._eef9(Lp, Lq); re9, rv = ct._eef9(Rp, Rq)
    act = np.concatenate([np.broadcast_to(ct.IDENT9, (n - 1, 9)), le9, ct._hand15(Lg[1:]),
                          re9, ct._hand15(Rg[1:])], axis=1).astype(np.float64)
    valid = np.stack([np.ones(n - 1, bool), lv, rv], axis=1)
    st = np.concatenate([np.broadcast_to(ct.HEAD9, (n, 9)), ct._abs9(Lp, Lq), ct._hand15(Lg),
                         ct._abs9(Rp, Rq), ct._hand15(Rg)], axis=1).astype(np.float64)
    spd = np.stack([np.r_[0, np.linalg.norm(np.diff(Lp, axis=0), axis=1)] * ct.FPS,
                    np.r_[0, np.linalg.norm(np.diff(Rp, axis=0), axis=1)] * ct.FPS], axis=1)
    base = ct._base_task(task); lang = ct.TASK_LANG.get(base, base)
    subgoals = [{"start": 0, "end": n, "lang": lang, "type": base}]
    out_dir = f"{DST}/taskdemo_{task}_{key}__2x"
    g = zarr.open_group(out_dir, mode="w")
    enc = np.empty(n, dtype=object)
    for i in range(n):
        bf = io.BytesIO(); Image.fromarray(imgs[i]).save(bf, "JPEG", quality=ct.JPEG_Q); enc[i] = bf.getvalue()
    g.create_array(name="images.front_1", shape=(n,), chunks=(1,), dtype="variable_length_bytes", overwrite=True)[:] = enc
    a = g.create_group("action57")
    a.create_array(name="action", shape=act.shape, chunks=(min(200, len(act)), 57), dtype="float64", overwrite=True)[:] = act
    a.create_array(name="valid", shape=valid.shape, chunks=(min(200, len(valid)), 3), dtype="bool", overwrite=True)[:] = valid
    a.attrs.update(dict(action57_hz=ct.FPS, layout="head9|Leef9|Lhand15|Reef9|Rhand15", domain="hand_pose"))
    s = g.create_group("state57")
    s.create_array(name="state", shape=st.shape, chunks=(min(200, n), 57), dtype="float64", overwrite=True)[:] = st
    s.create_array(name="valid", shape=(n, 3), chunks=(min(200, n), 3), dtype="bool", overwrite=True)[:] = np.ones((n, 3), bool)
    m = g.create_group("meta")
    m.create_array(name="eef_speed", shape=spd.shape, chunks=(min(200, n), 2), dtype="float64", overwrite=True)[:] = spd
    g.attrs.update(dict(embodiment="mimicdroid", robot="FloatingTwoHand", env="robocasa",
                        task_name=base, task_description=lang, fps=ct.FPS, total_frames=n,
                        source=f"TaskDemos/{task}/003/{key} [2x subsample::2]", action57_done=True, state57_abs=True, dim=57,
                        speed_meta_done=True, taskdemo=True, task_folder=task, twox=True,
                        subgoals=json.dumps(subgoals), n_subgoals=1))
    return n


if __name__ == "__main__":
    tot = 0
    for task in TASKS:
        for i in (1, 2, 3):
            key = f"{task}_D_demo_{i}"
            n = convert_2x(task, key)
            print(f"  taskdemo_{task}_{key}__2x : {n} frames (2x)"); tot += 1
    print(f"DONE {tot} 2x taskdemos -> {DST}/taskdemo_*__2x")

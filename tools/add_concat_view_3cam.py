# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Add ``images.sideleft`` + ``images.sideright`` to the robocasa egoverse zarrs.

LIBERO/DROID condition on a vertical concat of multiple cameras (Cosmos3 ``concat_view``;
the reader already vertically-concats ``spec.cams`` — that's how ``eva`` uses 3 cams).
The mimicdroid/gr1/taskdemo zarrs store only the center agentview (``images.front_1``).
This tool ADDS the two side agentviews as SEPARATE keys (each 128×128 JPEG, same format
as front_1), read from the source ``demo_im128_notp.hdf5``. **``images.front_1`` (center)
is left UNTOUCHED** — so the egoverse pre-train (mid-training), which reads only
``front_1``, stays center-only; only the GR1 post-train (which sets cams to all 3) gets
the concat. Everything else (action57/state57/attrs/frame count) untouched → pairs stay
valid. Reversible (raw hdf5 has the cams), idempotent (``sidecams_done`` attr). Default
DRY-RUN; ``--apply`` writes.

Episode → raw hdf5 mapping (verified by frame-count match):
  * play  (emb mimicdroid, src ``MimicDroid/<NNN>/<demo>``): training2 if ep endswith
    ``__2x`` else training; ``<NNN>/demo_im128_notp.hdf5``, demo key ``<demo>``.
  * gr1   (emb GR1, src ``GR1/<task>/<demo>``): ``GR1/<task>/<NNN>/...``, demo ``<demo>``.
  * taskdemo (ep ``taskdemo_<folder>_<demo>``, src ``TaskDemos/<folder>/<demo>``).
"""

from __future__ import annotations

import argparse
import glob
import io
import os

import h5py
import numpy as np
import zarr
from PIL import Image

EG = "/mnt/ddn/dataset/egoverse"
MDD = "/mnt/ddn/dataset/MimicDroidDataset"
CENTER = "robot0_agentview_center_image"
# side cams added as separate keys; the reader vertical-concats [front_1, sideleft, sideright]
SIDE = {"images.sideleft": "robot0_agentview_sideleft_image", "images.sideright": "robot0_agentview_sideright_image"}
JPEG_Q = 92


def _find_raw(folder_glob: str, demo_key: str, want_T: int) -> tuple[str, str] | None:
    for h5 in sorted(glob.glob(folder_glob)):
        try:
            f = h5py.File(h5, "r")
            if demo_key in f["data"] and f["data"][demo_key]["obs"][CENTER].shape[0] == want_T:
                f.close()
                return h5, demo_key
            f.close()
        except Exception:
            continue
    return None


def _resolve(ep: str, emb: str, source: str, want_T: int) -> tuple[str, str] | None:
    if ep.startswith("taskdemo_"):                      # src TaskDemos/<folder>/<demo>
        _, folder, demo = source.split("/", 2)
        return _find_raw(f"{MDD}/TaskDemos/{folder}/*/demo_im128_notp.hdf5", demo, want_T)
    if source.startswith("MimicDroid/"):                # play: src MimicDroid/<NNN>/<demo>
        _, nnn, demo = source.split("/", 2)
        d = "training2" if ep.endswith("__2x") else "training"
        h5 = f"{MDD}/{d}/{nnn}/demo_im128_notp.hdf5"
        return (h5, demo) if os.path.exists(h5) else None
    if source.startswith("GR1/"):                       # gr1: src GR1/<task>/<demo>
        _, task, demo = source.split("/", 2)
        return _find_raw(f"{MDD}/GR1/{task}/*/demo_im128_notp.hdf5", demo, want_T)
    return None


def _encode(imgs: np.ndarray) -> np.ndarray:
    enc = np.empty(imgs.shape[0], dtype=object)
    for t in range(imgs.shape[0]):
        bf = io.BytesIO()
        Image.fromarray(imgs[t]).save(bf, "JPEG", quality=JPEG_Q)
        enc[t] = bf.getvalue()
    return enc


def process(ep_dir: str, apply: bool) -> str:
    g = zarr.open_group(ep_dir, mode="r+" if apply else "r")
    a = dict(g.attrs)
    emb = a.get("embodiment", "")
    ep = os.path.basename(ep_dir)
    if not (emb in ("mimicdroid", "gr1", "GR1") or ep.startswith("taskdemo_")):
        return "skip (not robocasa)"
    if a.get("sidecams_done"):
        return "skip (already has side cams)"
    T = g["images.front_1"].shape[0]
    raw = _resolve(ep, emb, str(a.get("source", "")), T)
    if raw is None:
        return f"!! NO RAW MATCH (src={a.get('source')}, T={T})"
    h5, demo = raw
    obs = h5py.File(h5, "r")["data"][demo]["obs"]
    if apply:
        for key, cam in SIDE.items():
            enc = _encode(np.asarray(obs[cam]))
            if key in g:
                del g[key]
            g.create_array(name=key, shape=(T,), chunks=(1,), dtype="variable_length_bytes", overwrite=True)[:] = enc
        g.attrs["sidecams_done"] = True
    return ("APPLIED " if apply else "dry-run ") + f"T={T} +sideleft+sideright from {os.path.relpath(h5, MDD)}::{demo}"


def _unwrap(e: object) -> bytes:
    """VLen-bytes zarr elements come back as nested 0-d object ndarrays; unwrap to raw bytes."""
    while isinstance(e, np.ndarray):
        e = e[()] if e.shape == () else e.tolist()
    return bytes(e)


def process_twox(ep_dir: str, apply: bool) -> str:
    """Add side cams to a ``*__2x`` episode by stride-2 subsampling its 1x sibling.

    A ``__2x`` episode is the 1x sibling temporally subsampled ``[::2]`` (verified: its
    ``front_1`` is byte-exactly ``front_1_1x[::2]`` with ``T_1x == 2*T_2x``). The 1x sibling
    already has correct ``images.sideleft``/``images.sideright``; we copy those SAME encoded
    JPEG bytes at even indices so the output is byte-identical to the tool's own 1x output.
    Positional stride-2 (not front-byte lookup) is used so duplicate/static frames — whose
    side views legitimately differ — stay correctly aligned. ``front_1`` left UNTOUCHED.
    """
    ep = os.path.basename(ep_dir)
    if not ep.endswith("__2x"):
        return "skip (not __2x)"
    sib = ep_dir[: -len("__2x")]
    if not (os.path.isdir(sib) and os.path.exists(f"{sib}/images.sideleft/zarr.json")):
        return f"!! NO 1x SIBLING with side cams ({os.path.basename(sib)})"
    g2 = zarr.open_group(ep_dir, mode="r+" if apply else "r")
    if dict(g2.attrs).get("sidecams_done"):
        return "skip (already has side cams)"
    g1 = zarr.open_group(sib, mode="r")
    f2, f1 = g2["images.front_1"], g1["images.front_1"]
    t2, t1 = int(f2.shape[0]), int(f1.shape[0])
    if t1 != 2 * t2:
        return f"!! LEN MISMATCH T_1x={t1} != 2*T_2x={2 * t2}"
    b1 = [_unwrap(f1[i]) for i in range(t1)]
    b2 = [_unwrap(f2[j]) for j in range(t2)]
    if not all(b2[j] == b1[2 * j] for j in range(t2)):  # prove keep==[::2] before trusting it
        return "!! front_1 NOT byte-exact 1x[::2] — refusing (would misalign side cams)"
    keep = list(range(0, t1, 2))
    if apply:
        for key in SIDE:
            src = g1[key]
            enc = np.empty(t2, dtype=object)
            for j, i in enumerate(keep):
                enc[j] = _unwrap(src[i])  # copy 1x encoded bytes verbatim (byte-identical)
            if key in g2:
                del g2[key]
            g2.create_array(name=key, shape=(t2,), chunks=(1,), dtype="variable_length_bytes", overwrite=True)[:] = enc
        g2.attrs["sidecams_done"] = True
    return ("APPLIED " if apply else "dry-run ") + f"T_2x={t2} +sideleft+sideright from {os.path.basename(sib)}[::2]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--twox", action="store_true", help="Only *__2x episodes; subsample 1x sibling's side cams [::2].")
    args = ap.parse_args()
    n = miss = 0
    if args.twox:
        for d in sorted(x for x in os.listdir(EG) if x.endswith("__2x")):
            p = f"{EG}/{d}"
            if not os.path.isdir(p):
                continue
            try:
                msg = process_twox(p, args.apply)
            except Exception as e:  # noqa: BLE001
                msg = f"!! ERROR {type(e).__name__}: {str(e)[:80]}"
            print(f"{d}: {msg}", flush=True)
            miss += msg.startswith("!!")
            if not msg.startswith("skip"):
                n += 1
            if args.limit and n >= args.limit:
                break
        print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'} (twox) on {n} episode(s), {miss} unmatched")
        return
    for d in sorted(os.listdir(EG)):
        p = f"{EG}/{d}"
        if not os.path.isdir(p):
            continue
        try:
            a = dict(zarr.open_group(p, mode="r").attrs)
        except Exception:
            continue
        emb = a.get("embodiment", "")
        if not (emb in ("mimicdroid", "gr1", "GR1") or d.startswith("taskdemo_")):
            continue
        try:
            msg = process(p, args.apply)
        except Exception as e:  # noqa: BLE001 — log + continue so one bad ep can't abort the run
            msg = f"!! ERROR {type(e).__name__}: {str(e)[:80]}"
        print(f"{d}: {msg}", flush=True)
        miss += msg.startswith("!!")
        if not msg.startswith("skip"):
            n += 1
        if args.limit and n >= args.limit:
            break
    print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'} on {n} episode(s), {miss} unmatched")


if __name__ == "__main__":
    main()

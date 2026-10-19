# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate the shared RoboCasa action-normalization JSON from the HDF5 release.

Scans ``RoboCasa-Cosmos-Policy/success_only`` (metadata columns only — no image
reads) and writes two stats blocks:

  * ``"robocasa"``       — the 7D delta-EE action ``actions[:, :7]``.
  * ``"robocasa_state"`` — the absolute proprio conditioning row
    ``[ee_pos(3), ee_ori(3), gripper_states[0]]`` used when ``use_state=True``.

``min``/``max`` of the **action** block are copied verbatim from the release's
``dataset_statistics.json`` (``actions_min`` / ``actions_max``) so the Cosmos3 run
normalizes the action target to exactly the scale the Cosmos-Policy RoboCasa runs
use — the two are then directly comparable. Pass ``--no-match-cosmos-policy`` to
use the freshly scanned min/max instead (they should agree; the script reports the
delta either way). ``mean``/``std``/``q01``/``q99`` are always computed here — the
release JSON carries no quantiles.

The output format (``{"metadata": ..., "<key>": {"min": [...], ...}}``) is what
:func:`cosmos_framework.data.generator.action.action_normalization.load_action_stats`
and ``action_policy_server_libero.py --action-stats-path`` both consume, so
train-time normalization and eval-time denormalization stay identical.

Usage::

    python tools/gen_robocasa_normalization.py \\
        --dataset-root /mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

_ACTION_DIM = 7
_DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "cosmos_framework/data/generator/action/datasets/robocasa_hdf5_normalization.json"
)


def _stats_block(values: np.ndarray) -> dict[str, list[float]]:
    """Per-dim stats over ``(N, D)`` samples."""
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _scan(root: Path, task_split: list[str] | None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Collect every action row and proprio row under ``root``."""
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    n_files = n_demos = 0
    for path in sorted(root.rglob("*.hdf5")):
        task_dir = path.parent.parent.name
        if task_split is not None and task_dir not in task_split:
            continue
        n_files += 1
        with h5py.File(path, "r") as f:
            for demo in f["data"].keys():
                grp = f["data"][demo]
                obs = grp["obs"]
                actions.append(np.asarray(grp["actions"][:, :_ACTION_DIM], dtype=np.float64))
                state = np.concatenate(
                    [
                        np.asarray(obs["ee_pos"], dtype=np.float64),
                        np.asarray(obs["ee_ori"], dtype=np.float64),
                        np.asarray(obs["gripper_states"], dtype=np.float64)[:, :1],
                    ],
                    axis=1,
                )
                states.append(state)
                n_demos += 1
        print(f"  {path.relative_to(root)}: {n_demos} demos so far", flush=True)
    return np.concatenate(actions, axis=0), np.concatenate(states, axis=0), n_files, n_demos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="RoboCasa-Cosmos-Policy/success_only (the dir holding the kitchen_* groups).",
    )
    ap.add_argument("--tasks", nargs="+", default=None, help="Restrict to these task names (default: all 24).")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="Output JSON path.")
    ap.add_argument(
        "--no-match-cosmos-policy",
        dest="match_cosmos_policy",
        action="store_false",
        help="Use the scanned action min/max instead of the release dataset_statistics.json values.",
    )
    args = ap.parse_args()

    print(f"Scanning {args.dataset_root} ...", flush=True)
    actions, states, n_files, n_demos = _scan(args.dataset_root, args.tasks)
    print(f"  {n_files} files, {n_demos} demos, {len(actions)} frames", flush=True)

    action_block = _stats_block(actions)
    state_block = _stats_block(states)

    release_path = args.dataset_root / "dataset_statistics.json"
    matched = False
    if release_path.is_file():
        release = json.loads(release_path.read_text())
        rel_min = np.asarray(release["actions_min"], dtype=np.float64)
        rel_max = np.asarray(release["actions_max"], dtype=np.float64)
        d_min = np.abs(rel_min - np.asarray(action_block["min"])).max()
        d_max = np.abs(rel_max - np.asarray(action_block["max"])).max()
        print(f"  release dataset_statistics.json: max|Δmin|={d_min:.3e} max|Δmax|={d_max:.3e}")
        if args.match_cosmos_policy:
            action_block["min"] = rel_min.tolist()
            action_block["max"] = rel_max.tolist()
            action_block["mean"] = list(release["actions_mean"])
            action_block["std"] = list(release["actions_std"])
            matched = True
            print("  -> action min/max/mean/std taken from the release (Cosmos-Policy parity).")
    elif args.match_cosmos_policy:
        raise FileNotFoundError(
            f"{release_path} not found; pass --no-match-cosmos-policy to use the scanned action stats."
        )

    out = {
        "metadata": {
            "embodiment_type": "robocasa",
            "dataset_class": "RoboCasaHDF5Dataset",
            "dataset_root": str(args.dataset_root),
            "tasks": args.tasks,
            "num_files": n_files,
            "num_demos": n_demos,
            "num_frames": int(len(actions)),
            "action_dim": _ACTION_DIM,
            "action_layout": "[dpos(3), daxang(3), gripper] delta-EE command (actions[:, :7])",
            "state_layout": "[ee_pos(3), ee_ori(3), gripper_states[0]] absolute proprio",
            "action_minmax_source": "dataset_statistics.json" if matched else "scan",
            "_comment": (
                "Action min/max match the Cosmos-Policy RoboCasa runs so both stacks train on the "
                "same target scale. The state block is separate: the absolute kitchen pose range is "
                "far outside the delta-action range and would be clamped away by the action stats."
            ),
        },
        "robocasa": action_block,
        "robocasa_state": state_block,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

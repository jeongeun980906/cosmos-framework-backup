"""Round-trip unit test: does the server's action57/state57 integration EXACTLY invert
the converter's delta extraction (tools/convert_taskdemos.py / egoverse_pair_builder
73_gr1_to_egoverse57.py + 10_mimicdroid_convert.py)?

Checks, on a real gr1 query zarr:
  (A) state57[0] (absolute eef) + action57 deltas --_integrate_arm--> reproduces state57[1:]
      absolute eef poses (pos + rotation). This proves the server's _integrate_arm is the
      exact inverse of the converter's _eef9 (body-frame delta).
  (B) grip: project state57 hand15 onto OPEN15->CLOSED15 recovers g; action57 hand15[t]
      (grip at t+1) projects to the same g as state57 hand15[t+1].
  (C) h5 cross-check: recompute abs9 from the source h5's base_to_{l,r}_eef_{pos,quat}
      and confirm it matches the zarr state57 (so the whole chain is grounded in raw obs).

Run:  .venv/bin/python tools/test_gr1_action57_roundtrip.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

# Import the ACTUAL server integration helpers so we test the real eval code path.
from cosmos_framework.scripts.action_policy_server_gr1 import (
    _CLOSED15,
    _OPEN15,
    _grip_from_hand15,
    _integrate_arm,
    _sixd_to_R,
)

EGO = "/mnt/ddn/dataset/egoverse"
GR1_H5 = "/mnt/ddn/dataset/MimicDroidDataset/GR1"


def _abs9_to_pR(abs9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """state57 abs9 = [pos3, Rcol0(3), Rcol1(3)] -> (pos, R)."""
    return abs9[:3].copy(), _sixd_to_R(abs9[3:6], abs9[6:9])


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(R.from_matrix(Ra.T @ Rb).as_rotvec())))


def check_arm(name: str, state57: np.ndarray, action57: np.ndarray, s_lo: int, a_lo: int) -> dict:
    """Integrate one arm's deltas from state[0] and compare to the stored absolute states."""
    p0, R0 = _abs9_to_pR(state57[0, s_lo:s_lo + 9])
    eef9 = action57[:, a_lo:a_lo + 9]                      # (T-1, 9) body-frame deltas
    pos_pred, aa_pred = _integrate_arm(p0, R0, eef9)       # (T-1, 3) each (server code)
    T = len(eef9)
    pos_err, rot_err = [], []
    for t in range(T):
        p_gt, R_gt = _abs9_to_pR(state57[t + 1, s_lo:s_lo + 9])
        R_pred = R.from_rotvec(aa_pred[t]).as_matrix()
        pos_err.append(np.linalg.norm(pos_pred[t] - p_gt))
        rot_err.append(_rot_angle_deg(R_pred, R_gt))
    return {"pos_cm": 100 * np.max(pos_err), "pos_mean_cm": 100 * np.mean(pos_err),
            "rot_deg": float(np.max(rot_err)), "rot_mean_deg": float(np.mean(rot_err))}


def check_grip(name: str, state57: np.ndarray, action57: np.ndarray, s_lo: int, a_lo: int) -> dict:
    """state57 hand15 -> g; action57 hand15[t] (=grip t+1) should match state g[t+1]."""
    g_state = np.array([_grip_from_hand15(h) for h in state57[:, s_lo:s_lo + 15]])
    g_act = np.array([_grip_from_hand15(h) for h in action57[:, a_lo:a_lo + 15]])
    # algebraic self-consistency: re-lerp g_state and project back == identity
    relerp = np.stack([(1 - g) * _OPEN15 + g * _CLOSED15 for g in g_state])
    g_round = np.array([_grip_from_hand15(h) for h in relerp])
    return {"g_state_vs_action_t1": float(np.max(np.abs(g_act - g_state[1:]))),
            "lerp_roundtrip_err": float(np.max(np.abs(g_round - g_state))),
            "g_range": (float(g_state.min()), float(g_state.max()))}


def _abs9_from_pq(p, q):
    Rm = R.from_quat(q / (np.linalg.norm(q) + 1e-9)).as_matrix()
    return np.concatenate([p, Rm[:, 0], Rm[:, 1]]), Rm


def _eef9_from_h5(p, q):
    """Replicate the converter's _eef9: body-frame deltas from raw base_to_eef (pos,quat xyzw)."""
    Rm = R.from_quat(q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)).as_matrix()
    out = []
    for t in range(len(p) - 1):
        dpos = Rm[t].T @ (p[t + 1] - p[t])
        dR = Rm[t].T @ Rm[t + 1]
        out.append(np.concatenate([dpos, dR[:, 0], dR[:, 1]]))
    return np.asarray(out), Rm


def check_h5(ep: str, state57: np.ndarray) -> dict | None:
    """Ground in raw obs: from the GR1 h5 base_to_eef, (1) abs9 matches zarr state57, and
    (2) eef9 deltas re-integrated via the SERVER code reproduce the h5 absolute eef poses."""
    import h5py
    src = zarr.open_group(os.path.join(EGO, ep), mode="r").attrs.get("source")  # 'GR1/<task>/demo_k'
    if not src:
        return None
    task, demo = src.split("/")[1], src.split("/")[2]
    hits = glob.glob(f"{GR1_H5}/{task}/*/demo_im128_notp.hdf5")
    if not hits:
        return {"note": f"no h5 under {GR1_H5}/{task}"}
    d = h5py.File(hits[0], "r")["data"]
    if demo not in d:
        return {"note": f"demo {demo} not in h5 ({list(d.keys())[:3]}...)"}
    o = d[demo]["obs"]
    Lp = np.asarray(o["robot0_base_to_left_eef_pos"]); Lq = np.asarray(o["robot0_base_to_left_eef_quat"])
    n = min(len(Lp), len(state57))
    # (1) raw-obs abs9 == zarr state57 abs9_L
    grounding = max(np.linalg.norm(_abs9_from_pq(Lp[t], Lq[t])[0] - state57[t, 9:18]) for t in range(n))
    # (2) h5 -> eef9 -> server _integrate_arm -> back to h5 absolute eef
    eef9, Rm = _eef9_from_h5(Lp, Lq)
    pos_pred, aa_pred = _integrate_arm(Lp[0], Rm[0], eef9)
    inv_pos = max(np.linalg.norm(pos_pred[t] - Lp[t + 1]) for t in range(len(eef9)))
    inv_rot = max(_rot_angle_deg(R.from_rotvec(aa_pred[t]).as_matrix(), Rm[t + 1]) for t in range(len(eef9)))
    return {"h5_abs9_vs_zarr_cm": 100 * grounding,
            "h5_inverse_pos_cm": 100 * inv_pos, "h5_inverse_rot_deg": inv_rot}


def main() -> None:
    eps = sorted(os.path.basename(p) for p in glob.glob(f"{EGO}/*")
                 if os.path.isdir(p) and "__demo_" in os.path.basename(p)
                 and zarr.open_group(p, mode="r").attrs.get("embodiment") == "GR1")[:5]
    assert eps, "no GR1 zarr episodes found"
    print(f"Testing {len(eps)} GR1 episodes\n")
    worst_pos = worst_rot = worst_grip = 0.0
    for ep in eps:
        g = zarr.open_group(os.path.join(EGO, ep), mode="r")
        st = np.asarray(g["state57"]["state"]); ac = np.asarray(g["action57"]["action"])
        L = check_arm("L", st, ac, 9, 9)
        Rr = check_arm("R", st, ac, 33, 33)
        gL = check_grip("L", st, ac, 18, 18)
        gR = check_grip("R", st, ac, 42, 42)
        h5 = check_h5(ep, st)
        worst_pos = max(worst_pos, L["pos_cm"], Rr["pos_cm"])
        worst_rot = max(worst_rot, L["rot_deg"], Rr["rot_deg"])
        worst_grip = max(worst_grip, gL["g_state_vs_action_t1"], gR["g_state_vs_action_t1"])
        print(f"[{ep}]")
        print(f"  L integrate: pos max {L['pos_cm']:.4f} cm (mean {L['pos_mean_cm']:.4f}) | "
              f"rot max {L['rot_deg']:.4f}° (mean {L['rot_mean_deg']:.4f})")
        print(f"  R integrate: pos max {Rr['pos_cm']:.4f} cm (mean {Rr['pos_mean_cm']:.4f}) | "
              f"rot max {Rr['rot_deg']:.4f}° (mean {Rr['rot_mean_deg']:.4f})")
        print(f"  grip L: g_act-vs-state[t+1] {gL['g_state_vs_action_t1']:.2e}  lerp-rt {gL['lerp_roundtrip_err']:.2e}  range {gL['g_range']}")
        print(f"  grip R: g_act-vs-state[t+1] {gR['g_state_vs_action_t1']:.2e}  range {gR['g_range']}")
        print(f"  h5 cross-check: {h5}")
    print(f"\n=== WORST across episodes: pos {worst_pos:.4f} cm | rot {worst_rot:.4f}° | grip {worst_grip:.2e} ===")
    ok = worst_pos < 0.1 and worst_rot < 0.5 and worst_grip < 1e-2
    print("ROUND-TRIP", "PASS ✓" if ok else "FAIL ✗ — integration is NOT the exact inverse")


if __name__ == "__main__":
    main()

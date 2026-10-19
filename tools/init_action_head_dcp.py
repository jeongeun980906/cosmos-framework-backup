# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compose the action-head row a RoboCasa run indexes, out of EgoVerse's trained row.

Why this exists
---------------
The EgoVerse warm start ("mid") trains the backbone to speak through **domain 3's**
57-channel action head. RoboCasa post-train then asks that same backbone to speak
through **domain 5's** row, which the public Edge base never trained
(``llm2action.bias[5]`` is exactly 0). The backbone is warm; its mouth and ears are
brand new. That mismatch is the reason mid starts *above* base on action loss.

The obvious fix -- slice domain 3's right-arm channels into 0..6 -- was tried
(``remapS``) and was by far the worst run of the search (0.450 -> 1.40, grad 99 -> 229).
It fails because ``action2llm`` at domain 3 expects all 57 rows to be driven. A robot
episode always drives them: ``head9`` and the unused arm sit at the constant ``IDENT9``,
and each ``hand15`` block sits on the ``OPEN15 -> CLOSED15`` segment. Feeding 7 of 57
channels silently deletes every one of those constant contributions, so the backbone
receives an input it has never seen.

But every deleted contribution is *constant*, and ``DomainAwareLinear`` carries a
per-domain bias on both projections (``action2llm.bias`` is [num_domains, 2048]). So the
whole thing folds exactly.

The map
-------
EgoVerse action57 = ``[head9 | Leef9 | Lhand15 | Reef9 | Rhand15]`` and
``eef9 = [dpos(3), R[:,0](3), R[:,1](3)]``. Against RoboCasa's
``[pos(3), axisangle(3), gripper]``:

  pos       -> Reef9[0:3]                          identity
  axisangle -> Reef9[3:9], via R ~= I + [w]x       IDENT9 constant + a 6x3 linear part
               R[:,0] = [1, wz, -wy], R[:,1] = [-wz, 1, wx]
  gripper   -> Rhand15 = OPEN15 + (g+1)/2 * axis   midpoint constant + a 15x1 linear part
  head9, Leef9, Lhand15                            constant (IDENT9 / hand15(0.5))

So ``a57 = c + M^T a7`` exactly, with ``M`` 7x57 and ``c`` the constant 57-vector, and
the head composes as a change of basis rather than a slice::

    action2llm.W[0:7]  = M @ W3[0:57]              llm2action.W[:,0:7]  = W3[:,0:57] @ M+
    action2llm.bias   += c @ W3[0:57]              llm2action.bias[0:7] = (b3[0:57]-c) @ M+

``M``'s rows have disjoint support, so ``M+ = M^T diag(1/||row||^2)`` and the readout is
the textbook inverse. Every one of the 57 channels contributes; nothing is dropped.

Aligning the scales (``--align``, default ``std``)
------------------------------------------------
The geometric relation above holds between *raw* quantities, and neither stack trains on
raw ones. RoboCasa min-max normalizes controller commands that already live in [-1,1];
EgoVerse min-max normalizes metres and rotation-matrix entries. Composing the raw map
into normalized weights is wrong by 4x-55x, per channel and non-uniformly, so no single
output gain can repair it.

Matching the min-max *ranges* (``--align range``, the exact affine) is not right either:
EgoVerse's ranges are set by outliers, so its normalized actions have std ~0.01-0.04
while RoboCasa's are 0.10-0.96. Range matching would hand the backbone inputs an order
of magnitude outside anything it saw.

``--align std`` matches the second moment instead, which is what "in distribution" means
here. Each structural entry becomes ``dir[i,j] * sigma57[j] / sigma7[i]`` with both
sigmas measured in their own normalized space, and the constant folds as
``c[j] = mu57[j] - sum_i Mfinal[i,j] * mu7[i]``, so a RoboCasa action arrives on the
EgoVerse channels with EgoVerse's own mean and spread. The hand15 block works channel by
channel: its 15 entries are perfectly correlated with the grip scalar, so ``dir`` is the
sign of ``CLOSED15-OPEN15`` and each channel keeps its own observed std.

``--align raw`` keeps the pure geometry, for attribution only.

Other modes exist to attribute the effect: ``copy`` (plain slice, i.e. remapS),
``blend`` (interpolate a trained row toward the untrained one), and ``split``
(action2llm from one source, llm2action from another -- the two halves are not
symmetric, domain 3's action2llm is already near domain 5's scale while its llm2action
is ~3.7x larger).

Usage::

    python tools/init_action_head_dcp.py --src <mid-dcp> --out <dcp> \\
        ego57 --src-domain 3 --dst-domain 5 --rescale-to 5
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

# docs/action57_state57_convention.md sec.3f -- the hand15 lerp endpoints.
OPEN15 = torch.tensor(
    [0.0005, 0.0958, -0.034, 0.0285, 0.0222, -0.1263, 0.0085, 0.0217, -0.1285,
     -0.0116, 0.0222, -0.1282, -0.0314, 0.0229, -0.1259], dtype=torch.float64)
CLOSED15 = torch.tensor(
    [0.0052, 0.0775, -0.0701, 0.0266, 0.044, -0.0916, 0.0085, 0.0441, -0.1016,
     -0.0097, 0.044, -0.0937, -0.0278, 0.044, -0.0919], dtype=torch.float64)
IDENT9 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float64)
HEAD, LEEF, LHAND, REEF, RHAND = 0, 9, 18, 33, 42
EGO_DIM = 57


def ego57_basis() -> tuple[torch.Tensor, torch.Tensor]:
    """Structural map ``(dirs, c_raw)``: ``a57_raw = c_raw + dirs.T @ a7_raw``.

    Entries are the geometric directions only (+-1, and the hand15 lerp axis); scale is
    applied later by :func:`align`, because the two stacks train on differently
    normalized quantities.
    """
    axis = CLOSED15 - OPEN15
    mid15 = OPEN15 + 0.5 * axis          # gripper command 0.0 -> g = 0.5

    c = torch.zeros(EGO_DIM, dtype=torch.float64)
    c[HEAD:HEAD + 9] = IDENT9            # fixed camera -> zero-motion delta
    c[LEEF:LEEF + 9] = IDENT9            # unused arm holds still
    c[LHAND:LHAND + 15] = mid15
    c[REEF:REEF + 9] = IDENT9            # pos part is 0, rotation part is identity
    c[RHAND:RHAND + 15] = mid15

    M = torch.zeros(7, EGO_DIM, dtype=torch.float64)
    for i in range(3):                   # pos -> Reef9 dpos
        M[i, REEF + i] = 1.0
    M[3, REEF + 8] = 1.0                 # wx  ->  R[:,1][2]
    M[4, REEF + 5] = -1.0                # wy  -> -R[:,0][2]
    M[5, REEF + 4] = 1.0                 # wz  ->  R[:,0][1]
    M[5, REEF + 6] = -1.0                #      -> -R[:,1][0]
    M[6, RHAND:RHAND + 15] = 0.5 * axis  # gripper in [-1,1] -> the lerp's linear part
    return M, c


def _minmax(stats: dict, n: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(offset, scale)`` of the min-max normalizer both stacks use."""
    lo = torch.tensor(stats["min"], dtype=torch.float64)[:n]
    hi = torch.tensor(stats["max"], dtype=torch.float64)[:n]
    scale = (hi - lo) / 2
    return (hi + lo) / 2, torch.where(scale.abs() < 1e-9, torch.ones_like(scale), scale)


def align(M: torch.Tensor, c: torch.Tensor, rc: dict, ev: dict, mode: str
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the raw structural map into the normalized space both stacks train on."""
    if mode == "raw":
        return M, c
    o7, s7 = _minmax(rc)
    o57, s57 = _minmax(ev, EGO_DIM)
    if mode == "range":
        return torch.diag(s7) @ M @ torch.diag(1 / s57), (c + M.T @ o7 - o57) / s57
    if mode != "std":
        raise ValueError(f"unknown --align {mode}")
    # Second-moment matching, in each stack's own normalized units.
    mu7 = (torch.tensor(rc["mean"], dtype=torch.float64) - o7) / s7
    sd7 = torch.tensor(rc["std"], dtype=torch.float64) / s7
    mu57 = (torch.tensor(ev["mean"], dtype=torch.float64)[:EGO_DIM] - o57) / s57
    sd57 = torch.tensor(ev["std"], dtype=torch.float64)[:EGO_DIM] / s57
    sd7 = torch.where(sd7.abs() < 1e-9, torch.ones_like(sd7), sd7)
    Mn = torch.sign(M) * sd57.unsqueeze(0) / sd7.unsqueeze(1)
    cn = mu57.clone()
    cn -= Mn.T @ mu7                      # so the composed mean lands on EgoVerse's
    # Channels no row writes keep their exact normalized constant.
    untouched = (M.abs().sum(0) == 0)
    cn[untouched] = ((c - o57) / s57)[untouched]
    return Mn, cn


def _pinv_disjoint(M: torch.Tensor) -> torch.Tensor:
    """Right pseudo-inverse of a matrix whose rows have disjoint support (57x7)."""
    n2 = M.pow(2).sum(1)
    off = M @ M.T - torch.diag(n2)
    assert off.abs().max() < 1e-9, "rows are not orthogonal"
    n2 = torch.where(n2.abs() < 1e-12, torch.ones_like(n2), n2)
    return M.T / n2


def _rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def _load(path: Path) -> dict:
    md = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype)
          for k in md if hasattr(md[k], "size")}
    dcp.load(sd, storage_reader=dcp.FileSystemReader(path), no_dist=True)
    return sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--action-dim", type=int, default=64)
    ap.add_argument("--width", type=int, default=7, help="Channels the target embodiment uses.")
    ap.add_argument("--rescale-to", type=int, default=None,
                    help="Rescale llm2action's written columns to this domain's RMS over "
                         "its 0..width-1 prefix. Keeps direction, fixes gain.")
    sub = ap.add_subparsers(dest="mode", required=True)

    e = sub.add_parser("ego57", help="Exact affine 57->7 change of basis (see module docstring).")
    e.add_argument("--src-domain", type=int, default=3)
    e.add_argument("--dst-domain", type=int, default=5)
    e.add_argument("--dst-offset", type=int, default=0)
    e.add_argument("--align", choices=("std", "range", "raw"), default="std",
                   help="How to reconcile the two stacks' normalizers (see module docstring).")
    e.add_argument("--rc-stats", default="cosmos_framework/data/generator/action/datasets/"
                                         "robocasa_hdf5_normalization.json")
    e.add_argument("--rc-key", default="robocasa")
    e.add_argument("--ev-stats", default="cosmos_framework/data/generator/action/datasets/"
                                         "egoverse_normalization.json")
    e.add_argument("--ev-key", default="mimicdroid",
                   help="EgoVerse source whose stats stand in for a robot arm.")

    c_ = sub.add_parser("copy", help="Plain channel-window slice (this is what remapS did).")
    c_.add_argument("--src-domain", type=int, required=True)
    c_.add_argument("--src-offset", type=int, default=0)
    c_.add_argument("--dst-domain", type=int, required=True)
    c_.add_argument("--dst-offset", type=int, default=0)

    b = sub.add_parser("blend", help="alpha*A + (1-alpha)*B into the destination row.")
    b.add_argument("--a-domain", type=int, required=True)
    b.add_argument("--a-offset", type=int, default=0)
    b.add_argument("--b-domain", type=int, required=True)
    b.add_argument("--b-offset", type=int, default=0)
    b.add_argument("--dst-domain", type=int, required=True)
    b.add_argument("--dst-offset", type=int, default=0)
    b.add_argument("--alpha", type=float, required=True)

    s = sub.add_parser("split", help="action2llm and llm2action from different sources.")
    s.add_argument("--a2l-domain", type=int, required=True)
    s.add_argument("--a2l-offset", type=int, default=0)
    s.add_argument("--l2a-domain", type=int, required=True)
    s.add_argument("--l2a-offset", type=int, default=0)
    s.add_argument("--dst-domain", type=int, required=True)
    s.add_argument("--dst-offset", type=int, default=0)

    args = ap.parse_args()
    A, W = args.action_dim, args.width
    print(f"loading {args.src} ...")
    sd = _load(args.src)

    for prefix in ("net.", "net_ema."):
        ka_w, ka_b = f"{prefix}action2llm.fc.weight", f"{prefix}action2llm.bias.weight"
        kl_w, kl_b = f"{prefix}llm2action.fc.weight", f"{prefix}llm2action.bias.weight"
        if ka_w not in sd:
            continue
        H = sd[ka_w].shape[1] // A
        va = lambda d: sd[ka_w][d].double().view(A, H)      # noqa: E731  rows are in-channels
        vl = lambda d: sd[kl_w][d].double().view(H, A)      # noqa: E731  cols are out-channels
        dst, doff = args.dst_domain, args.dst_offset

        new_a, new_ab = va(dst).clone(), sd[ka_b][dst].double().clone()
        new_l, new_lb = vl(dst).clone(), sd[kl_b][dst].double().clone()

        if args.mode == "ego57":
            src = args.src_domain
            M, c = ego57_basis()
            M, c = align(M, c, json.load(open(args.rc_stats))[args.rc_key],
                         json.load(open(args.ev_stats))[args.ev_key], args.align)
            Mp = _pinv_disjoint(M)
            Wa, Wl = va(src)[:EGO_DIM], vl(src)[:, :EGO_DIM]
            new_a[doff:doff + 7] = M @ Wa
            new_ab = sd[ka_b][src].double() + c @ Wa      # the constant channels, folded in
            new_l[:, doff:doff + 7] = Wl @ Mp
            new_lb[doff:doff + 7] = (sd[kl_b][src].double()[:EGO_DIM] - c) @ Mp
            what = (f"ego57/{args.align} dom{src}[0:57] -> dom{dst}[{doff}:{doff+7}]"
                    f"  (+{_rms(c @ Wa):.4f} into a2l bias)")
        elif args.mode == "copy":
            so = args.src_offset
            new_a[doff:doff + W] = va(args.src_domain)[so:so + W]
            new_l[:, doff:doff + W] = vl(args.src_domain)[:, so:so + W]
            new_lb[doff:doff + W] = sd[kl_b][args.src_domain].double()[so:so + W]
            what = f"copy dom{args.src_domain}[{so}:{so+W}] -> dom{dst}[{doff}:{doff+W}]"
        elif args.mode == "blend":
            al, ao, bo = args.alpha, args.a_offset, args.b_offset
            mix = lambda x, y: al * x + (1 - al) * y       # noqa: E731
            new_a[doff:doff + W] = mix(va(args.a_domain)[ao:ao + W], va(args.b_domain)[bo:bo + W])
            new_l[:, doff:doff + W] = mix(vl(args.a_domain)[:, ao:ao + W], vl(args.b_domain)[:, bo:bo + W])
            new_lb[doff:doff + W] = mix(sd[kl_b][args.a_domain].double()[ao:ao + W],
                                        sd[kl_b][args.b_domain].double()[bo:bo + W])
            new_ab = mix(sd[ka_b][args.a_domain].double(), sd[ka_b][args.b_domain].double())
            what = (f"blend {al:.2f}*dom{args.a_domain}[{ao}:] + {1-al:.2f}*dom{args.b_domain}[{bo}:]"
                    f" -> dom{dst}[{doff}:]")
        else:  # split
            ao, lo = args.a2l_offset, args.l2a_offset
            new_a[doff:doff + W] = va(args.a2l_domain)[ao:ao + W]
            new_ab = sd[ka_b][args.a2l_domain].double()
            new_l[:, doff:doff + W] = vl(args.l2a_domain)[:, lo:lo + W]
            new_lb[doff:doff + W] = sd[kl_b][args.l2a_domain].double()[lo:lo + W]
            what = (f"split a2l<-dom{args.a2l_domain}[{ao}:]  l2a<-dom{args.l2a_domain}[{lo}:]"
                    f" -> dom{dst}[{doff}:]")

        if args.rescale_to is not None:
            tgt = _rms(vl(args.rescale_to)[:, :W])
            cur = _rms(new_l[:, doff:doff + W])
            if cur > 0:
                k = tgt / cur
                new_l[:, doff:doff + W] *= k
                new_lb[doff:doff + W] *= k
                what += f"  (l2a x{k:.4f} -> dom{args.rescale_to} RMS)"

        sd[ka_w][dst] = new_a.reshape(-1).to(sd[ka_w].dtype)
        sd[ka_b][dst] = new_ab.to(sd[ka_b].dtype)
        sd[kl_w][dst] = new_l.reshape(-1).to(sd[kl_w].dtype)
        sd[kl_b][dst] = new_lb.to(sd[kl_b].dtype)
        h = torch.randn(2048, H, dtype=torch.float64)
        print(f"  {prefix:<9} {what}")
        print(f"  {'':<9} l2a RMS {_rms(new_l[:, doff:doff+W]):.5f}  out std "
              f"{(h @ new_l[:, doff:doff+W]).std():.4f}   a2l RMS {_rms(new_a[doff:doff+W]):.5f}"
              f"  a2l bias RMS {_rms(new_ab):.5f}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {args.out} ...")
    dcp.save(sd, storage_writer=dcp.FileSystemWriter(args.out), no_dist=True)
    cfg = args.src / "config.json"
    if cfg.is_file():
        shutil.copy2(cfg, args.out / "config.json")
    print("done")


if __name__ == "__main__":
    main()

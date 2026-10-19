# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bind every local GPU before torchrun, and name the node if one is unusable.

Two multi-node applies died inside ``distributed.init()`` at its very first statement,
``torch.cuda.set_device(local_rank)``, with ``cudaErrorDevicesUnavailable`` — both times
on local rank 5, once on worker_3 and once on worker_0. The pod requests all eight GPUs
(``nvidia.com/gpu: 8``, requests == limits), so a device reported "busy or unavailable"
is held by something the device plugin does not account for, typically a process left
behind by a job that died without releasing it.

Letting torchrun discover this is workable but hard to act on: the traceback names the
local rank and never the host, so identifying the node to exclude means racing kubectl
against a pod that is already terminating. Binding here instead turns a rendezvous-time
crash into a one-line verdict in the pod's own log, with the hostname in it.

Exit 0 if every device binds, 1 otherwise. Run it after the all-nodes-ready barrier and
before torchrun, so a bad host fails fast rather than hanging its peers in rendezvous.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    import torch

    node = os.environ.get("NODE_NAME", "unknown")
    rank = os.environ.get("NODE_RANK", "?")
    n = int(os.environ.get("GPU_PREFLIGHT_COUNT", "8"))

    visible = torch.cuda.device_count()
    if visible < n:
        print(f"FATAL: {node} (NODE_RANK={rank}) sees {visible} GPUs, expected {n} "
              f"— EXCLUDE THIS NODE", flush=True)
        return 1

    bad: list[tuple[int, str, str]] = []
    for i in range(n):
        try:
            torch.cuda.set_device(i)
            torch.zeros(1, device=f"cuda:{i}")
            torch.cuda.synchronize(i)
        except Exception as e:  # noqa: BLE001 — any failure here disqualifies the node
            bad.append((i, type(e).__name__, str(e).splitlines()[0][:100]))

    if bad:
        print(f"FATAL: {node} (NODE_RANK={rank}) cannot bind {len(bad)} of {n} GPUs "
              f"— EXCLUDE THIS NODE", flush=True)
        for i, kind, msg in bad:
            print(f"    {node} GPU {i}: {kind}: {msg}", flush=True)
        return 1

    free = [round(torch.cuda.mem_get_info(i)[0] / 2**30, 1) for i in range(n)]
    print(f">>> {node} (NODE_RANK={rank}): all {n} GPUs bind OK; free GiB {free}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

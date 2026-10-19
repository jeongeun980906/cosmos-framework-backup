# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain ID helpers for cross-embodiment action datasets."""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    # RoboCasa (24-task kitchen) shares LIBERO's domain id on purpose: both are a
    # Panda arm with a parallel gripper driven by the SAME 7D delta-EE axis-angle
    # command, so the pretrained embodiment embedding transfers directly. (Same
    # rationale as droid_lerobot/robomind-franka sharing id 8.)
    "robocasa": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,  # Both Droid and RoboMIND-Franka are using robotiq and franka
    "embodiment_b": 9,
    "robomind-franka-dual": 12,
    # RoboMIND's bimanual robot is the AgileX Cobot Magic, not a dual Franka — the name
    # above predates the data. Same 20-D contract [Lpos3, Lrot6d6, Lgrip, Rpos3,
    # Rrot6d6, Rgrip], so it shares domain 12, which the public Edge base already
    # trained (20 non-zero bias entries, exactly its width).
    "robomind-agilex-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "embodiment_c_gripper": 15,
    "embodiment_c_gripper_ext": 15,
    "xdof_yam": 16,
    "molmoact2_yam": 16,  # MolmoAct2 uses the same YAM 20D FK action contract
    "abc_yam": 16,  # ABC uses the same YAM 20D FK action contract
    "eva": 16,  # EgoVerse eva_bimanual robot (20D cartesian cmd + gripper)
    # Tabletop-Sim ALOHA (bimanual). ``actions/ee_6d_pos`` is [T, 20] = per arm
    # [pos(3), rot6d(6), grip(1)] x 2, the same channel contract row 16 already carries
    # for the YAM/eva 20D embodiments. Row 16 is ALSO untrained in every checkpoint we
    # compare (public Edge base, the ICL mid, and the BC-replay mid all have
    # ``llm2action.bias[16]`` identically 0), which is what makes it the 20D analogue of
    # row 5 for RoboCasa: the head starts random for all three, so a probe through it
    # measures the BACKBONE and nothing else.
    "aloha-tabletop": 16,
    "fractal": 20,
    "drawanything": 21,
}


EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-agilex-dual": 20,
    "robomind-ur": 10,
    "embodiment_b": 30,
    "agibotworld": 29,
    "embodiment_c_gripper": 29,
    "embodiment_c_gripper_ext": 29,
    "xdof_yam": 20,
    "molmoact2_yam": 20,
    "abc_yam": 20,
    "eva": 20,  # EgoVerse eva_bimanual: [L ee_pose(6) + L grip(1) + R ee_pose(6) + R grip(1)] ≈ 20D
    "aloha-tabletop": 20,  # [L pos(3)+rot6d(6)+grip(1)] x 2, from actions/ee_6d_pos
    "fractal": 10,
    "drawanything": 3,
    "libero": 7,  # [dpos(3), daxang(3), gripper] delta-EE command (local axis-angle recipe)
    "robocasa": 7,  # same 7D delta-EE command; actions[:, 7:] (mobile base) is dropped
    # NOTE: upstream leaves ``libero`` out (7/10/13 depending on ``rotation_space``);
    # the local LIBERO recipe always uses the 7D delta-EE axis-angle command, so the
    # canonical width is pinned here for the action-policy server / eval clients.
    # NOTE: ``hand_pose``
    # (variable with ``keypoint_option`` and ``rotation_format``) are absent
    # because their raw width is set per-dataset at construction time. Inference
    # in inverse_dynamics/WAM modes is not supported for these domains until
    # canonical widths are added here.
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


def get_action_dim(embodiment_type: str) -> int:
    """Get the raw action dimension for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_RAW_ACTION_DIM.keys())}"
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


def is_valid_domain_name(embodiment_type: str) -> bool:
    """Check if the given embodiment type is recognized."""
    key = embodiment_type.lower().strip()
    return key in EMBODIMENT_TO_RAW_ACTION_DIM

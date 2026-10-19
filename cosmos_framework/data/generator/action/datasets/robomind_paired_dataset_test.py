# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the RoboMIND ICL pair descriptor and the ``[demo, query]`` assembly."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robomind_franka_dataset import RoboMINDFrankaDataset
from cosmos_framework.data.generator.action.datasets.robomind_paired_dataset import (
    RoboMINDPairedDataset,
    TaskPairDescriptor,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan


class _RowsStub:
    """Enough of ``RoboMINDFrankaDataset`` for the index arithmetic under test."""

    def __init__(self, episode_lengths: list[int], chunk_length: int, sample_stride: int = 1) -> None:
        self._rows = [
            {"episode_index": ep, "index": i}
            for ep, n in enumerate(episode_lengths)
            for i in range(n)
        ]
        for i, row in enumerate(self._rows):
            row["index"] = i
        self._chunk_length = chunk_length
        self._sample_stride = sample_stride

    def __len__(self) -> int:
        return max(0, (len(self._rows) - self._chunk_length + self._sample_stride - 1) // self._sample_stride)


def _spans(episode_lengths: list[int], chunk_length: int, sample_stride: int = 1):
    stub = _RowsStub(episode_lengths, chunk_length, sample_stride)
    return stub, RoboMINDFrankaDataset.episode_window_spans(stub)


def test_spans_cover_only_windows_inside_one_episode():
    # Three 10-frame episodes, chunk 4 => a window spans rows [i, i+4].
    stub, (ids, starts, ends) = _spans([10, 10, 10], chunk_length=4)
    assert ids.tolist() == [0, 1, 2]
    # Episode 0 owns rows 0..9, so windows 0..5 stay inside it (window 5 ends at row 9).
    assert (starts.tolist(), ends.tolist()) == ([0, 10, 20], [6, 16, 26])
    for start, end, ep in zip(starts, ends, ids):
        for w in range(int(start), int(end)):
            first, last = w * stub._sample_stride, w * stub._sample_stride + stub._chunk_length
            assert stub._rows[first]["episode_index"] == ep
            assert stub._rows[last]["episode_index"] == ep


def test_spans_drop_episodes_shorter_than_one_window():
    # Episode 1 has 3 rows but a window needs chunk+1 = 5, so it contributes nothing.
    _, (ids, starts, ends) = _spans([10, 3, 10], chunk_length=4)
    assert ids.tolist() == [0, 2]
    assert np.all(ends > starts)


def test_spans_respect_sample_stride():
    stub, (ids, starts, ends) = _spans([10, 10], chunk_length=4, sample_stride=3)
    for start, end, ep in zip(starts, ends, ids):
        for w in range(int(start), int(end)):
            first, last = w * 3, w * 3 + 4
            assert stub._rows[first]["episode_index"] == ep
            assert stub._rows[last]["episode_index"] == ep


def _descriptor(n_episodes: int, keys: np.ndarray | None = None) -> TaskPairDescriptor:
    return TaskPairDescriptor(
        episode_ids=np.arange(n_episodes, dtype=np.int64),
        starts=np.arange(n_episodes, dtype=np.int64) * 10,
        ends=np.arange(n_episodes, dtype=np.int64) * 10 + 10,
        keys=keys,
    )


class _KeyStub(_RowsStub):
    """``_RowsStub`` plus a retrieval key per window, for descriptor build tests."""

    def __init__(self, episode_lengths, chunk_length, features):
        super().__init__(episode_lengths, chunk_length)
        self._features = np.asarray(features, dtype=np.float32)

    def window_retrieval_keys(self):
        return self._features

    def retrieval_key_names(self):
        names = ["pos_x", "pos_y", "pos_z", "gripper", "disp_x", "disp_y", "disp_z", "grasps"]
        return names[: self._features.shape[1]]

    def episode_window_spans(self):
        return RoboMINDFrankaDataset.episode_window_spans(self)


def test_other_episode_never_returns_the_query_episode():
    desc = _descriptor(5)
    rng = random.Random(0)
    for _ in range(500):
        for slot in range(5):
            assert desc.sample_other_episode(slot, rng) != slot


def test_other_episode_reaches_every_other_episode():
    desc = _descriptor(4)
    rng = random.Random(1)
    seen = {desc.sample_other_episode(2, rng) for _ in range(200)}
    assert seen == {0, 1, 3}


def test_single_episode_task_falls_back_to_itself():
    desc = _descriptor(1)
    assert desc.sample_other_episode(0, random.Random(0)) == 0


def test_slot_of_reports_missing_episode():
    desc = TaskPairDescriptor(
        episode_ids=np.array([0, 2, 5], dtype=np.int64),
        starts=np.array([0, 10, 20], dtype=np.int64),
        ends=np.array([10, 20, 30], dtype=np.int64),
    )
    assert desc.slot_of(2) == 1
    assert desc.slot_of(1) == -1
    assert desc.slot_of(9) == -1


def _one_clip(frames: int = 5, dim: int = 4) -> dict:
    return {
        "video": torch.randint(0, 255, (3, frames, 16, 16), dtype=torch.uint8),
        "action": torch.randn(frames - 1, dim),
        "image_size": torch.tensor([16, 16, 16, 16]),
        "domain_id": torch.tensor(8, dtype=torch.long),
        "raw_action_dim": torch.tensor(dim, dtype=torch.long),
        "conditioning_fps": torch.tensor(30, dtype=torch.long),
        "ai_caption": "put the marker in the box",
        "sequence_plan": SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=True,
            condition_frame_indexes_vision=[0],
            condition_frame_indexes_action=[],
            action_start_frame_offset=1,
        ),
    }


def _bare_paired(share_temporal: bool = True) -> RoboMINDPairedDataset:
    ds = object.__new__(RoboMINDPairedDataset)
    ds._share_temporal = share_temporal
    ds._text_tokenizer = None
    return ds


def test_assemble_emits_the_two_item_icl_layout():
    q, c = _one_clip(), _one_clip()
    out = _bare_paired()._assemble(q, c, "robomind-franka")

    assert [len(out[k]) for k in ("video", "action", "domain_id", "raw_action_dim", "image_size")] == [2] * 5
    # Demo first, query last: the packer supervises only the final item.
    assert torch.equal(out["video"][1], q["video"].float().div(127.5).sub(1.0))
    assert torch.equal(out["action"][1], q["action"])
    assert out["conditioning_fps"].shape == (2,)
    assert out["num_frames"] == q["video"].shape[1]
    assert out["dataset_name"] == "robomind_paired"


def test_assemble_makes_both_vision_items_float():
    out = _bare_paired()._assemble(_one_clip(), _one_clip(), "robomind-franka")
    # A mixed uint8/float list makes the model take the raw path and then assert.
    assert {v.dtype for v in out["video"]} == {torch.float32}
    assert all(float(v.min()) >= -1.0 and float(v.max()) <= 1.0 for v in out["video"])


def test_assemble_keeps_the_query_plan_and_turns_on_the_shared_grid():
    q, c = _one_clip(), _one_clip()
    out = _bare_paired()._assemble(q, c, "robomind-franka")
    plan = out["sequence_plan"]
    assert plan.share_vision_temporal_positions is True
    assert plan.condition_frame_indexes_vision == q["sequence_plan"].condition_frame_indexes_vision
    assert plan.action_start_frame_offset == q["sequence_plan"].action_start_frame_offset
    assert q["sequence_plan"].share_vision_temporal_positions is False  # query plan not mutated


def test_assemble_appends_the_demo_and_target_clauses():
    q, c = _one_clip(), _one_clip()
    out = _bare_paired()._assemble(q, c, "robomind-franka")
    assert out["ai_caption"] == f"{q['ai_caption']} demo robomind-franka target robomind-franka"


def test_assemble_rejects_a_shape_mismatch_under_the_shared_grid():
    q, c = _one_clip(frames=5), _one_clip(frames=9)
    with pytest.raises(ValueError, match="share_temporal"):
        _bare_paired()._assemble(q, c, "robomind-franka")
    # Without the shared grid, differing lengths are legal (each item keeps its own grid).
    out = _bare_paired(share_temporal=False)._assemble(q, c, "robomind-franka")
    assert out["video"][0].shape[1] == 9 and out["video"][1].shape[1] == 5


def test_matched_window_is_the_nearest_phase_in_the_episode():
    # 3 episodes x 10 windows; key is a 1-D "phase" ramp repeated per episode, so window
    # e*10 + t has phase t. The nearest match for phase 7 must be phase 7, not a random one.
    keys = np.tile(np.arange(10, dtype=np.float32), 3).reshape(-1, 1)
    desc = _descriptor(3, keys=keys)
    query_key = keys[7]  # phase 7 of episode 0
    for slot in (1, 2):
        picked = {desc.sample_window(slot, random.Random(s), query_key, top_k=1) for s in range(20)}
        assert picked == {slot * 10 + 7}


def test_matched_window_respects_top_k():
    keys = np.tile(np.arange(10, dtype=np.float32), 2).reshape(-1, 1)
    desc = _descriptor(2, keys=keys)
    picked = {desc.sample_window(1, random.Random(s), keys[7], top_k=3) for s in range(200)}
    # The 3 nearest phases to 7 are 6, 7, 8.
    assert picked == {16, 17, 18}


def test_unmatched_draw_is_uniform_over_the_episode():
    desc = _descriptor(2, keys=None)
    picked = {desc.sample_window(1, random.Random(s)) for s in range(500)}
    assert picked == set(range(10, 20))


def test_descriptor_build_whitens_then_weights_by_channel_meaning():
    # One arm: [x, y, z, gripper, dx, dy, dz]. x varies 10x more than the gripper in raw
    # units, so without whitening the gripper would be invisible to an L2 match.
    n = 12
    feats = np.zeros((n, 8), dtype=np.float32)
    feats[:, 0] = np.linspace(0.0, 1.0, n)       # x: raw range 1.0
    feats[:, 3] = np.linspace(0.0, 0.1, n)       # gripper: raw range 0.1
    feats[:, 4] = np.linspace(0.0, 0.01, n)      # dx: raw range 0.01
    feats[:, 7] = np.linspace(0.0, 5.0, n)       # grasps: raw range 5
    stub = _KeyStub([n + 4], chunk_length=4, features=feats)
    desc = TaskPairDescriptor.build(
        stub, match=True, gripper_weight=2.0, motion_weight=3.0, progress_weight=4.0
    )
    assert desc.keys is not None
    # Whitening equalizes the channels; the weights then scale them by meaning.
    assert desc.keys[:, 0].std() == pytest.approx(1.0, rel=1e-4)
    assert desc.keys[:, 3].std() == pytest.approx(2.0, rel=1e-4)
    assert desc.keys[:, 4].std() == pytest.approx(3.0, rel=1e-4)
    assert desc.keys[:, 7].std() == pytest.approx(4.0, rel=1e-4)


def test_descriptor_build_survives_a_constant_channel():
    feats = np.zeros((10, 8), dtype=np.float32)
    feats[:, 1] = np.linspace(0, 1, 10)  # every other channel never moves
    desc = TaskPairDescriptor.build(_KeyStub([14], chunk_length=4, features=feats), match=True)
    assert np.isfinite(desc.keys).all()


def test_match_off_leaves_the_descriptor_keyless():
    feats = np.random.RandomState(0).randn(10, 8).astype(np.float32)
    desc = TaskPairDescriptor.build(_KeyStub([14], chunk_length=4, features=feats), match=False)
    assert desc.keys is None


class _GraspStub(_RowsStub):
    """``_RowsStub`` whose gripper channel follows a scripted open/close pattern."""

    _GRASP_CLOSED = RoboMINDFrankaDataset._GRASP_CLOSED

    def __init__(self, episode_gripper: list[list[float]]):
        super().__init__([len(g) for g in episode_gripper], chunk_length=1)
        self._grip = np.concatenate([np.asarray(g, dtype=np.float32) for g in episode_gripper])

    def features(self) -> np.ndarray:
        f = np.zeros((len(self._grip), 4), dtype=np.float32)
        f[:, 3] = self._grip
        return f


def test_grasp_counts_accumulate_per_episode():
    # Two episodes; closures are rising edges of closedness > 0.5.
    stub = _GraspStub([[0, 1, 1, 0, 1, 0], [0, 0, 1, 0]])
    counts = RoboMINDFrankaDataset._row_grasp_counts(stub, stub.features())
    assert counts[:, 0].tolist() == [0, 1, 1, 1, 2, 2, 0, 0, 1, 1]


def test_grasp_counts_do_not_leak_across_an_episode_boundary():
    # Episode 0 ends CLOSED and episode 1 starts CLOSED: that is a new grasp, not a
    # continuation, and episode 1 must restart from its own first closure.
    stub = _GraspStub([[0, 1, 1], [1, 1, 0, 1]])
    counts = RoboMINDFrankaDataset._row_grasp_counts(stub, stub.features())
    assert counts[:, 0].tolist() == [0, 1, 1, 1, 1, 1, 2]


def test_grasp_counts_handle_an_episode_with_no_grasp():
    stub = _GraspStub([[0, 0, 0], [0, 1, 0]])
    counts = RoboMINDFrankaDataset._row_grasp_counts(stub, stub.features())
    assert counts[:, 0].tolist() == [0, 0, 0, 0, 1, 1]


def _ramp_descriptor(n_episodes: int, per_episode: int = 10) -> TaskPairDescriptor:
    """Episodes whose keys are ramps with a per-episode offset, so episode ``e``'s best
    match to a query of phase p sits at distance ``e``: episode 0 matches exactly, and the
    further the episode, the worse its best window."""
    keys = np.concatenate(
        [np.arange(per_episode, dtype=np.float32) + e for e in range(n_episodes)]
    ).reshape(-1, 1)
    return TaskPairDescriptor(
        episode_ids=np.arange(n_episodes, dtype=np.int64),
        starts=np.arange(n_episodes, dtype=np.int64) * per_episode,
        ends=np.arange(n_episodes, dtype=np.int64) * per_episode + per_episode,
        keys=keys,
    )


def test_more_candidate_episodes_find_a_closer_match():
    desc = _ramp_descriptor(8)
    query_key = desc.keys[35]  # episode 3, phase 5

    def median_distance(n_candidates):
        d = [
            float(
                abs(
                    desc.keys[desc.retrieve(3, random.Random(s), query_key, top_k=1,
                                            candidate_episodes=n_candidates)][0]
                    - query_key[0]
                )
            )
            for s in range(200)
        ]
        return float(np.median(d))

    # Searching more episodes can only keep or improve the best match.
    assert median_distance(4) <= median_distance(1)
    assert median_distance(8) <= median_distance(4)


def test_retrieve_never_returns_the_excluded_episode():
    desc = _ramp_descriptor(5)
    for slot in range(5):
        for seed in range(60):
            w = desc.retrieve(slot, random.Random(seed), desc.keys[slot * 10 + 4],
                              candidate_episodes=8)
            assert not (slot * 10 <= w < slot * 10 + 10)


def test_retrieve_without_keys_falls_back_to_a_uniform_draw():
    desc = _descriptor(3, keys=None)
    picked = {desc.retrieve(0, random.Random(s), None, candidate_episodes=8) for s in range(400)}
    assert picked == set(range(10, 30))  # every window of the two other episodes, none of ep 0

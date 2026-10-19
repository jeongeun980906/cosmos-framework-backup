"""cosmos3 EDGE 시퀀스 레이아웃 + 어텐션 훅 가능성 프로브 (샘플 1개).

목적: feature/attention 추출 스크립트를 쓰기 전에 아래 세 가지를 **실제 배치에서** 확정한다.
  1) packed_sequence 의 모달리티별 토큰 위치
     action.sequence_indexes / vision.sequence_indexes / vision_item_split_lens / text_indexes
     -> 우리 cosmos_policy 의 action_latent_idx, retrieved_video, current_image, future 에
        대응하는 구간을 잡을 수 있는지
  2) condition_mask (1=조건, 0=예측 대상) 로 "검색된 데모 액션" 과 "예측할 액션" 이 갈리는지
  3) 어텐션 경로. GEN 타워는 FlexAttention 이라 attention matrix 가 안 나오므로,
     flex_attention_varlen 을 감싸 q/k 를 가로채는 방식이 성립하는지 (레이어별 호출 횟수/shape)

usage:
  python tools/edge_probe_layout.py \
      --experiment action_policy_egoverse_v10_icl_edge \
      --run-root /mnt/ddn/tmp/egoverse_v10_icl_edge_4node/cosmos3_action/action_sft/action_policy_egoverse_v10_icl_edge \
      --iter 8000 --row 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.data.generator.action.datasets.egoverse_zarr_dataset import (
    EgoVersePairedDataset,
    EgoVersePairedTransform,
    EgoVerseWindowReader,
)
from cosmos_framework.data.generator.action.action_processing import ActionProcessingRecord
from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint
import cosmos_framework.model.generator.mot.attention as MOTATT

_D = "cosmos_framework/data/generator/action/datasets"


def _to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, list):
        return [_to_cuda(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_cuda(v) for k, v in x.items()}
    return x


def _spans(idx: torch.Tensor):
    """정렬된 인덱스 텐서 -> 연속 구간 목록 [(start, end_exclusive), ...]."""
    a = np.asarray(idx.detach().cpu(), np.int64)
    if a.size == 0:
        return []
    a = np.sort(a)
    brk = np.where(np.diff(a) != 1)[0]
    out, s = [], a[0]
    for b in brk:
        out.append((int(s), int(a[b]) + 1)); s = a[b + 1]
    out.append((int(s), int(a[-1]) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="action_policy_egoverse_v10_icl_edge")
    ap.add_argument("--run-root", default="/mnt/ddn/tmp/egoverse_v10_icl_edge_4node/"
                                          "cosmos3_action/action_sft/action_policy_egoverse_v10_icl_edge")
    ap.add_argument("--iter", type=int, default=8000)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data-root", default="/mnt/ddn/dataset")
    ap.add_argument("--pairs-dir", default="/mnt/ddn/dataset/egoverse_pairs_v10_test")
    ap.add_argument("--norm", default=f"{_D}/egoverse_normalization.json")
    ap.add_argument("--speed", default=f"{_D}/egoverse_speed_stats.json")
    ap.add_argument("--chunk-length", type=int, default=88)
    ap.add_argument("--vae-path", default="/mnt/ddn/ckpt/wan2.2/Wan2.2_VAE.pth")
    ap.add_argument("--row", type=int, default=0, help="dataset row (query window) index")
    ap.add_argument("--out", default="/mnt/ddn/tmp/feature_analysis/edge_layout_probe.json")
    a = ap.parse_args()

    ckpt = a.checkpoint or os.path.join(a.run_root, "checkpoints", f"iter_{a.iter:09d}")
    assert os.path.isdir(ckpt), f"checkpoint dir not found: {ckpt}"

    model, config = load_model_from_checkpoint(
        experiment_name=a.experiment, checkpoint_path=ckpt,
        parallelism_config={"data_parallel_shard_degree": 1, "data_parallel_replicate_degree": 1},
        experiment_opts=[f"model.config.tokenizer.vae_path={a.vae_path}"],
    )
    model = model.cuda().eval()
    tk_cfg = config.model.config.vlm_config.tokenizer

    reader = EgoVerseWindowReader(
        data_root=a.data_root, pairs_dir=a.pairs_dir, normalization_path=a.norm,
        speed_stats_path=a.speed, chunk_length=a.chunk_length,
        action_source="state", use_state=True,
    )
    transform = EgoVersePairedTransform(
        resolution="256", max_action_dim=config.model.config.max_action_dim,
        share_temporal=True, keep_aspect_ratio=False, tokenizer_config=tk_cfg,
    )
    ds = EgoVersePairedDataset(reader=reader, pairs_dir=a.pairs_dir, transform=transform)

    r = a.row
    q = int(ds._query_idx[r])
    ms = [int(m) for m in ds._match_idx[r] if m >= 0]
    m = q if not ms else ms[0]
    sample = transform({"query": reader.read_window(q), "condition": reader.read_window(m)})
    batch = {k: _to_cuda(v) for k, v in custom_collate_fn([sample]).items()}
    batch["action_processing_record"] = [ActionProcessingRecord(raw_action_dim=57,
                                                                action_normalizer=None)]
    if isinstance(batch.get("image_size"), list):
        batch["image_size"] = batch["image_size"][0]

    # ---- 3) 어텐션 훅 + packing 훅 ----
    # packing 은 forward 안에서 일어난다(모델에 get_data_and_condition 이 없다). 그래서
    # _pack_input_sequence 를 감싸 결과를 가로채고, 어텐션은 flex_attention_varlen 을 감싼다.
    calls, packs = [], []
    _orig_flex = MOTATT.flex_attention_varlen
    _orig_pack = type(model)._pack_input_sequence

    def _wrapped_flex(fq, fk, fv, meta, return_lse=False, **kw):
        calls.append(dict(q=tuple(fq.shape), k=tuple(fk.shape), dtype=str(fq.dtype)))
        return _orig_flex(fq, fk, fv, meta, return_lse=return_lse, **kw)

    def _wrapped_pack(self, *aa, **kk):
        out = _orig_pack(self, *aa, **kk)
        packs.append(out)
        return out

    MOTATT.flex_attention_varlen = _wrapped_flex
    type(model)._pack_input_sequence = _wrapped_pack
    try:
        with torch.no_grad():
            model.generate_samples_from_batch(batch, guidance=1.5, seed=[0],
                                              n_sample=1, num_steps=2, shift=3.0)
    except Exception as e:
        print(f"!! generate 중 예외(레이아웃만 필요하므로 무시 가능): {type(e).__name__}: {e}")
    finally:
        MOTATT.flex_attention_varlen = _orig_flex
        type(model)._pack_input_sequence = _orig_pack
    ps = packs[0] if packs else None
    print(f"\n_pack_input_sequence 호출 {len(packs)}회, flex_attention_varlen 호출 {len(calls)}회")
    if calls:
        print(f"   첫 호출 q={calls[0]['q']} k={calls[0]['k']} dtype={calls[0]['dtype']}")
        print(f"   고유 shape 수 {len({c['q'] for c in calls})}  "
              f"(레이어 수 x denoise step 수와 맞아야 한다)")

    rep = {"row": r, "qidx": q, "didx": m,
           "batch_keys": sorted([k for k in batch]),
           "flex_calls": len(calls), "flex_shapes": calls[:3],
           "pack_calls": len(packs)}
    print(json.dumps({k: v for k, v in rep.items() if k != "flex_shapes"}, indent=1))

    # ---- 1)+2) packed_sequence 구조 ----
    if ps is None:
        print("\n!! packed_sequence 를 get_data_and_condition 에서 못 꺼냈다. "
              "batch 안의 후보 키를 덤프한다:")
        for k, v in batch.items():
            print(f"   {k}: {type(v).__name__}"
                  + (f" shape={tuple(v.shape)}" if torch.is_tensor(v) else ""))
    else:
        for name in ("action", "vision", "sound"):
            md = getattr(ps, name, None)
            if md is None:
                print(f"\n[{name}] 없음"); continue
            si = md.sequence_indexes
            print(f"\n[{name}] tokens={si.numel()}  spans={_spans(si)[:8]}")
            print(f"   token_shapes={md.token_shapes[:4]}")
            cm = md.condition_mask
            if cm:
                c0 = cm[0]
                print(f"   condition_mask[0]: shape={tuple(c0.shape)} "
                      f"clean={int(c0.sum())}/{c0.numel()}")
            if getattr(md, "mse_loss_indexes", None) is not None:
                print(f"   mse_loss_indexes={md.mse_loss_indexes.numel()}개  "
                      f"spans={_spans(md.mse_loss_indexes)[:4]}")
        ti = getattr(ps, "text_indexes", None)
        if ti is not None:
            ti = torch.as_tensor(ti)
            print(f"\n[text] tokens={ti.numel()} spans={_spans(ti)[:6]}")
        vsl = getattr(ps, "vision_item_split_lens", None)
        print(f"\n[vision_item_split_lens] {vsl}")
        rep["layout"] = {
            name: dict(n=int(getattr(ps, name).sequence_indexes.numel()),
                       spans=_spans(getattr(ps, name).sequence_indexes),
                       token_shapes=[list(map(int, t)) for t in getattr(ps, name).token_shapes[:8]])
            for name in ("action", "vision") if getattr(ps, name, None) is not None}
        rep["vision_item_split_lens"] = (list(map(int, vsl)) if vsl is not None else None)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()

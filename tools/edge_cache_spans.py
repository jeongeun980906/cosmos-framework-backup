"""cosmos3 EDGE hidden-state 추출 — cosmos_policy 의 161_cache_spans.py 와 같은 포맷.

산출 npz 는 egoverse_pair_builder 의 164/165/193 이 그대로 읽는다:
    feat  (N, n_steps, n_layers, n_spans, D)   float16
    meta  [{q, emb, cls}, ...]
    steps [0]        (denoise step 0 만 — 세 그림 모두 step0 만 쓴다)
    spans ["action", "cur", "future", "demo"]

edge 레이아웃 (tools/edge_probe_layout.py 로 실측, 3s / resolution 256):
    text     0    .. 38
    vision   38   .. 3718     아이템 2개 x (23,8,10)
             mse_loss 구간 1958..3718 = 생성할 쿼리 비디오  -> 앞쪽 38..1958 이 데모
    action   3718 .. 3896     아이템 2개 x (89,)
             mse_loss 구간 3808..3896 = 예측할 ACTION       -> 앞쪽 3718..3808 이 검색된 데모 액션
구간은 하드코딩하지 않고 **매 샘플 packed_sequence 의 mse_loss_indexes 로 나눈다**
(해상도/청크가 바뀌면 숫자가 달라지므로).

우리 span 과의 대응:
    action = 예측 ACTION 토큰      cur/future = 쿼리 비디오 (첫 프레임 / 나머지)
    demo   = 데모 비디오 토큰
hidden dim 은 우리(2048)와 다르지만 164/165/193 은 전부 z-정규화 + cosine / PCA-50 이라
차원이 달라도 나란히 놓을 수 있다. **거리 절대값끼리 비교하면 안 된다.**

usage:
  python tools/edge_cache_spans.py --tag post            # post-train iter 8000
  python tools/edge_cache_spans.py --tag base \
      --checkpoint /mnt/ddn/ckpt/cosmos3_edge_dcp/model  # 사전학습 base
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

_D = "cosmos_framework/data/generator/action/datasets"
SPANS = ["action", "cur", "future", "demo"]


def _to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, list):
        return [_to_cuda(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_cuda(v) for k, v in x.items()}
    return x


def _span_indices(ps):
    """packed_sequence -> {span: LongTensor(위치)}.  mse_loss_indexes 로 조건/생성을 가른다."""
    out = {}
    act = ps.action
    a_all = act.sequence_indexes.reshape(-1)
    a_gen = act.mse_loss_indexes.reshape(-1)
    gen_set = set(a_gen.tolist())
    out["action"] = a_gen                                     # 예측할 ACTION
    out["ret_action"] = torch.tensor([i for i in a_all.tolist() if i not in gen_set],
                                     dtype=torch.long)
    vis = ps.vision
    v_all = vis.sequence_indexes.reshape(-1)
    v_gen = vis.mse_loss_indexes.reshape(-1)
    vg = set(v_gen.tolist())
    out["demo"] = torch.tensor([i for i in v_all.tolist() if i not in vg], dtype=torch.long)
    # 쿼리 비디오는 (T,H,W) 로 프레임 단위 -> 첫 프레임 = cur, 나머지 = future
    T, H, W = vis.token_shapes[-1]
    per = int(H) * int(W)
    vgs = torch.sort(v_gen)[0]
    out["cur"] = vgs[:per]
    out["future"] = vgs[per:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="출력 이름: feat_edge_<tag>.npz")
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
    ap.add_argument("--set", default="/mnt/ddn/tmp/feature_analysis/invariance_set4.json",
                    help="우리 분석과 **같은 359 윈도**를 쓴다 (q 는 v10_test 윈도 인덱스로 공통)")
    ap.add_argument("--out-dir", default="/mnt/ddn/tmp/feature_analysis")
    ap.add_argument("--limit", type=int, default=0)
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

    rows = json.load(open(a.set))["rows"]
    if a.limit:
        rows = rows[:a.limit]
    # 데모 선택. **rows 의 r 은 그 행의 쿼리가 아니라 "그 쌍이 나온 pairs 행"이다** —
    # 한 쌍의 q 와 d 가 같은 r 을 공유한다. 그래서 _match_idx[r][0] 로 고르면 A 쌍의 q 와 d
    # 가 같은 데모를 받아(120중 119) 입력이 거의 같아지고 L0 거리가 0.0008 로 붕괴한다.
    # 반드시 **q 자신의 pairs 행**에서 찾고, 없으면(데모 전용 윈도) rows 의 d 를 쓴다.
    _MI = ds._match_idx
    _Q2R = {int(x): i for i, x in enumerate(np.asarray(ds._query_idx))}

    # ---- hidden-state 훅 ----
    # 주의: reasoner 타워는 decoder_layer.reasoner_forward(...) 를 **메서드로 직접** 부른다
    # (unified_mot.py:1517). 그 경로엔 register_forward_hook 이 안 걸린다. GEN 타워만
    # decoder_layer(...) 로 __call__ 을 탄다(:941). 어느 쪽이 실제로 도는지 모르므로
    # 후보 ModuleList 전부에 대해 __call__ 훅 + reasoner_forward 패치를 모두 건다.
    cands = [(n, m.layers) for n, m in model.named_modules()
             if isinstance(getattr(m, "layers", None), torch.nn.ModuleList) and len(m.layers) >= 8]
    assert cands, "decoder layer ModuleList 를 못 찾았다"
    print("[edge] 후보 ModuleList: " + ", ".join(f"{n or '<root>'}({len(l)})" for n, l in cands),
          flush=True)

    store = {}          # (list_name, layer_idx) -> Tensor[T,D]
    _lid_ref = {}

    _seen = {"printed": False, "t": False}

    def _describe(x, d=0):
        if torch.is_tensor(x):
            return f"T{tuple(x.shape)}"
        if isinstance(x, dict):
            return "{" + ", ".join(f"{k}:{_describe(v, d+1)}" for k, v in x.items()) + "}"
        if isinstance(x, (tuple, list)) and d < 2:
            return "[" + ", ".join(_describe(v, d + 1) for v in x) + "]"
        return type(x).__name__

    def _find_tok(x, d=0):
        """중첩된 tuple/dict 안에서 [T,D] 또는 [B,T,D] 토큰 텐서를 찾는다.
        MoT 레이어는 (hidden_dict, lbl_metadata, kv) 를 돌려주고 hidden 은 모달리티/타워별
        dict 다(smoke10: elems=['dict','dict','NoneType'])."""
        if torch.is_tensor(x):
            return x if x.dim() >= 2 else None
        if d > 3:
            return None
        if isinstance(x, dict):
            for k in ("gen", "hidden_states", "h", "x", "und", "text"):
                if k in x:
                    r = _find_tok(x[k], d + 1)
                    if r is not None:
                        return r
            for v in x.values():
                r = _find_tok(v, d + 1)
                if r is not None:
                    return r
        elif isinstance(x, (tuple, list)):
            for v in x:
                r = _find_tok(v, d + 1)
                if r is not None:
                    return r
        return None

    def _take(key, h):
        if not _seen.get("t"):
            _seen["t"] = True
            print(f"[edge] layer out = {_describe(h)}", flush=True)
        # MoT 레이어 출력은 두 타워로 쪼개져 있다(smoke12 에서 확인):
        #   causal_seq (und/text)  + _causal_indices
        #   full_only_seq (GEN)    + _full_indices   <- 우리가 원하는 gen tower
        # 각 행의 **전역 시퀀스 위치**가 인덱스 텐서에 들어 있으므로 그걸로 되돌려 조립하면
        # packed_sequence 의 전역 span 인덱스를 그대로 쓸 수 있다.
        pk = h[0] if isinstance(h, (tuple, list)) and isinstance(h[0], dict) else (
            h if isinstance(h, dict) else None)
        if pk is not None and "full_only_seq" in pk and "_full_indices" in pk:
            fs_, fi = pk["full_only_seq"], pk["_full_indices"]
            cs_, ci = pk.get("causal_seq"), pk.get("_causal_indices")
            n = int(fi.max().item()) + 1
            if torch.is_tensor(ci) and ci.numel():
                n = max(n, int(ci.max().item()) + 1)
            H = torch.zeros(n, fs_.shape[-1], dtype=fs_.dtype, device=fs_.device)
            H[fi.long()] = fs_.detach()
            if torch.is_tensor(cs_) and torch.is_tensor(ci) and ci.numel():
                H[ci.long()] = cs_.detach()
            store[key] = H
            return
        t = _find_tok(h)
        if torch.is_tensor(t):
            t = t.detach()
            store[key] = t[0] if t.dim() == 3 else t

    # ---- 레이어 forward 패치 (훅은 발화하지 않는 경우가 있어 클래스 레벨로 건다) ----
    LayerCls = type(cands[0][1][0])
    CNT = {"forward": 0, "reasoner_forward": 0, "impl_forward": 0}
    import cosmos_framework.model.generator.mot.unified_mot as _UM
    _orig_impl = _UM._impl_forward

    def _impl_counted(self, *aa, **kk):
        CNT["impl_forward"] += 1
        return _orig_impl(self, *aa, **kk)

    _UM._impl_forward = _impl_counted
    _orig_rf_cls = getattr(LayerCls, "reasoner_forward", None)
    _orig_fwd = LayerCls.forward
    # id(self) 는 못 쓴다(실행 객체가 named_modules() 결과와 다름). 레이어는 매 pass 마다
    # 0..NL-1 순서로 도므로 호출 순서 % NL 로 인덱스를 잡는다.
    NLc = len(cands[0][1])
    _pass = {"i": 0}

    def _patched_fwd(self, *aa, **kk):
        CNT["forward"] += 1
        out = _orig_fwd(self, *aa, **kk)
        _take(("__cls__", _pass["i"] % NLc), out)
        _pass["i"] += 1
        return out

    LayerCls.forward = _patched_fwd

    handles, patched = [], []
    for lname, layers in cands:
        for i, lay in enumerate(layers):
            key = (lname, i)
            handles.append(lay.register_forward_hook(
                lambda _m, _in, out, k=key: _take(k, out)))
            rf = getattr(lay, "reasoner_forward", None)
            if callable(rf):
                def wrap(orig, k=key):
                    def f(*aa, **kk):
                        out = orig(*aa, **kk)
                        _take(k, out)
                        return out
                    return f
                lay.reasoner_forward = wrap(rf)
                patched.append((lay, rf))

    packs = []
    _orig_pack = type(model)._pack_input_sequence

    def _wrapped_pack(self, *aa, **kk):
        out = _orig_pack(self, *aa, **kk)
        packs.append(out)
        return out

    type(model)._pack_input_sequence = _wrapped_pack

    feats, meta = [], []
    chosen = [None, 0]   # (ModuleList 이름, 레이어 수)
    try:
        for n, rw in enumerate(rows):
            q = int(rw["q"])
            rq = _Q2R.get(q)                       # q 자신이 쿼리인 행
            ms = ([int(x) for x in _MI[rq] if x >= 0] if rq is not None else [])
            m = ms[0] if ms else int(rw["d"])
            sample = transform({"query": reader.read_window(q), "condition": reader.read_window(m)})
            batch = {k: _to_cuda(v) for k, v in custom_collate_fn([sample]).items()}
            batch["action_processing_record"] = [
                ActionProcessingRecord(raw_action_dim=57, action_normalizer=None)]
            if isinstance(batch.get("image_size"), list):
                batch["image_size"] = batch["image_size"][0]
            packs.clear(); store.clear(); _pass["i"] = 0
            with torch.no_grad():
                # num_steps=1 -> denoise step 0 만. 세 그림 모두 step0 만 쓴다.
                model.generate_samples_from_batch(batch, guidance=1.5, seed=[1],
                                                  n_sample=1, num_steps=1, shift=3.0)
            if not packs or not store:
                if n == 0:
                    raise SystemExit(
                        f"첫 행부터 hidden state 를 못 잡았다 (pack={len(packs)} store=0).\n"
                        f"  호출 카운터: {CNT}\n"
                        f"  호출 카운터: {CNT}\n"
                        f"  후보 ModuleList={[(nm, len(l)) for nm, l in cands]}\n"
                        f"  LayerCls={LayerCls.__name__}, layer_idx 속성="
                        f"{hasattr(cands[0][1][0], 'layer_idx')}\n"
                        "  -> 레이어 forward 가 다른 이름으로 불리는지 확인 필요.")
                print(f"  !! q={q}: pack={len(packs)} store={len(store)} — 건너뜀", flush=True)
                continue
            if chosen[0] is None:               # 첫 성공 샘플에서 어느 타워를 쓸지 확정
                cnt = {}
                for (ln, li) in store:
                    cnt[ln] = cnt.get(ln, 0) + 1
                chosen[0] = max(cnt, key=cnt.get)
                chosen[1] = cnt[chosen[0]]
                print(f"[edge] 사용할 타워: '{chosen[0] or '<root>'}'  레이어 {chosen[1]}개  "
                      f"(채워진 것: {cnt})", flush=True)
            LN, NLu = chosen
            hs = {li: store[(LN, li)] for (ln, li) in store if ln == LN}
            if len(hs) < NLu:
                print(f"  !! q={q}: layers {len(hs)}/{NLu} — 건너뜀", flush=True)
                continue
            # store 에는 **마지막 pass** 의 레이어 출력만 남는다(키가 layer_idx % NL).
            # pack 은 CFG cond/uncond 로 여러 번 불리고 패스마다 시퀀스 길이가 다르므로
            # packs[0] 이 아니라 packs[-1] 과 짝지어야 한다(smoke13: 36/88 인덱스 초과).
            si = _span_indices(packs[-1])
            row = np.zeros((1, NLu, len(SPANS), hs[0].shape[-1]), np.float16)
            for li in range(NLu):
                h = hs[li].float()
                for sj, sp in enumerate(SPANS):
                    ix = si[sp]
                    n_out = int((ix >= h.shape[0]).sum())
                    if n_out and n == 0:
                        print(f"  !! span={sp}: {n_out}/{ix.numel()} 인덱스가 "
                              f"hidden 길이({h.shape[0]}) 밖 — 매핑 확인 필요", flush=True)
                    ix = ix[ix < h.shape[0]]
                    row[0, li, sj] = (h[ix].mean(0).cpu().numpy().astype(np.float16)
                                      if ix.numel() else np.nan)
            feats.append(row)
            meta.append(dict(q=q, emb=str(rw.get("emb", "")), cls=str(rw.get("cls", ""))))
            if (n + 1) % 20 == 0:
                print(f"  {n+1}/{len(rows)}", flush=True)
    finally:
        LayerCls.forward = _orig_fwd
        _UM._impl_forward = _orig_impl
        if _orig_rf_cls is not None:
            LayerCls.reasoner_forward = _orig_rf_cls
        for h in handles:
            h.remove()
        for lay, rf in patched:
            lay.reasoner_forward = rf
        type(model)._pack_input_sequence = _orig_pack

    F = np.stack(feats)                       # (N,1,NL,S,D)
    out = os.path.join(a.out_dir, f"feat_edge_{a.tag}.npz")
    np.savez_compressed(out, feat=F, meta=np.array(meta, dtype=object),
                        steps=np.array([0]), spans=np.array(SPANS))
    print(f"saved {out}  feat={F.shape} ({F.nbytes/1e9:.2f} GB)  n={len(meta)}")


if __name__ == "__main__":
    main()

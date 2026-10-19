# Action Policy / Transfer — PE & Conditioning 코드 레벨 정리

두 예제(**action policy** = LIBERO/PushT VLA, **transfer** = ControlNet 스타일 제어 비디오 생성)는
같은 백본 `OmniMoTModel` → `Cosmos3VFMNetwork`(Mixture-of-Transformers, Qwen3-VL 기반) 위에서 돈다.
차이는 **시퀀스에 무엇을 어떤 modality로 패킹하는가**와 **position id(PE)를 어떻게 매기는가** 뿐이다.
이 문서는 PE가 어디서 어떻게 들어가는지와 conditioning 구조를 코드 위치와 함께 정리한다.

---

## 0. 한눈에 보는 데이터 흐름

```
inference 빌더            sequence packing                 network forward
build_*_batch()      →   pack_*_tokens() (PE 부여)    →    _encode_vision / _encode_action
 (inference/action.py)    (data/vfm/sequence_packing.py)   (mot/cosmos3_vfm_network.py)
 (inference/transfer.py)   └ get_3d_mrope_ids_*()            └ 토큰 임베드 + modality/time embed
                                                            attention: apply_rotary_pos_emb(cos,sin)
                                                             (mot/unified_mot.py)
```

핵심: **양쪽 예제 모두 토큰을 하나의 1D 시퀀스로 패킹**한다. modality를 cross-attention으로 분리하지 않고,
`[text, vision, (action)]` 토큰을 한 줄에 이어붙인 뒤 **3D mRoPE position id**로 시·공간 관계를 표현한다.
"conditioning"은 별도 인코더가 아니라 **(a) 어떤 토큰을 clean(조건)으로 두고 어떤 토큰에 noise를 줄지**와
**(b) position id를 어떻게 정렬할지**로 구현된다.

---

## 1. PE 두 가지 regime

`Cosmos3VFMNetwork.__init__` (`cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:160-209`)에서
`config.position_embedding_type`으로 분기한다.

| regime | 동작 | 코드 |
|---|---|---|
| `3d_rope` (레거시 기본값) | **additive** `VideoRopePosition3DEmb`를 토큰 임베딩에 더함 + attention은 1D RoPE | `:162`, `:192` |
| `flattened_sin_cos` | additive sin/cos | `:176` |
| `unified_3d_mrope` | **additive 임베딩 없음**(`latent_pos_embed=None`), position 정보는 `(3,N)` 3D position id로 attention의 mRoPE에 직접 투입 | `:179-181`, `:205-207` |

> 출하되는 nano/super 체크포인트(=action policy, transfer가 쓰는 것)는 모두 **`unified_3d_mrope`**.
> - `configs/base/experiment/sft/models/nano_model_config.py:43` → `position_embedding_type="unified_3d_mrope"`, `enable_fps_modulation=True`, `unified_3d_mrope_reset_spatial_ids=True`, `unified_3d_mrope_temporal_modality_margin=15000`
> - `super_model_config.py:65` 동일
> - 레거시 기본값 `model_config.py:32`은 `"3d_rope"`이지만 두 recipe는 nano/super config를 deepcopy 해서 override.

즉 **PE = "3D multi-axis RoPE position id를 토큰별로 부여 → attention에서 Q/K 회전"**. 아래 2~4가 그 디테일.

---

## 2. 3D mRoPE position id 생성 (PE의 핵심)

파일: `cosmos_framework/model/vfm/mot/unified_3dmrope_utils.py`

세 축 `(temporal, height, width)`로 position을 매긴다. Qwen3-VL mRoPE 설계.

### 2.1 Text 토큰 — `get_3d_mrope_ids_text_tokens` (`:41`)
- 세 축이 **동일한 단조 증가 값**: `(o,o,o),(o+1,o+1,o+1),...` (`:65-70`).
- 다음 offset = `temporal_offset + num_tokens`.

### 2.2 Vision/Action 토큰 — `get_3d_mrope_ids_vae_tokens` (`:75`)
- `(T,H,W)` 그리드를 T-major로 flatten → `(3, T*H*W)`.
- **temporal 축**: `t_index = arange(T) + temporal_offset + start_frame_offset` (`:154-160`).
- **spatial 축**: `h_index`, `w_index`는 그리드 좌표 (`:164-171`).
- `reset_spatial_indices=True`(기본): 각 vision 세그먼트마다 H,W가 0부터 시작 → "이미지 내 절대 공간 위치" (`:13-18`).
- 세그먼트 종료 후 `next_temporal_offset = ceil(max_position)+1` (`:189-192`) → 다음 모달리티가 겹치지 않게 이어짐.

### 2.3 FPS 변조 (`enable_fps_modulation=True`일 때, `:135-149`)
프레임 인덱스가 아니라 **실시간**을 반영하도록 temporal 축을 스케일:
```
tps      = fps / temporal_compression_factor
base_tps = base_fps / base_temporal_compression_factor
scaled_t = (frame_index + start_frame_offset)/tps * base_tps + temporal_offset
```
→ FPS가 다른 영상도 temporal position이 비교 가능해진다. temporal 축은 float, spatial은 long.

---

## 3. 시퀀스 패킹 — PE가 실제로 부여되는 지점

파일: `cosmos_framework/data/vfm/sequence_packing.py`. `SequencePlan`(`:496`)이 "이 샘플에 어떤 modality가 있고
무엇을 조건으로 둘지"를 기술하며, 패커가 모달리티별로 `get_3d_mrope_ids_*`를 호출해 `packed_seq.position_ids`에 쌓는다.

### `SequencePlan` 주요 필드 (`:524-545`)
| 필드 | 의미 |
|---|---|
| `has_text / has_vision / has_action / has_sound` | 시퀀스에 포함할 modality |
| `condition_frame_indexes_vision` | **clean(조건) latent 프레임 인덱스**. `[]`=전부 noise(생성·supervise), 전부 지정=전부 clean(조건) |
| `condition_frame_indexes_action` | 동일하게 action step에 적용 |
| `share_vision_temporal_positions` | **transfer 전용**: 여러 vision item이 동일 temporal 그리드 공유 (아래 5) |
| `action_start_frame_offset` (기본 1) | action[0]을 vision frame 1에 정렬 (아래 4) |

패킹 순서: text(`_pack_text_tokens` → `get_3d_mrope_ids_text_tokens`) → vision(`:1550 _pack_vision_tokens`) →
action(`:1583 _pack_action_tokens`). 각 호출이 `packed_seq._mrope_temporal_offset`을 갱신하며 이어붙인다.

---

## 4. Action Policy 예제

### 4.1 모델 구성
- recipe: `configs/base/experiment/action/posttrain_config/action_policy_libero_nano.py` (pusht/droid도 동형)
  - `model.config = deepcopy(NANO_MODEL_CONFIG)` → `action_gen=True`, `max_action_dim=64`, `unified_3d_mrope`.
  - 학습 head: `action2llm`, `llm2action`, `action_modality_embed` (+ generation head) — `optimizer.keys_to_select:91-99`, lr 5배 (`:101-105`).

### 4.2 입력 배치 만들기 — `inference/action.py`
- `build_action_batch` (`:74`)가 `video`, `action`, `domain_id`, `sequence_plan` 등을 채움.
- `build_sequence_plan_from_mode(mode, ...)` (`:109`)가 mode별 `SequencePlan` 생성:
  - **`policy`**: 관측 비디오는 clean 조건, action은 noise(생성 대상). 입력 action은 zeros (`_load_actions:43-45`).
  - **`forward_dynamics`**: action을 조건으로 다음 frame 생성.
  - **`inverse_dynamics`**: 두 frame 사이 action 추론.
- **proprio state 조건화 (use_state)**: LIBERO/PushT/DROID는 현재 절대 proprio state[:7]를 `action[0]`(조건 step)으로
  prepend → action 텐서가 `(chunk+1, 7)`이 됨 (`action_policy_libero_nano.py:222-226`). eval 서버는 매칭되는 state를 보내야 함.
- **language 조건화**: instruction이 `ai_caption`(JSON, `ActionPromptJsonFormatter`)으로 text 토큰에 들어감 — 별도 T5/cross-attn 없이 reasoner 텍스트 토큰으로 in-context.
- **이미지 조건화**: 관측 frame은 VAE로 latent화되어 clean vision 토큰으로 시퀀스에 들어감(2-camera는 위/아래 concat).

### 4.3 Action PE (왜 `start_frame_offset=1`인가)
`_pack_action_tokens` (`sequence_packing.py:882-899`):
```python
get_3d_mrope_ids_vae_tokens(
    grid_t=action_split_len, grid_h=1, grid_w=1,      # action은 공간 차원 없음 → 1x1 그리드
    temporal_offset=action_temporal_offset,            # vision과 같은 temporal 공간 공유
    temporal_compression_factor=1,                     # action은 frame rate 그대로(압축 없음)
    start_frame_offset=action_start_frame_offset)      # =1: action[0]을 vision frame 1에 정렬
```
- **action과 vision은 temporal 공간을 공유**한다 (`action_temporal_offset = vision_start_temporal_offset`, `:1589`).
  패킹 후 `_mrope_temporal_offset`을 action으로 **갱신하지 않음** (`:900-901`) — action은 vision과 "병렬"로 흐르기 때문.
- `temporal_compression_factor=1`: vision은 4프레임이 latent 1개로 압축되지만 action은 step=frame이라 비압축. base_tps는 vision 압축 기준으로 잡아 두 모달리티 시간축을 정렬.

### 4.4 Action conditioning이 네트워크에서 들어가는 법 — `_encode_action` (`cosmos3_vfm_network.py:733-792`)
1. `pack_action` → **`action2llm`(`DomainAwareLinear`)** 로 `action_dim(64) → hidden_size` 투영. 임베딩 weight가 **embodiment domain별**로 분기(`per_token_domain_id`)되어 robot마다 다른 action 의미를 흡수 (`:752-755`).
2. `unified_3d_mrope`이므로 additive `action_pos_embed`는 `None`(skip) — PE는 position id로만 (`:758`).
3. **`action_modality_embed`**(학습 파라미터) 를 더해 modality 식별 (`:768-770`).
4. noise step에만 **timestep embedding**(`time_embedder`)을 더함 — flow-matching의 σ (`:773-788`).
5. 디코드: `llm2action`(역시 domain-aware)로 hidden→action velocity 예측 (`_decode_action:794`).

> 정리: action은 **"domain-aware 선형 투영 + modality 임베딩 + (조건 step 제외) timestep 임베딩"**으로 토큰화되고,
> vision frame과 **시간축이 정렬된** mRoPE id를 받아 한 시퀀스 안에서 self-attention으로 vision/text와 상호작용한다.

---

## 5. Transfer 예제 (ControlNet 스타일)

### 5.1 시퀀스 레이아웃
`inference/transfer.py`의 `build_transfer_batch` (`:142`):
```
video = [ctrl_1, ..., ctrl_N, target]      # 모두 vision item (별도 인코딩)
sequence layout = [text, ctrl_1_tokens, ..., ctrl_N_tokens, noisy_target_tokens]
```
- `num_vision_items_per_sample = N+1` (`:160,174`).
- 시스템 프롬프트 `_SYSTEM_PROMPT_TRANSFER` + 프롬프트가 text 토큰으로.
- edge/blur 등 control은 `load_transfer_control_frames`로 디스크에서 읽거나 즉석 계산(`apply_transfer_control_augmentor:55`).

### 5.2 Control conditioning = "앞 item들은 전부 clean, 마지막 item만 noise"
패커 `sequence_packing.py:1521-1527`:
```python
if num_vis > 1 and item_idx < num_vis - 1:
    item_condition_frames = list(range(latent_t))   # control item → 전 프레임 clean(조건)
else:
    item_condition_frames = sequence_plan.condition_frame_indexes_vision  # target → 보통 [] = 전부 생성
```
→ ControlNet의 "control은 고정 조건, target만 denoise" 구조를, 별도 control-branch 없이 **같은 시퀀스의 clean vs noisy 토큰**으로 구현.

### 5.3 Transfer PE — `share_vision_temporal_positions` (핵심)
`build_transfer_batch`가 `SequencePlan(share_vision_temporal_positions=share_temporal)` 설정 (`transfer.py:179-185`, 기본 True).
패커 `sequence_packing.py:1494-1548`:
```python
shared_grid = sequence_plan.share_vision_temporal_positions and num_vis > 1
items_temporal_offset_snapshot = packed_seq._mrope_temporal_offset
...
if shared_grid:
    # latent_t, H, W 가 모든 item에서 동일해야 함(assert)
    packed_seq._mrope_temporal_offset = items_temporal_offset_snapshot  # 매 item 직전 offset 되감기
```
- **효과**: 모든 control item과 target이 **동일한 temporal mRoPE 그리드**를 받음 → control frame `i`가 target frame `i`와 시·공간적으로 정렬. 이게 transfer 품질의 핵심.
- shared-grid 모드여도 각 `_pack_vision_tokens`는 내부적으로 offset을 `latent_t`만큼 전진시키지만, loop 종료 후 offset은 `snapshot + latent_t`(=단일 클립과 동일)라서 이후 토큰(EOV/다음 모달리티) position이 자연스럽게 이어짐 (`:1487-1493`).
- ⚠️ 이 플래그는 **학습 체크포인트의 regime과 일치해야** 한다. 불일치 시 control↔target frame drift 발생 (`transfer.py:176-178` 주석, `docs/transfer_temporal_id_fix.md` 참조).

### 5.4 Autoregressive chunk 조건화 (시간 연장)
`generate_transfer_sample` (`transfer.py:196`)는 긴 영상을 chunk로 잘라 생성:
- `num_conditional_frames`만큼 **이전 chunk의 마지막 출력 frame을 다음 chunk의 clean 조건**으로 `target_norm`에 넣음 (`:314-325`).
- 첫 chunk는 `num_first_chunk_conditional_frames` + `vision_path` 입력 frame으로 조건화 (`:300-313`).
- 조건 frame 수 → latent 인덱스 변환: `condition_frame_indexes = range((cond-1)//tcf + 1)` (`:161-164`) → `SequencePlan.condition_frame_indexes_vision`로 전달.

### 5.5 Transfer가 네트워크에서 들어가는 법 — `_encode_vision` (`cosmos3_vfm_network.py:603-670`)
- control/target 모두 동일 경로: `patchify_and_pack_latents` → **`vae2llm`**(`:636`) → (unified_3d_mrope이므로 additive pos emb skip `:640`) → **noise frame에만 timestep embed**(`:648-665`).
- control 토큰은 clean(noise step 없음)이라 timestep이 안 붙고, target의 noisy frame에만 σ가 주입 → 같은 attention 안에서 target이 control을 참조하며 denoise.

---

## 6. PE가 attention에서 실제로 적용되는 곳

`cosmos_framework/model/vfm/mot/unified_mot.py`:
- `position_ids`(`(3,N)` mRoPE)로부터 `rotary_emb`가 `cos,sin` 생성 (`:771-775`). mRoPE 3축은 내부에서 head_dim에 interleave되어 collapse.
- attention 내부에서 **understanding tower / generation(MoE) tower 각각** Q,K에 회전 적용 (`:556-569`):
  ```python
  q_und_, k_und_ = self._apply_rotary_pos_emb(q_und, k_und, cos_und, sin_und, unsqueeze_dim=1)
  q_gen_, k_gen_ = self._apply_rotary_pos_emb(q_gen, k_gen, cos_gen, sin_gen, unsqueeze_dim=1)
  ```
- `_apply_rotary_pos_emb`는 layer type에 따라 Qwen3-VL용/partial mRoPE용으로 선택 (`:118-129`).
- 즉 **PE는 weight가 아니라 Q/K 회전**으로 들어가며, modality마다 다른 position id를 줬을 뿐 적용 메커니즘은 동일.

---

## 6.5 conditioning의 "noise 차등" 메커니즘 (clean vs noisy)

별도 cross-attention 모듈은 없다. modality는 **하나의 joint self-attention**에서 같이 attend한다.
"무엇이 조건인가"는 **flow-matching의 σ(noise level)를 토큰별로 차등 적용**해서 구현:
- `condition_frame_indexes_*`로 지정된 clean 토큰 → **σ=0** (noise 안 줌), MSE loss에서도 제외.
- 나머지(생성 대상) 토큰 → σ=σ_t, 여기에만 `time_embedder`로 timestep embed 주입(`_encode_*`의 `_apply_timestep_embeds_to_noisy_tokens`), MSE supervise.
- `omni_mot_model.py`의 noise/loss 단계에서 `condition_mask`(1=clean, 0=generate)로 `sigma_eff = sigma * (1 - condition_mask)` 적용.

→ action policy: 관측 vision + state(action[0])는 clean, 미래 action은 noisy.
→ transfer: control item 전체는 clean, target만 noisy.

## 7. 요약

| 항목 | Action Policy | Transfer |
|---|---|---|
| 시퀀스 | `[text, clean-vision(관측), noisy-action]` | `[text, clean ctrl_1..N, noisy target]` |
| conditioning 방식 | clean vision + text(instruction) + proprio(state→action[0]) | clean control items + text + AR 이전출력 frame |
| 생성 대상(noise) | action token (policy mode) | target vision token |
| PE 특이점 | action: `1x1` grid, `tcf=1`, `start_frame_offset=1`로 vision과 시간축 정렬 | `share_vision_temporal_positions=True`로 control↔target frame 정렬 |
| modality 투영 | `action2llm`(domain-aware) + `action_modality_embed` | `vae2llm`(공통) |
| PE regime | `unified_3d_mrope` (additive 없음, position id로만) | `unified_3d_mrope` |

공통 원리: **별도 conditioning encoder/cross-attention 없이, 하나의 토큰 시퀀스 안에서 (1) clean vs noisy 토큰 구분으로 "무엇이 조건인가"를, (2) 3D mRoPE position id 정렬로 "무엇이 시간·공간적으로 대응하는가"를 표현**한다.

### 참고 코드 위치
- PE 생성: `model/vfm/mot/unified_3dmrope_utils.py`
- 시퀀스/조건 패킹: `data/vfm/sequence_packing.py` (`SequencePlan:496`, vision `:1474-1563`, action `:1568-1598`, shared-grid `:1487-1548`)
- 네트워크 임베드: `model/vfm/mot/cosmos3_vfm_network.py` (`__init__:160-219`, `_encode_vision:603`, `_encode_action:733`)
- PE attention 적용: `model/vfm/mot/unified_mot.py:553-569`, `:748-775`
- inference 배치: `inference/action.py`, `inference/transfer.py`
- config: `configs/.../sft/models/nano_model_config.py`, `.../action/posttrain_config/action_policy_*_nano.py`

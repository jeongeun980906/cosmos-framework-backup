# Mid 모델 Feature/Output 분석 계획 (2026-08-04)

대상 모델 (mid 8K, DCP→pt):
- **BC** = `egoverse_ret_v3_448_state_blankret_bv2_v10` (blank 레이아웃, state_t 29)
- **Cotrain** = `egoverse_ret_v3_448_state_blankret_bv2_cotrain` (state_t 29)
- **Ours** = `egoverse_ret_v3_448_state` (transfer/retrieval 레이아웃, state_t 50)

데이터: **v10 test** = `egoverse_pairs_v9_test` (held-out 에피소드), **EgoDex** =
`egodex_pairs_top3.npz` (held-out embodiment). 샘플 ~500 윈도/데이터셋.

## Feature 정의 (모든 분석 공통)

- 추출 경로: `eval_egoverse_ret.py`의 로딩 경로 재사용, DiT 블록에 forward hook.
- **클린 latent + 고정 소음수준 σ_small 1-step forward** (생성 없이 1회 통과) —
  결정적·저렴, 표현 비교의 표준 세팅.
- 레이어: 중간 블록 (예: 28블록 중 14) + 민감도로 {1/4, 1/2, 3/4} 지점 비교.
- 풀링: **쿼리 윈도에 해당하는 토큰 span의 mean-pool** (레이아웃이 모델마다 달라도
  쿼리 future/action 토큰은 공통 존재 — 모델별 span 매핑 테이블 작성).

## 분석

**D. Probing** — frozen feature → 선형 회귀(ridge)로 실현량 복원:
**접촉 속도·회전 총량·감속 구간 길이** (motivation과 동일 지표, 107 파이프라인 재사용해
윈도별 GT 산출). 평가 = held-out R², 모델 3종 × 데이터 2종.
판정: Ours/Cotrain ≫ BC이면 "데이터에 있던 실현 정보가 표현에 남았다".

**E. 표현 거리 구조** — 쌍 구성:
- A쌍: **motion 같음(retrieval top-K 이웃) + 실현 다름** (dynamics 지표 차이 상위)
- B쌍: **motion 같음 + 실현도 같음** (dynamics 차이 하위)
판정: d(A) > d(B) 유의 (Mann-Whitney)이면 표현이 실현을 보존; BC는 구분 못할 것 예측.
(실현 지표는 D와 동일 3종의 합성 z-스코어.)

**F. Attention 시각화 (3모델 전부)** — self-attention에서 쿼리 토큰 → demo-슬롯 토큰
attention mass: Ours는 demo의 어느 시점/슬롯을 보는지 히트맵, BC/Cotrain은 blank 슬롯
대조군 (무시/균등이어야 정상) — "demo를 읽는다"의 직접 증거 + 대조를 한 그림에.
블록별 (초/중/후반) × 대표 샘플.

**A. 출력 반응성 (Ours만)** — 같은 관측 o, 다른 demo m (top-K에서 실현이 다른 m들):
- 민감도 = Δoutput(생성 action) / Δm(디스크립터 거리) — BC류는 ~0, Ours는 유의해야.
- **비지정 축 반응**: m이 경로만 다를 때 출력의 접촉 속도까지 변하면 복사가 아닌
  grounding — Δm(경로축) 대비 Δoutput(접촉속도축) 교차 민감도 행렬.

**B. 장면 반응성 (Ours만)** — m 고정, o 변경: 같은 의도라도 다른 물체·배치면 실현이
달라져야. 쌍 구성 = retrieval 근사 (같은 demo m을 공유하는 서로 다른 쿼리 o들 — pairs
역인덱스로 "m이 같고 o가 다른" 쌍 수집; counterfactual 물체 교체는 시뮬레이터가 없어 근사로).
판정: Δoutput/Δo 유의 — **A(조건 반응)와 B(장면 반응)가 함께 성립해야 "복사"도 "무시"도
아닌 grounding**. A만 성립=복사 의심, B만 성립=조건 무시 의심.

## 실행 순서
1. 스모크: Ours ckpt 로드 → 1배치 forward hook → hidden 추출 확인 (今)
2. 3모델 × 2데이터 feature 대량 추출 (~500윈도, 캐시)
3. D → E → F → A 순 (D/E는 같은 캐시 사용)

산출물: `/mnt/ddn/tmp/feature_analysis/` (feature 캐시, R² 표, 거리 분포, attention 히트맵,
민감도 행렬), 스크립트 `egoverse_pair_builder/110_*.py` 계열.


## 학습-세팅 감사 결과 (2026-08-04, 원문 코드 확인 — 추출 설정 확정)

`egoverse_ret_experiment_configs.py` 정독 결과 (재발 방지용 명문화):

| 모델 | 학습 manifest | state 정규화 | action stats | t5 |
|---|---|---|---|---|
| Ours (`_ret_v3_448_state`) | `egoverse_pairs_v10` (base_kw 디폴트 = _DATASET_KW_V10) | **ROBUST** `v10/state57_stats_staterobust.json` (mean±4σ) | `v10/action57_stats.json` | v10 |
| BC (`blankret_bv2_v10`) | `egoverse_pairs_v10` | 동일 (v10 staterobust) | v10 | v10 |
| Cotrain (`blankret_bv2_cotrain`) | `egoverse_pairs_v10_cotrain` | **v10_cotrain**/staterobust (he·eva 포함 재계산본) | v10_cotrain (+_global fallback) | v10_cotrain |

**사고 원인**: 추출 시 `manifest_dir`만 v10_test로 바꿔 → robust 통계가 **test manifest에서
재계산**되어 학습 정규화와 다른 값 사용. (참고: `_clean`/pctl 통계는 별도 실험용 — 이 3모델은
전부 staterobust가 맞음. "clean으로 학습 안 했다"는 유저 지적과 정합.)

**확정 추출 오버라이드** (모델별):
```
dataloader_train.dataset.manifest_dir=/mnt/ddn/dataset/egoverse_pairs_v10_test
dataloader_train.dataset.robust_stats_file=<학습 manifest>/state57_stats_staterobust.json
dataloader_train.dataset.action_stats_path=<학습 manifest>/action57_stats.json
dataloader_train.dataset.t5_text_embeddings_path=<학습 manifest>/t5_embeddings.pkl
dataloader_train.dataset.use_image_aug=False, use_stronger_image_aug=False
  (학습은 aug ON이지만 결정적 표현 추출을 위해 OFF — 문서화된 의도적 선택)
```

## 소규모 5분석 검증 절차 (heavy 런 전 의무, 산출=영상 우선)

샘플 8~16개로 D/E/F/A/B 전부 end-to-end 1회씩:
- **A/B (영상)**: 같은 o × 다른 m 3개 / 같은 m × 다른 o 3개 → 생성 latent를
  `model.decode`로 비디오 복원 → [조건 demo | 생성 결과] 나란히 mp4.
- **F (영상)**: 쿼리 토큰→demo 토큰 attention을 쿼리 프레임 진행에 따라 애니메이션
  (3모델: Ours=구조적, BC/Cotrain=blank 대조).
- **D/E (이미지 — 영상 부적합)**: D=pred-vs-GT 산점도(소표본), E=쌍 거리 분포.
- 전 스크립트 **샘플 단위 로그+ETA** ([[realtime-logging-rule]]).
유저 검수 승인 후에만 전체 (400/모델) 실행.

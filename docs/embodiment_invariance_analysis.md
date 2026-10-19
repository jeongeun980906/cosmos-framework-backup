# Pair 학습의 embodiment 불변성 분석 (2026-08-10)

**질문.** 기존 pretraining / latent-action 계열은 한 embodiment·환경 안에서 next-action 만
모델링하므로 같은 모션이어도 사람/환경에 따라 표현 공간의 다른 곳에 매핑된다. retrieval pair
학습을 한 Ours 는 비슷한 모션이 사람/환경과 무관하게 비슷한 곳에 매핑되는가?

**결론.** **초기 인코딩 단계(L0–L2)에서만 성립한다.** Ours 는 그 구간에서 embodiment 성분을
특이적으로 억제하지만(emb/task 대비 0.67–0.99 vs 베이스라인 1.6–2.0), L3 이후에는 action
토큰이 현재 관측 슬롯의 외형 정보를 흡수하면서(RSA 0.40→0.75) 세 모델 모두 정체성이 지배하게
되어 차이가 사라진다. "embodiment-invariant" 보다 **"초기 단계에서 embodiment gap 을 완화"**
가 데이터가 뒷받침하는 서술이다.

대상 모델은 mid 8K 체크포인트 3종 — **Ours** (`egoverse_ret_v3_448_state`), **BC**
(`..._blankret_bv2_v10`), **Co-train** (`..._blankret_bv2_cotrain`). 자세한 체크포인트 경로는
`egoverse_pair_builder/147_invar_cache.py` 의 `MODELS` 참조.

---

## 1. 평가셋

### 1.1 v1 (`146_invariance_set.py` → `invariance_set.json`) — **교차 설계용, 현재는 참고용**

기존 `visgrp` 셋은 task 와 embodiment 가 완전 교락(scale={unpack_grocery,
put_object_in_container}, MECKA={fill_scoop, sort_bins})이라 이 질문에 쓸 수 없었다.
`egoverse_pairs_v10_test` 의 18개 cls 중 **2개 이상 embodiment 에 걸친 5개**를 골라
(cls × emb) 10셀 × 20윈도 = **200윈도 / 75 에피소드** 로 구성.

| cls | 분포 |
|---|---|
| put_object_in_container | MECKA 557 / scale 1434 |
| sort_bins | MECKA 1174 / scale 717 |
| cup_on_saucer | aria 447 / scale 922 |
| do_dishes | MECKA 905 / scale 169 |
| bagging_groceries | aria 237 / scale 295 |

**한계 (v2 로 넘어간 이유)**: 성기게 뽑아 retrieval 파트너가 셋 안에 거의 없다
(top-20 매치 2361개 중 내부 8개, cross-emb 0개). "같은 모션" 을 손별 8개 스칼라로 매칭해야
했는데, 그 라벨이 모션을 거의 못 가른다 (같은 task 모션거리 4.83 vs 다른 task 5.27, 9% 차이;
`bagging_groceries` 내부 5.20 > `cup_on_saucer`×`put_object_in_container` 교차 4.91).

### 1.2 v2 (`151_pairset.py` → `invariance_set2.json`) — **본 분석에 사용**

모션 유사 라벨을 **공유 데모(shared demo)** 로 정의: 두 query 가 `pairs.npz` 의 top-20 에
같은 데모를 가지면 retrieval 디스크립터 기준으로 서로 모션이 비슷하다. 둘 다 정상 query 라
각자 자기 데모를 쓰므로 **서로를 입력으로 받지 않는다 → 누설 없음**.
(파트너 윈도를 직접 넣으면 Ours 는 그 파트너를 데모로 입력받은 상태라 순환이 되고, 게다가
파트너는 학습 풀 소속이라 자기 데모가 없다 — `pairs.npz` note: "demos from full train pool".)

- **MS-X** (모션유사 × 다른 사람) 48쌍 / **MS-S** (모션유사 × 같은 사람) 113쌍
- 모든 쌍에 **같은 task + 다른 에피소드** 조건 → task·장면 통제
- **301 윈도 / 122 에피소드**, emb = scale 154 / MECKA 124 / aria 23
- 분석 시 MS-S 는 MS-X 가 존재하는 5개 cls 로 제한해 분포를 맞춘다 (MSS 294, MDS 1366)

---

## 2. Feature 추출 (`147_invar_cache.py`)

DiT 28개 블록에 forward hook → 블록 출력의 **슬롯별 공간(28×28) 평균**. 윈도당
`(스텝, 28블록, T슬롯, 2048)` fp16.

- **판정 지점 = step0** — `t=1000` 단일 denoise. 타깃 슬롯이 순수 노이즈라 **정답 누설 0**.
  (`--step0-only`. 전체 생성 35스텝 모드도 있으나 이번 분석엔 미사용 — §5 참조)
- `T` = ours 50 / bc·cotrain 29. **BC/Co-train 은 데모 비디오가 blank 2슬롯으로 접혀
  레이아웃이 다르므로, 반드시 모델별 layout 으로 슬롯을 잡아야 한다.**

| | T | action | future | target span 크기 |
|---|---|---|---|---|
| ours | 50 | 26 | 27–49 | 24 |
| bc / cotrain | 29 | 5 | 6–28 | 24 |

- 정규화: 각 모델의 **학습 manifest staterobust** (ours/bc = v10, cotrain = v10_cotrain).
  test manifest 에서 재계산 금지 (`dataset.state_stats` 를 instantiate 후 override).
- 비용: step0 기준 ours 5.2s/win, bc·cotrain 3.0s/win. 301윈도 × 3모델 ≈ 50분 (2 GPU 병렬).

---

## 3. 지표

윈도당 span 슬롯 평균 → 2048차원 → 윈도축 z-정규화 → L2 정규화 → **cosine 거리**.

### 3.1 2×2 분해 (`152_pair_decomp.py`)

|  | 모션 유사 (공유데모) | 모션 상이 (비공유) |
|---|---|---|
| 같은 사람 | `MSS` | `MDS` |
| 다른 사람 | `MSX` | `MDX` |

- 모션 효과 `M = ((MDS-MSS) + (MDX-MSX)) / 2`
- 정체성 효과 `I = ((MSX-MSS) + (MDX-MDS)) / 2`
- `R = M / I` (>1 이면 모션 지배), `X/S = MSX / MSS` (1이면 불변)
- 쌍 단위 부트스트랩으로 CI

### 3.2 메커니즘 판별 (`153_mechanism.py`)

`I` 가 작은 것이 **embodiment 특이 억제**인지 **전반적 비뭉침**인지 가른다.

- `d_all` : 전체 쌍 거리 중앙값 (모델 스케일)
- `I_emb` : 같은 task 안에서 (다른 사람 − 같은 사람)
- `I_task` : 같은 사람 안에서 (다른 task − 같은 task)
- **`I_emb / I_task`** : 스케일 무관 판정 지표. 전부 균일하게 줄었으면 이 값이 유지되고,
  embodiment 만 억제됐으면 떨어진다.
- participation ratio : 고유값 분포 평탄도 (등방성)

### 3.3 정보 흐름 (`154_flow.py`)

- 슬롯별 `I_emb/I_task` : cur / action / future / demo
- **RSA** : action 슬롯의 쌍거리 행렬 vs cur 슬롯의 쌍거리 행렬 (Spearman)

---

## 4. 결과

### 4.1 초기 레이어 — Ours 가 embodiment 를 특이적으로 억제

| layer | metric | BC | Co-train | **Ours** |
|---|---|---|---|---|
| L0 | emb/task | 1.920 | 1.605 | **0.816** |
| L0 | I_emb / d_all | 0.300 | 0.277 | **0.097** |
| L1 | emb/task | 1.966 | 1.729 | **0.672** |
| L1 | I_emb / d_all | 0.596 | 0.580 | **0.124** |
| L2 | emb/task | 2.003 | 1.926 | **1.491** |
| L2 | I_emb / d_all | 0.532 | 0.525 | **0.241** |

- 베이스라인은 **정체성이 task 보다 1.6–2.0배 강하게** 표현을 가른다. Ours 는 L0/L1 에서
  **1 아래로 역전** — task 가 정체성보다 강하다.
- **스케일 효과가 아니다**: `d_all` 이 세 모델 1.00–1.07 로 사실상 동일 (SUMMARY 패널 D).
  등방성(PR)은 Ours 가 2배 높지만, 그것이 원인이라면 task 대비도 같이 줄어야 하는데 아니다.
- `X/S` 비율도 **20/28 레이어에서 Ours 최소** (L1: 1.079 vs 2.284 / 2.706), 평균 격차 0.098.

### 4.2 L3 이후 사라지는 이유 — action 슬롯이 현재 관측의 외형을 흡수

| | cur 슬롯 emb/task | action 슬롯 emb/task | RSA(action, cur) |
|---|---|---|---|
| | L0–2 → L3–27 | L0–2 → L3–27 | L0–2 → L3–27 |
| BC | 3.26 → 2.91 | 1.96 → 2.72 | 0.45 → 0.65 |
| Co-train | 3.19 → 2.61 | 1.75 → 2.36 | 0.47 → 0.65 |
| **Ours** | **1.97** → 2.59 | **0.99** → 2.86 | **0.40** → 0.58 |

- **현재 이미지 슬롯은 처음부터 정체성 편향** (emb/task 2–3.3). 같은 사람이 찍은 영상은
  배경·조명·손 외형이 같으므로 구조적이다. 이것이 정체성 오염원.
- action 슬롯의 RSA 가 L0 의 0.37–0.40 에서 L11–L19 의 0.75–0.83 으로 오른다 →
  action 토큰이 cur 슬롯의 거리 구조를 물려받는다. Ours 도 0.99 → 2.86 으로 가장 크게 상승.
- Ours 의 **demo 슬롯 RSA 는 0.25–0.40 으로 낮게 유지**된다 (데모는 다른 사람 영상이라
  정체성 오염이 적음). action 이 demo 보다 cur 을 훨씬 강하게 따라간다.

### 4.3 성립하지 않은 것 (기록)

`R = M/I` 가 전 조건에서 **0 근처 또는 음수**다. 네 칸을 보면 `MSS≈MDS`, `MSX≈MDX` —
**공유데모 라벨조차 표현 거리를 거의 못 가른다** (정체성은 0.1–0.8 벌리는데 모션은 0.00–0.09).
즉 "모션이 표현을 지배한다" 는 어느 모델에서도 성립하지 않는다.

시도했으나 우위가 나오지 않은 조합 (모두 `paper_figs/invar/` 에 산출물 있음):

| 검정 | 결과 |
|---|---|
| kNN purity / PCA-50 분류 / Fisher ratio (embodiment) | L3 이후 세 모델 0.93–1.00 포화, 무차별 |
| motion probing (ridge CV Spearman) | 세 모델 동일. 단 `span=action` 후반 레이어는 Ours 우위 (L23 .531 vs .505/.506) |
| P5 = 모션매칭 파트너의 표현 순위 | 전 조건 chance (0.43–0.52) |
| decorr center / proj (embodiment 성분 제거) | 세 모델 모두 개선 → 정체성은 **가산 오프셋**. 제거 후 Ours 가 가장 덜 정렬 |
| 삼중항 (다른사람·같은task vs 같은사람·다른task) | 세 모델 0.0–0.3. 0.5 넘는 유일 지점(action L13)은 **베이스라인 우위** |

**삼중항이 무의미했던 이유**: positive/negative 를 language task 라벨로 정의했는데 그 라벨이
모션을 못 가른다(§1.1). 측정된 것은 "외형이 task 라벨을 이기는가" 에 가깝다.

---

## 5. 한계 / 다음

- **step0 한정.** `t=1000` 이라 타깃 슬롯이 순수 노이즈고 hidden 이 사실상 조건(현재 이미지·
  데모·텍스트)을 반영한다. §4.2 의 cur 지배도 이것과 무관하지 않다. 후반 denoise 스텝에서는
  타깃에 생성된 모션이 들어차므로 cur 의존이 줄 가능성이 있다. `147 --num-steps 10` 으로
  301윈도 × 3모델 ≈ 1.5시간이면 확인 가능 (스텝 축 캐시 → `152 --step K`).
- **aria 23윈도** 로 적다. cross-emb 쌍(MS-X) 48개도 넉넉하지 않다.
- mid 8K 체크포인트 3종에 한정. 더 학습된 체크포인트나 GR1 post-train 모델은 미확인.
- `152` 의 `R` 은 공유데모 라벨의 약한 판별력 때문에 해석력이 낮다. 모션 축을 쓰려면
  action chunk(92×57) 전체 거리 등 더 세밀한 정의가 필요하다.

---

## 6. 파일 맵

### 스크립트 (`/mnt/ddn/dataset/egoverse_pair_builder/`)

| 파일 | 역할 |
|---|---|
| `146_invariance_set.py` | v1 평가셋 (cls × emb 교차, 200윈도) |
| `151_pairset.py` | **v2 평가셋** (공유데모 쌍 기반, 301윈도) |
| `147_invar_cache.py` | feature 추출 (`--step0-only` / 전체 생성, `--set`, `--suffix`) |
| `152_pair_decomp.py` | 2×2 분해, `R` / `X/S` |
| `153_mechanism.py` | 스케일·등방성 판별, `I_emb/I_task` |
| `154_flow.py` | 슬롯별 대비 + RSA (L3 이후 원인) |
| `155_summary_fig.py` | 요약 figure (json 만 읽어 렌더) |
| `148_invar_analysis.py` | v1 셋 지표 묶음 (purity/probing/P3-P5/t-SNE) |
| `150_triplet.py` | 삼중항 검정 (§4.3) |

### 캐시 / 산출물 (`/mnt/ddn/tmp/feature_analysis/`)

| 경로 | 내용 |
|---|---|
| `invariance_set.json` / `invariance_set2.json` | v1 / v2 평가셋 |
| `cache_invar_{ours,bc,cotrain}.npz` | v1 셋 step0 feature (200윈도) |
| `cache_invar_{ours,bc,cotrain}_p2.npz` | **v2 셋 step0 feature (301윈도)** |
| `paper_figs/invar/SUMMARY_invariance.{png,pdf}` | **요약 figure (A–D)** |
| `paper_figs/invar/pair_{action,target}.png` + `pair_decomp.json` | 2×2 분해 |
| `paper_figs/invar/mech_{action,target}.png` + `mech.json` | 메커니즘 판별 |
| `paper_figs/invar/flow.png` + `flow.json` | 정보 흐름 / RSA |
| `paper_figs/invar/P12_layers_*.png`, `T_tsne_*.png`, `triplet_*.png` | v1 셋 부속 결과 |

### 요약 figure 패널

| 패널 | 내용 |
|---|---|
| **A** | emb/task 대비 비율 (레이어별). L0–L2 음영 = Ours 만 1 아래 |
| **B** | L1 의 2×2 네 칸 원거리 — 같은/다른 사람 간극이 Ours 만 작음 |
| **C** | RSA(action, cur) — L3 이후 상승이 억제 소멸의 원인 |
| **D** | 전체 스케일 동일 — A 가 스케일 효과가 아님을 보증 |

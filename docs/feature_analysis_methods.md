# Mid-모델 Feature 분석: 측정 방법 (2026-08-05)

대상: mid 8K 체크포인트 3종 — **Ours** (`egoverse_ret_v3_448_state`, transfer/retrieval 레이아웃),
**BC** (`..._blankret_bv2_v10`, demo 비디오+state 전부 blank, 2슬롯 캡),
**Co-train** (`..._blankret_bv2_cotrain`, BC와 동일 blank + human/robot cotrain manifest).
결과물: `/mnt/ddn/tmp/feature_analysis/` (`final_visgrp/`, `final_egodex/`, `paper_figs/`).
스크립트: `egoverse_pair_builder/125`(캐시 추출)·`129`(메인 분석)·`131`(전수조사)·`132`(논문 figure).

## 1. 평가 셋 구성

**같은 language instruction + 시각적으로 유사한 관측** 쌍만 사용 (장면 prior를 쌍 내에서 통제).

- **Test set (visgrp)**: `egoverse_pairs_v10_test` (held-out 에피소드, 3 s 윈도).
  (cls, embodiment) 상위 4버킷(unpack_grocery/scale, put_object_in_container/scale,
  fill_scoop/MECKA, sort_bins/MECKA)에서, 현재 관측 프레임의 28×28 gray 썸네일(z-정규화)
  MSE가 0.55 이하인 쌍을 그리디 매칭 → **47쌍 = 94윈도(chunk)**.
- **EgoDex (held-out embodiment)**: `egodex_pairs_top3_3s.npz`. cls가 전부 상수이므로
  task는 에피소드명(`add_remove_lid__0`)에서 파생, 상위 8 task 버킷 → **50쌍 = 100윈도**.
- **Demo 슬롯 = 각 행의 저장된 retrieval top-1** (offline eval과 동일; 임의 조합 없음).
  BC/Co-train은 학습과 동일하게 dataset이 demo를 blank 처리.
- 정규화: 각 모델 **학습 manifest의 staterobust** (Ours/BC=v10, Co-train=v10_cotrain;
  egodex는 egodex-native robust) — test manifest에서 재계산 금지.

## 2. Feature/Attention 추출 (실제 생성 과정에서, GPU 1패스)

eval과 동일한 `generate_samples_from_batch(num_steps=35, guidance=1.5, seed=1)`
(rectified flow, 스텝당 denoise 1콜)을 돌리면서 훅으로 캐시:

- **hidden**: 전 28블록, 슬롯별(공간 28×28 평균) (T,2048), fp16 —
  스텝 {0, 5, 10, 15, 20, 25, 30, 34}. **스텝 0 = 타깃 슬롯이 순수 노이즈**인 생성
  시작점으로, 타깃 정답 누설이 0인 유일한 클린 프로빙 지점
  (t≤0.7의 1-step teacher-forcing 세팅은 정답 latent 잔존(30~90%)으로 세 모델이 동일하게
  나오는 것을 사전 실험으로 확인, 폐기).
- **frame-level attention** (T×T): 전 28레이어, 스텝 {0,17,34}. 쿼리 프레임 청크별
  QKᵀ softmax를 명시 계산(헤드 평균, 키 공간합·쿼리 공간평균).
- **spatial attention** (T×28×28): 전 28레이어, 스텝 {0,34}, 쿼리 슬롯
  [action, future 초/중/후].
- **출력**: 생성 action chunk(92×57, 정규화 공간), 최종 latent. 입력/생성 비디오는 mp4 분리.

## 3. 물리량 (GT) 정의 — 분산 분석(104)과 동일 함수

원시 state57에서, **정확히 chunk [w0, w1) = 90프레임 안에서만**, 양손(L/R) 각각 계산 후 평균:

- 속도 sp(t) = ‖∇pos‖·30 Hz. **접촉 프록시** t_c = peak 이후 첫 국소최소(sp < 0.25·peak).
- 축1 (retrieval 디스크립터 포함 → 순환 주의, 참고용): `disp_norm`, `pathlen`.
- 축2 (실현 dynamics): `rot_travel` = Σ geodesic(R_t, R_{t+1}), `mean_speed`, `peak_speed`,
  `s_peak` = argmax(sp)/(T−1) (감속 타이밍), `decel_len` = (t_c − t_peak)/(T−1),
  속도 프로파일 = sp를 24점 보간 → 양손 concat 48차원 → PCA 상위 3계수.
- 축3 (접촉, 프록시 의존): `contact_speed` = sp(t_c),
  `contact_rot` = geodesic(R₀, R_{t_c}) [deg] (104의 그룹 분산은 회귀 타깃이 될 수 없어
  윈도당 스칼라로 재정의).
- 스케일 지표(disp, pathlen, speeds)는 log1p(y/median) 변환.

## 4. D — 선형 프로빙 (chunk-wise)

- 샘플 = chunk: **X_i** = 해당 스텝·레이어 hidden에서 **타깃 토큰(future 비디오 + action
  슬롯)만 mean-pool**한 2048차원, **y_i** = 그 chunk의 물리량 스칼라.
- 회귀 = kernel(dual-form) ridge (α=10), **5-fold CV**, 평가 = Spearman ρ(예측, GT).
  (LOO는 소표본에서 체계적 음수 편향을 합성 대조로 확인, 배제. 파이프라인은
  양성(신호 심기)·음성(라벨 셔플)·검정력(n-곡선) 대조로 검증: n≥47에서 신뢰.)
- **전수조사**: 지표 9종 × 레이어 28 × 스텝 8 × 풀링 span 3종(targetall/action만/future만)
  × 3모델 → 지표별 Δ(ours−bc) 최대 셀 표 (`sweep_top.md`).
- **관측-prior 잔차 프로빙 (판정 지표)**: BC의 입력은 관측+명령뿐이므로 BC의 CV 예측치가
  "관측 prior"의 추정치. y에서 prior 성분을 선형 회귀로 제거한 잔차를 타깃으로 다시 프로빙.
  raw 프로빙에서 motion 축(rot_travel·speed·path)은 prior만으로 ρ≈0.6이 나와 판별 불가임을
  확인했고(3모델 동일), 잔차에서는 prior가 설명 못 하는 축(s_peak, decel_len, contact_speed)만
  남는다 — 여기서 Ours만 양의 ρ.

## 5. E — 표현 거리 vs 실현 차이 (연속 상관)

"motion 같음/실현 다름"의 이산 라벨은 불가(둘 다 같은 궤적에서 나옴)하므로 연속량으로:

- 쌍 = 같은 instruction·유사 관측 쌍 (§1). **실현 차이** = 축2+축3 7지표의 |Δz| 평균
  (양손 포함). **표현 거리** = 타깃-span feature의 평균 제거 후 cosine 거리.
- 판정 = Spearman ρ(실현차, 표현거리), 레이어×스텝 전 셀. 지표별 스윕 병행.
- **이산 형태(원문 요건) 병행**: 같은-motion 쌍을 실현차 중앙값으로 분할해
  "실현 같음" vs "실현 다름" 그룹의 거리 분포를 Mann-Whitney(단측)로 검정
  (거리는 same-그룹 중앙값으로 정규화, `Fig_E_discrete`). 결과: **EgoDex에서 원문 예측
  그대로 성립** — Ours만 diff>same 유의(p=0.024), BC p=0.49(방향 역), Co-train p=0.25.
  Test set은 Ours 최강(p=0.002)이나 baseline도 약하게 유의(0.038/0.007) — 같은 명령·유사
  관측 안에서도 실현차가 잔여 외관차와 상관돼 baseline 표현이 일부 잡기 때문. 따라서
  test set의 주 판정은 연속 상관(0.62 vs 0.33), 이산 형태는 EgoDex가 대표.

## 6. 출력 수준 (grounding)

생성 action chunk(정규화 공간)에 §3과 동일한 win_features를 적용해 생성 궤적의 실현량을 계산.
- **주 지표 = 생성 vs 쿼리 GT** (`Fig_output_vs_gt`): 실현량 예측 정확도.
  EgoDex 회전 0.72 vs BC 0.35, 접촉자세 0.60 vs 0.34; test set 접촉자세 0.58 vs 0.43,
  감속 타이밍 Ours만 양수.
- **보조 = 생성 vs demo** (`Fig_output_follow_demo`): demo 추종. 단 demo~GT가 retrieval로
  상관돼 raw 상관은 GT 경로로도 생기므로 판정은 Δ(ours−bc) — bc는 demo를 못 보니
  bc의 vs-demo 상관이 곧 GT 경유분.

## 7. Attention 판독

frame-level attention에서 future 쿼리 행들의 demo 키 구간 mass(키 공간 합, 쿼리·샘플 평균)를
레이어별로 비교. 시각화는 PushT `_heatmap_overlay`(per-frame min-max + cubic resize + JET
α0.55)로 demo/cur_img 프레임 위 오버레이.

## 8. 주요 결과 요약

- raw 프로빙의 motion 축: 3모델 동일 (관측 prior 축 — 판별 불가 확정).
- 잔차 프로빙: s_peak +0.30 / decel_len +0.25 / contact_speed +0.15 (Ours만 양수;
  baseline ≤ +0.04). EgoDex 재현 (sweep Δ +0.38/+0.27/+0.21).
- E: Ours 생성 초기 스텝(0~10) × 레이어 13~27에 ρ 0.4~0.6 밴드; composite 판정 셀
  test 0.62 vs BC 0.33, egodex 0.49 vs 0.24.
- 출력: 생성 vs demo — rot_travel 0.42/0.63(test/egodex), contact_rot 0.41/0.50
  (baseline 대비 ~2배).
- attention: Ours만 demo에 유의미 mass (전 레이어 5~10×; mid 레이어에서 demo 손 추적).

**논거 체인**: 읽기(attention) → 보존(잔차 D, E) → 사용(출력) — test set과 held-out
embodiment 양쪽에서 성립.

## 9. E 두 형태의 역할 분담 (판정 가이드)

| | 이산 형태 (원문 M.1) | 연속 형태 (ρ) |
|---|---|---|
| 정의 | 실현차 중앙값 분할 → same/diff 그룹 거리 Mann-Whitney | ρ(실현차, 표현거리) 전 쌍 |
| **EgoDex** | **원문 예측 성립: Ours만 유의 (p=0.024), BC n.s.(0.49)·Co-train n.s.(0.25)** | 0.49 vs 0.24 |
| **Test set** | Ours 최강 (p=0.002, diff/same 1.7×)이나 baseline도 약유의 (0.038/0.007) | **0.62 vs 0.33 — 주 판정** |
| 사용 | EgoDex 대표 (교과서적 성립) | 양 세트 공통 주 지표 |

Test set에서 baseline이 약하게 잡히는 이유: 같은 명령·유사 관측으로 묶어도 실현 차이가
잔여 외관 차이와 상관 → 중앙값 분할은 그 성분까지 섞임. 연속 상관은 gradation 전체를
써서 격차가 명확해짐. 논문 구성 권장 = **이산 박스플롯(EgoDex) + 연속 맵(양 세트)**.

## 10. 논문 figure 목록 (`/mnt/ddn/tmp/feature_analysis/paper_figs/`, PNG 300dpi + PDF)

팔레트: Ours=GREEN(146,176,130), BC=YELLOW(212,176,87), Co-train=BLUE(126,156,171);
막대/범례 순서 BC → Co-train → Ours.

| 파일 | 내용 | 역할 |
|---|---|---|
| `Fig_attention_demomass` | 전 28레이어 demo attention mass (step 0, future 토큰) | ① 읽기 |
| `Fig_residual_probing` | 관측-prior 잔차 probing (y∈[−0.12, 0.34], 잘린 바 값 표기) | ② 보존 (D) |
| `Fig_E_maps` | 연속 E ρ 레이어×스텝 맵 (Ours vs BC × 2세트) | ② 보존 (E 주판정) |
| `Fig_E_discrete` | 이산 E 박스플롯 (same/diff, same-중앙값 정규화, p값) | ② 보존 (원문 형태) |
| `Fig_output_vs_gt` | 생성 실현량 vs 쿼리 GT ρ | ③ 사용 (주지표) |
| `Fig_output_follow_demo` | 생성 실현량 vs demo ρ (판정=Δ(ours−bc)) | ③ 사용 (보조) |
| `Fig_tsne` | 판정 지점 feature t-SNE (색=실현 합성 z, 선=쌍) — 보조 | 보조 시각화 |


---

## 11. 추가 분석 (2026-08-05 오후 확정)

### 출력 지표 전수 스윕 (`output_metric_sweep.json`)
104 win_features 전 계열(17지표: 회전 총량/속도평균/속도피크/지터, 속력 평균/피크,
경로/변위, 접촉 속도/타이밍/자세/위치 xyz, 감속 타이밍/길이/률, 속도프로파일 PC3)로
생성 vs 쿼리GT Spearman. **발견**: test set 격차는 타이밍·접촉자세 계열에 집중
(contact_s +0.27, s_peak +0.26 — 잔차 probing과 동일 축), EgoDex는 전 dynamics 축 압도
(pathlen +0.40, rot_travel +0.39, mean_speed +0.37). Co-train 대비도 동일 구조
(EgoDex 회전 Δ+0.47). → `Fig_output_vs_gt` = 2세트 × 8지표.

### 부분상관: 복사 vs grounding (§6 보강)
rank-공간 잔차 부분상관 — gen~GT|demo (관측 적응 성분) vs gen~demo|GT (demo 잉여 성분):
- Ours: **GT|demo 우세**(test 회전 0.60·접촉자세 0.52) + **demo|GT 유의 잔존**(+0.22~0.31)
  = 관측 우선 + demo 추가 반영 (grounding 시그니처).
- BC: demo|GT ≈ 0 — demo 경로 부재의 음성 대조 성립.
- 예외(정직): EgoDex mean_speed는 Ours demo|GT(0.25) > GT|demo(0.17) — 속도축 복사 성향.

### Action DTW 오차 (`Fig_dtw_error`, `dtw_errors_both.npz`, `egodex_dtw.npz`)
양손 손목 위치(staterobust 역정규화, m), DTW/max(T):
| | Test set | EgoDex |
|---|---|---|
| Ours | **6.8cm** | **7.6cm** |
| BC / Co-train | 10.6 / 12.1 | 10.0 / 10.5 |
| Demo 복사 | 37.4 | 10.4 |
Ours < demo-복사 (양 세트) = 복사 아님; test set demo복사 37cm에도 Ours 6.8cm = 어려운
환경에서의 관측 정렬; EgoDex는 demo·관측 결합으로만 도달하는 정확도.

### 검증 2건 (반박 대비)
- **retrieval 누설 검사**: 두 세트 모두 demo=자기윈도 0 / 같은 에피소드 0 / 윈도 겹침 0.
  EgoDex에서 demo가 current와 비슷해 "보이는" 것은 같은-책상 반복 데이터셋 특성
  (+아래 mp4 함정) — demo는 항상 다른 에피소드.
- **표시용 mp4 함정**: `videos_{ds}/` mp4는 여러 실행(12샘플 구버전·중단된 300런)의
  잔재가 섞여 캐시 샘플과 불일치할 수 있음 → **시각화 프레임은 zarr 원본에서 직접 읽기로
  통일** (egodex 이미지 = `images.front_1`, 이중중첩 bytes: `arr[i].item().tobytes()` →
  imdecode). 기존 attention 정지그림 중 mp4 기반은 신뢰 금지.

### Attention 비디오 (모든 쿼리 시점, `attn_videos/`)
`137_attn_fullq.py`로 쿼리 전 슬롯(action+future 23)의 spatial attention을 재캡처
(생성 step0 입력과 동일: denoise(noise, noise, t=1000, cond)) →
`attn_fullq_egodex_s0_L{9,21}.mp4`: 시간축=쿼리 future t, 레이아웃
[fut_t 프레임 | cur_img attn | demo attn ×7], 행=3모델. 관찰: cur mass는 3모델 모두
0.13~0.17로 강함(baseline이 오히려 약간 높음), demo mass만 Ours 0.107 vs baseline 0.01대
— attention 수준에서도 "관측 유지 + demo 추가 읽기".

## 12. 실행 가이드 (재현)

환경 2개: **추출(GPU)** = `/mnt/ddn/cosmos_policy/.venv` (torch, zarr 없음·matplotlib 시각화 금지),
**분석(CPU)** = `/mnt/ddn/cosmos-framework/.venv` (zarr/scipy/cv2, sklearn 없음 → ridge는
numpy dual-form). 단독 GPU 실행에는 반드시
`RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=<port>` 필요.
스크립트 위치 = `/mnt/ddn/dataset/egoverse_pair_builder/`.

```bash
# 1) 생성 캐시 추출 (모델당 GPU 1개, visgrp 94: ours ~3h / bc·cotrain 각 ~1h)
ENV="RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29550"
$ENV CUDA_VISIBLE_DEVICES=0 /mnt/ddn/cosmos_policy/.venv/bin/python     125_cache_generation.py --model ours [--dataset egodex --limit 100] [--limit 8  # 스모크]
# -> cache_gen_{model}_{dataset}[_nN].npz + videos_{dataset}/{model}/*.mp4 (표시용, §11 함정 참조)

# 2) 메인 분석 (CPU): D 맵/판정 bar/잔차 + E 맵 + 출력 + attention 프로파일
/mnt/ddn/cosmos-framework/.venv/bin/python 129_final_analysis.py --dataset visgrp
#   egodex는 --dataset egodex (EGX_SFX=n100 TAG="" 환경변수로 캐시 접미사/출력 태그 제어)
# -> final_{dataset}/: D_map_*, D_delta_map, D_bars_step0, E_map, O_output_rho,
#    A_demomass_full, summary.json (li_star=판정 레이어 포함)

# 3) 전수조사 (CPU): 지표9 x 레이어28 x 스텝8 x span3
/mnt/ddn/cosmos-framework/.venv/bin/python 131_sweep_all.py --dataset visgrp
# -> final_{dataset}/sweep_{D,E}.json, sweep_top.md (지표별 best-Δ 셀 표)

# 4) 논문 figure (CPU; 팔레트 Ours=GREEN/BC=YELLOW/Cotrain=BLUE, 순서 BC->Cotrain->Ours)
/mnt/ddn/cosmos-framework/.venv/bin/python 132_paper_figs.py     # 잔차/E맵/출력2종/attention
/mnt/ddn/cosmos-framework/.venv/bin/python 136_E_discrete_fig.py # 이산 E 박스플롯
/mnt/ddn/cosmos-framework/.venv/bin/python 133_tsne_fig.py       # t-SNE (numpy 구현)
# -> paper_figs/*.{png,pdf}

# 5) 모든-쿼리-시점 attention 비디오 (GPU 캡처 ~20s/샘플 + CPU 렌더)
$ENV CUDA_VISIBLE_DEVICES=0 /mnt/ddn/cosmos_policy/.venv/bin/python     137_attn_fullq.py --model ours --dataset egodex --sample 0
# 렌더는 대화 중 인라인 스크립트(zarr 프레임 직접 읽기) — attn_videos/ 참조
```

파이프라인 함정 모음: NpzFile은 키 접근마다 재압축해제(`np.asarray(d["hid"])`로 1회
실체화 필수); npz 저장 직후 로드하면 BadZipFile(플러시 대기); pgrep 자기매칭(대기 루프
금지, 파일 존재+로드 검증으로); egodex pairs 키=names/tasks_ep/win_ep/win_w0/win_wl,
cls 상수라 task는 ep명 `rsplit("__",1)`; egodex 버킷은 task 파생 필수(아니면 12샘플 함정).


---

## 13. GR1 post-train 도메인 attention (2026-08-06)

**목적**: cur_img attention은 3모델 동일(관측 인코딩 공통 자산)이므로, 차이가 나는
demo 참조를 **post-train 도메인(GR1 robot)** 에서 확인.

**데이터/신뢰**: GR1 manifest 계보 정리 — `egoverse_pairs_gr1_v4`는 stale(존재하지 않는
에피소드 경로 참조). 유효 = **`egoverse_pairs_gr1_v6`** (에피소드 zarr in
`/mnt/ddn/dataset/egoverse`, HUM_ROOT 일치 검증). GR1 v6 **aligned dataset**(mid와 layout-exact:
state_t=50, chunk 197, 3cam 2x2 grid) at final_image_size=448. state는 pctl stats(데모 VIDEO는
픽셀이라 무관). 캡처 = `140_gr1_attn.py`: action 토큰 spatial attn 전28레이어 step0(=denoise
noise,noise,1000,cond), **입력 프레임을 npz에 직접 저장** → 렌더 자기완결(mp4/zarr 불필요,
스테일 위험 0). demo = aligned `__getitem__`의 저장 top-K 매치(np.random.seed(1000+idx)로
결정적, 3모델 동일 샘플).

**두 조건**:
- **real-demo-to-all**(140 기본): 3모델에 동일 real demo. bc도 mass 높음(0.25) — demo 슬롯이
  ~23/50 latent라 blank여도 token-baseline ~0.1을 먹기 때문 → **raw mass는 판별력 약**,
  공간 집중도(peak/mean)가 판별: ours L7-11 33~97 vs bc 16~38(같은 real demo에도 더 확산).
- **blank-demo**(`--blankdemo`, egodex 조건 재현): bc/cotrain demo 비디오+state 0.

**결과 (n=12, clean: ours real / bc·cotrain blank)** — `paper_figs/trusted/GR1clean_*.png`,
`GR1clean_demomass.png`:
- **ours demo mass(L5-11) 0.193 vs bc-blank 0.096 / cotrain-blank ~0.09** — 깨끗한 2× 분리,
  by-layer 곡선에서 ours(green)가 L3-27 전 구간 baseline 위. bc·cotrain blank는 서로 겹침.
- **ours가 egodex보다 GR1에서 demo를 3× 강하게 읽음** (mass 0.19 vs 0.06) — post-train 도메인이
  retrieval machinery의 본령이라 정합.
- 공간: ours는 GR1 2x2 그리드 씬 content에 구조적 attention(L5-11 분산 + L10-11 hotspot),
  blank baseline은 평탄+우측 edge-sink.

**주의(정직)**: mass 비율이 egodex(5×)보다 작은 2×인 것은 GR1 demo 블록의 큰 token 비중 때문
(blank baseline ~0.1). 판별의 핵심은 공간 구조/집중도. real-demo-to-all에서 bc mass가 높은 것도
같은 baseline 효과 — 구조를 봐야 함.

스크립트: `140_gr1_attn.py`(캡처), `141_gr1_viz.py`(real-demo-to-all viz),
`142_gr1_clean_viz.py`(clean 3-way). 산출 `gr1_attn_s{idx}_{model}[_blank].npz` + `gr1_mass_summary.json`.

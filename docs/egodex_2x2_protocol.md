# EgoDex 2×2 — 무엇을 어떻게 테스트했는가 (n=2048)

held-out embodiment(EgoDex)에서 **{ICL mid, BC mid} × {conditional, unconditional}** 네 셀을
같은 프로토콜로 채점한 실험의 프로토콜 기록. 결과표는
`/mnt/ddn/tmp/egodex_2x2_report_final.md` (42 tests) / `_3cell.md` (14 tests).

---

## 1. 네 셀의 정의

`cond` / `uncond` 는 **데모만 끄는 것이 아니라 레이턴트 레이아웃까지 바꾼다.** 이걸 놓치면
표를 잘못 읽는다.

| | conditional | unconditional |
|---|---|---|
| experiment | `cosmos_predict2p5_2b_egodex_ret_v3_448_state_eval` | `..._state_bv2blank_eval` |
| 데모 | **실제 매칭된 데모** (top-1) | **blank** |
| 레이아웃 | k=23, `state_t`=50 | k=2, `state_t`=29 |

이걸 두 체크포인트에 교차시킨다:

| 셀 | mid 체크포인트 | 출력 |
|---|---|---|
| `icl_cond` | `cosmos_predict2p5_2b_egoverse_ret_v3_448_state` @ 8K | `/mnt/ddn/tmp/egodex_eval_fid600_iter8000/egodex` |
| `icl_uncond` | 동일 (ICL) | `/mnt/ddn/tmp/egodex_eval_icl_uncond_2048_iter8000/egodex` |
| `bc_cond` | `..._ret_v3_448_state_blankret_bv2_v10` @ 8K | `/mnt/ddn/tmp/egodex_eval_bc_cond_2048_iter8000/egodex` |
| `bc_uncond` | 동일 (BC) | `/mnt/ddn/tmp/egodex_eval_bc_uncond_2048_iter8000/egodex` |

> ⚠️ `icl_cond` 의 디렉터리 이름이 `fid600` 인 것은 **이름만 남은 흔적**이다. 실제 내용은
> n=2048 이고(`summary.json: n_samples 2048`), 2×2 의 icl_cond 셀이 맞다. n=600 짜리 옛 런은
> `/mnt/ddn/tmp/egodex_eval_iter8000` 에 따로 있으니 혼동하지 말 것.

**대각선이 "자기 계약(own contract)"이다.**

|  | cond | uncond |
|---|---|---|
| **ICL** | ✅ 학습한 레이아웃 | ❌ off-contract |
| **BC** | ❌ off-contract | ✅ 학습한 레이아웃 |

BC 는 blank 레이아웃(k=2, `state_t`=29)으로 학습됐으므로 `bc_cond` 는 **BC 를 본 적 없는
레이아웃에 넣은 셀**이다. 이미지 지표가 그 셀에서 무너지는 것(`img_l1` 0.083, PSNR 18.6,
LPIPS 0.375 — 4셀 중 최악)은 성능 차이가 아니라 off-distribution 의 결과로 읽어야 한다.

---

## 2. 표본: 2048 윈도 / 2048 에피소드

- 매니페스트: `dataset/egodex_pairs_top3_3s.npz` (195,328 윈도, 46,234 에피소드, K=3,
  **within-EgoDex 검색** — EgoDex 는 embodiment 가 하나라 cross-emb 축이 없다).
- 열거 파라미터: `EPISODES=2048 PER_EPISODE=4 TOPK=1`.
- 실측 결과: **행 2048 / 서로 다른 에피소드 2048 / 고유 `idx` 2048** → 결과적으로
  **에피소드당 윈도 1개**. (per-episode 4 는 캡 이전 값이고, 2048 캡이 먼저 걸린다.)
- `TOPK=1` → 쿼리당 데모 1개 (within-EgoDex top-3 중 1순위).

**네 셀이 같은 윈도를 같은 순서로 채점한다 (검증함).** 열거는 pairs 테이블과 캡에만
의존하고 blank 플래그와 무관하기 때문이다. 네 CSV 의 `idx` 집합이 **정확히 동일**함을
확인했다:

```
icl_cond   rows=2048 eps=2048 idx_uniq=2048
icl_uncond identical idx set: True
bc_cond    identical idx set: True
bc_uncond  identical idx set: True
```

따라서 모든 대조가 **paired** 이고, 두 평균 비교보다 훨씬 민감하다.

### 정규화 통계 (EgoDex 전용)

`ACT_STATS=/mnt/ddn/dataset/egodex_state57_stats_staterobust.json`.
EgoDex 는 자기 embodiment 이므로 자기 액션 레인지를 쓴다. **1/99 퍼센타일이어야 하고
min/max 면 안 된다** — ARKit 프레임드롭 outlier 몇 개가 레인지를 17~100× 부풀려서
"transfer 안 됨" 이라는 잘못된 결론을 낸 전례가 있다
(`egoverse_pairing_v9_log.md` §EgoDex OFFLINE eval).

### 생성 설정

`num_steps=35`, `guidance=1.5`, seeded — 네 셀 동일. 해상도 448.

---

## 3. 지표

### 3.1 액션 — 모델 자신의 [-1,1] 정규화 공간

`cosmos_policy/scripts/eval_egoverse_ret_batch.py::_norm_space_metrics`.
논문 간 비교에 쓰이는 관례를 따른다: **액션 차원 × 청크 horizon 평균**.

| 지표 | 정의 |
|---|---|
| `msen_all` ↓ | `mean((|â − a|)²)` — 전 차원·전 horizon |
| `l1n_all` ↓ | `mean(|â − a|)` |
| `acc0.5` ↑ | `|â − a| ≤ 0.5` 인 **액션 차원의 비율** — 대략적 모션 의도 |
| `acc0.1` ↑ | 같은 것, τ=0.1 — 정밀도 |
| `*_eef` | 위 네 개를 eef 차원으로 한정 |

두 가지 설계 이유가 있다:

- **정규화 공간을 쓰는 이유**: 차원별로 scale-free 라 position·rotation·finger 가 비슷한
  비중으로 기여하고, 한 블록의 magnitude 폭주가 전체를 삼키지 못한다.
- **`action_mask` 를 쓰는 이유**: 샘플이 실제로 들고 있는 차원만 채점한다. 마스크된 차원
  (예: eva 행의 hand 차원)을 정답으로 세면 accuracy 가 부풀려진다.

### 3.2 궤적 / 영상

- `dtws_LR` ↓, `cos_LR` ↑ — 양손 손목 궤적 (scale-invariant DTW, per-step 방향 코사인)
- `img_l1` ↓, `img_psnr` ↑, `img_ssim` ↑ — 쿼리 클립 GEN vs GT 픽셀 (per-sample)
- `lpips` ↓ — per-sample 로 복원해서 붙임 (§4 참조)
- `FID`, `FVD` — **분포 거리라 per-sample 값이 없다 → 검정 불가, 점추정만 보고**

---

## 4. LPIPS per-sample 복원 (주의 지점)

`wilor_metrics.json` 은 LPIPS 를 행 단위로 저장하지만 **쿼리 인덱스를 남기지 않는다.**
`tools/wilor_metric.py` 가 `sorted(glob(f"{dump}/*.npz"))` 순서로 행을 만들기 때문에
i번째 행 = i번째 정렬 덤프이고, 각 덤프는 자기 `idx` 를 들고 있다. 이 매핑을 재구성해
per-sample LPIPS 를 붙인다.

**검증 없이 쓰면 안 된다.** 스크립트는 복원한 평균이 런이 보고한 SSIM/LPIPS 를 재현하는지
확인하고, 어긋나면 조용히 잘못 짝지은 컬럼을 내보내는 대신 **중단한다**.
(2048개 npz 헤더를 읽는 데 ~1분 걸려 `lpips_by_idx.csv` 로 캐시한다.)

---

## 5. 통계 프로토콜

`cosmos_policy/analysis_tools/egodex_2x2_stats.py`. A/B/C 리포트와 같은 방식이다.

1. **에피소드-매크로 집계** — 에피소드 안에서 평균 → 에피소드들 위에서 평균.
   (여기서는 에피소드당 윈도가 1개라 사실상 항등이지만, 윈도가 여러 개인 런과 정의를
   맞추기 위해 같은 경로를 쓴다.)
2. **paired Wilcoxon signed-rank** — 대조마다. 같은 윈도를 짝지으므로 유효하다.
3. **Cohen's dz** — 짝지은 에피소드 차분 위에서.
4. **Holm-Bonferroni** — 그 런의 모든 (지표, 대조) 검정에 걸쳐. `final` 리포트는 42개,
   `3cell` 은 14개 기준이라 **같은 대조라도 두 파일의 `p(Holm)` 이 다르다** (보정 대상
   개수가 달라서). raw `p` 는 동일하다.

재현:

```bash
cd /mnt/ddn/cosmos_policy
python analysis_tools/egodex_2x2_stats.py \
  --cell icl_cond=/mnt/ddn/tmp/egodex_eval_fid600_iter8000/egodex \
  --cell icl_uncond=/mnt/ddn/tmp/egodex_eval_icl_uncond_2048_iter8000/egodex \
  --cell bc_cond=/mnt/ddn/tmp/egodex_eval_bc_cond_2048_iter8000/egodex \
  --cell bc_uncond=/mnt/ddn/tmp/egodex_eval_bc_uncond_2048_iter8000/egodex \
  --out /mnt/ddn/tmp/egodex_2x2_report_final.md
```

`--contrast` 를 안 주면 **공급된 셀의 모든 쌍**을 검정한다.

---

## 6. 실행 방법

### 클러스터 (8 GPU × 1 node, 셀당 1잡)

```bash
kubectl apply -f /mnt/ddn/kube_yaml/cosmos-egodex-fid600-8gpu.yaml              # icl_cond
kubectl apply -f /mnt/ddn/kube_yaml/cosmos-egodex-eval-icl-uncond-2048-8gpu.yaml
kubectl apply -f /mnt/ddn/kube_yaml/cosmos-egodex-eval-bc-cond-2048-8gpu.yaml
kubectl apply -f /mnt/ddn/kube_yaml/cosmos-egodex-eval-bc-uncond-2048-8gpu.yaml
```

네 yaml 공통 환경: `EGODEX_SAMPLES=2048 PER_EPISODE=4 TOPK=1 VIDEO_METRICS=1
WILOR_EVERY=1 SKIP_POSE=1 FVD=1`.

### 로컬 2 GPU 순차 (BC 두 셀)

`/mnt/ddn/tmp/egodex_2x2_bc_driver.sh` — 출력 경로를 yaml 과 맞춰놨으므로 나중에
클러스터로 돌려도 같은 자리에 떨어진다.

### 비용

| | stage1 | stage2 | FVD | 합 |
|---|---|---|---|---|
| conditional (T=50) | 2048/8 × 48 s ≈ 3.4 h | ~24 min | ~31 min | **~4.3 h** |
| unconditional (T=29) | 더 쌈 | 동일 | 동일 | |

### 왜 n=2048 인가

**FVD 때문이다.** FVD 는 표본 수에 강하게 편향되고 n=600 에서는 전혀 수렴하지 않는다
(실측: n=50 → 530, 100 → 373, 150 → 311, 200 → 284 …). 액션·이미지 지표는
프레임 단위(2048 × 92장)라 600 에서도 이미 충분했다. **n 을 올린 이유는 오직 FVD 다.**

---

## 7. 이 설계로 답할 수 있는 것 / 없는 것

**답할 수 있는 것**

- 데모 조건화가 각 모델에 얼마나 도움이 되는가 (`icl_cond`↔`icl_uncond`, `bc_cond`↔`bc_uncond`)
- 데모를 뺏었을 때 두 모델이 어디에 서는가 (`icl_uncond`↔`bc_uncond` — 둘 다 BC 계약이라
  공정한 비교)

**답할 수 없는 것 — 표의 구멍**

`final` 리포트의 42개 검정에는 **`icl_cond` vs `bc_cond` 가 없다.** 지표당 3쌍
(icl 내부 / bc 내부 / uncond 끼리)만 돌렸고, 2×2 에서 빠진 대각선이 하필
**"둘 다 정상 입력일 때 우리가 BC 를 이기는가"** 다. 점추정으로는 icl_cond 가 전 지표
우위지만(`msen` 0.0325 vs 0.0370, `acc0.1` 0.600 vs 0.572) **유의성은 검정되지 않았다.**

단 이 대조는 해석이 까다롭다 — `bc_cond` 는 BC 에게 off-contract 셀이므로, 이겨도
"레이아웃 불일치를 이겼다" 와 "데모를 더 잘 읽는다" 가 섞인다. **BC 의 공정한 최선은
`bc_uncond`(자기 계약)** 이고, 그 대조(`icl_cond` vs `bc_uncond`)도 표에 없다.
같은 2048 에피소드 paired 데이터가 이미 있으므로 `--contrast` 로 몇 초면 추가된다.

---

## 8. 파일 맵

| 경로 | 내용 |
|---|---|
| `cosmos_policy/scripts/eval_egoverse_ret_batch.py` | 생성 + per-sample 지표 → CSV |
| `cosmos_policy/scripts/run_batch_8gpu.sh` | 샤딩 런처 (stage1 → stage2 → FVD) |
| `cosmos_policy/tools/wilor_metric.py` | stage2: SSIM/PSNR/LPIPS/FID |
| `cosmos_policy/tools/fvd_metric.py` | FVD (I3D Kinetics-400) |
| `cosmos_policy/analysis_tools/egodex_2x2_stats.py` | paired 검정 + Holm → 리포트 |
| `/mnt/ddn/tmp/egodex_2x2_report_final.md` | 4셀 / 42 tests |
| `/mnt/ddn/tmp/egodex_2x2_report_3cell.md` | BC 내부 대조만 / 14 tests |
| `/mnt/ddn/tmp/egodex_2x2_bc_driver.sh` | 로컬 2 GPU 순차 드라이버 |

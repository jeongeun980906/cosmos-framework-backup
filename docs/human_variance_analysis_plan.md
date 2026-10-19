# Human vs Robot 실현 분산 분석 — 실행 계획 (모델 학습 없음)

논문 도입부 주장 뒷받침용 데이터 통계 분석.
주장: (a) 로봇 데이터는 실현 분산이 작아 조건부 평균이 유효,
(b) human은 같은 motion 라벨에서도 분산이 커서 평균이 대표성을 잃음,
(c) motion 유사도로 정렬하면 그 분산이 줄어듦.

---

## 0. 데이터 소스 (확정)

| 그룹 | 소스 | 규모 (v10_cotrain manifest 기준) |
|---|---|---|
| ① 로봇 | **he** (Humanoid Everyday, cotrain에 쓰인 로봇 쿼리) | 92,136 윈도 / 태스크 라벨 = ep 접두 (`Articulated__adjust_...`) |
| ①' 로봇 보조 | eva (eva-robot) | 33,176 윈도 — he 결과의 재현 확인용 |
| ② human 같은-태스크 | **MECKA + scale + aria**, 같은 `cls` 라벨 묶음 | MECKA 558K / scale 373K / aria 111K 윈도 |
| ③ human motion-정렬 | ②와 동일 풀, **`pairs.npz`의 기존 retrieval top-K 이웃 그룹** | 재계산 불필요 (v10 매칭 재사용) |
| 보조 human | EgoDex (`/mnt/ddn/dataset/egodex_egoverse/<task>__<n>/state57` zarr) | held-out 검증용 (2차) |

- human 본 분석은 세 rig(MECKA·scale·aria) **전체 풀 × 전체 태스크**로 집계
  (MECKA 태스크: fill_scoop 122K·sort_bins 112K·do_dishes 87K 등 16종; cup/bagging은 scale·aria 전용).
- 분산 표/판정은 전체 풀 기준, **focus 태스크는 예시 그림용** (아래).

- manifest: `/mnt/ddn/dataset/egoverse_pairs_v10_cotrain/{windows,pairs}.npz` (`ep/w0/w1/emb/cls`)
- 에피소드 state57 zarr 접근은 `EgoVerseRetV3StateDataset`의 그룹 리더 재사용

**예시 그림용 focus 태스크 (scale∩aria 공통, 실측 교집합 상위):**

| cls | scale 윈도 | aria 윈도 |
|---|---|---|
| **cup_on_saucer** (유저 지목 "컵 옮기기") | 31,842 | 41,991 |
| **bagging_groceries** | 61,200 | 31,647 |

→ 본 분석(표·판정)은 전체 풀로 하되, **논문 그림 예시는 이 2개 태스크**로 뽑는다:
같은 라벨 안에서 rig(scale/aria)까지 섞이는 케이스라 "라벨 대비 실현 분산"이
시각적으로 가장 극적일 후보. rig 합산·rig별 분포 모두 산출.

## 1. 전처리

1. **좌표계**: state57의 wrist 위치(양손, convention 문서 인덱스) — waist 기준 정규화
   상태 그대로 사용 (전역 bias 제거됨 → 그룹 간 위치 비교 가능). **양손 모두 산출** (L+R 룰).
2. **스케일 정규화 유/무 2조건**: (무) waist-frame 미터 원값 / (유) per-emb min-max
   (`state57_stats.json`) 정규화 단위. 두 조건 분산 차 = embodiment 성분 크기.
3. **구간 절단**: 접근 시작 ~ 접촉 추정 시점.
   - 접촉 추정 = 손 속력의 **peak 이후 첫 국소 최소** (Savitzky-Golay 평활 후).
   - 접근 시작 = 접촉 이전 구간에서 속력이 peak의 10%를 마지막으로 상회한 시점.
   - **검출 실패 로깅 의무**: (peak 없음 / 국소최소 없음 / 구간 < 8프레임) 사유별 카운트를
     `detect_failures.json`으로 — 실패율 > 20%면 파라미터 재조정 후 재실행.
4. **시간축 정규화**: 절단 구간을 N=32 지점으로 리샘플 (s∈[0,1]).

## 2. 측정량

**위치량**
- P1. **접촉 시점 손 위치** (x,y,z 각각 — **z(높이)축 최우선**)
- P2. 접근 시작 위치
- P3. 전체 변위 벡터 (크기+방향)

**스타일량**
- S1. 감속 프로파일: 감속 시작 시점 s_dec (속력 peak 이후 감속 개시), 감속 구간 길이
- S2. 접촉 순간 속력
- S3. peak 속력 크기 + 타이밍 (s_peak)
- S4. 접근 방향 (변위 벡터의 단위방향, 구면각)
- S5. 궤적 직선성 (경로길이 / 시점-종점 직선거리)
- S6. 재조정 횟수 (속력 국소최소 개수 − 1)

## 3. 분석

**분석 1 — 그룹별 분산 비교 (핵심)**
- 그룹: ① he 로봇 (자기 태스크 라벨 내) / ② human 같은-태스크 (cup, bagging) /
  ③ human motion-top-K (K=8; 민감도용 K=16, 32)
- 각 측정량에 대해 3그룹 분포 겹친 히스토그램/바이올린 + 분산(σ²)·IQR 표, 스케일 유/무 별도.
- 판정: ②≫① → 주장(b) 성립; ③<② → 주장(c) 성립; ③≈② → motion 표현 재검토 신호;
  스케일 유/무 차 = embodiment 성분.
- 분산비 신뢰구간: 에피소드 단위 bootstrap 1000회 (윈도 단위는 자기상관으로 과신).

**분석 2 — 노이즈 바닥**
- 같은 클립 내 인접 프레임 2차 차분으로 고주파 jitter σ_noise 추정 (에피소드별 → 그룹 중앙값).
- 분석 1 모든 그림에 바닥선 표시. 판정: 관측 분산 ≫ 바닥선이어야 분석 1이 유효.

**분석 3 — 구간별 분산 구조**
- 정규화 시간축 s를 따라 그룹 내 위치 분산 곡선 σ²(s) (3그룹 겹침 + 바닥선).
- 판정: 접근 구간 낮고 접촉 근처 급등 → "거시 motion 공유, 접촉은 사례별" 성립. 평평하면 폐기.

## 4. 산출물 / 실행 순서

- 스크립트: `cosmos-framework/egoverse_pair_builder/95_variance_analysis.py` (단일 진입,
  `--task --emb --scale-norm --topk` 인자)
- 출력: `/mnt/ddn/tmp/var_analysis/` — plot(png), 표(csv), detect_failures.json
- 순서 (verify-one-before-batch 룰):
  1. cup_on_saucer × scale 50 에피소드 파일럿 → 절단/접촉 검출 오버레이 plot을 유저 검수
  2. 승인 후 전체 (cup+bagging × scale+aria × 스케일 유/무 × he/eva) 일괄
  3. 분석 2·3 동일 파이프라인에서 산출
  4. (2차) EgoDex로 held-out 재현

## 5. 리스크 / 반박 대비

- 접촉 추정이 태스크 의미와 어긋날 수 있음 (cup: 잡기 vs 놓기 두 접촉) → 1차 접촉(잡기)만
  사용, 파일럿 오버레이에서 육안 확인.
- he 태스크 ≠ human 태스크 (라벨 불일치) → 주장 (a)는 "로봇 자기 태스크 내 분산"으로 충분,
  단 표에 명시.
- top-K 그룹의 K 의존성 → K∈{8,16,32} 민감도 곡선 첨부.
- 윈도 중복(같은 에피소드 다수 윈도) → 모든 통계는 **에피소드 단위**로 집계.

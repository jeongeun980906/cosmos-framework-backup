# GR1 v6 데이터 법칙 — FINAL (2026-07-14 동결)

v5 전면 폐기 후 재설계 → 여러 반복 끝에 아래로 확정. **버전 업그레이드 시 전 제약 승계** (해제는 명시 지시로만).
용어: **retarget = GR1 학습 데이터 생성**, **augmentation = taskdemo 증식(demo 풀)**.
**최상위 원칙: 모든 설계 결정은 one-shot eval 순간(taskdemo 조건 + demo-앵커 씬 + tomato_1/corn_1)과의 일치로만 정당화한다.**

## 1. 궤적 법칙
에피소드 = `[open-loop approach @1cm/f, q0(홈+settle) 고정] → [handoff blend] → [demo 무변형 replay]`,
마지막에 success 1회 판정(성공 저장/실패 폐기), tail 없음.
- `TASK_RECIPE`: Door (30f,10f) / Plate (20,10) / Cabinet (0,16) / Faucet (0,16)
- **GRASP_DWELL** (실증 Plate 1/6→5/6): PnP는 닫힘 직전 12f 홀드 + 닫힘 6f 2배 감속 (gen_unified 상수)
- 알려진 잔여: Door 원거리 소스(갭 45cm+)에서 blend 속도 점프 — (45,10) 확장은 스모크 검증됐으나 미적용 (기존 50개 유지 결정)

## 2. 물체/씬 법칙 (pin 폐기, 2026-07-14)
- **GR1 데이터 = tomato(Cabinet)/corn(Plate) 메인**: 씬은 `--veg_set tomato`/`corn` (종 단위 제한, 인스턴스 pin 없음 — veg_pin 코드·데이터 삭제됨)
- 물체 **스왑 금지** (실측 0/8 — grip이 물체 크기에 기록돼 있어 종 교체 시 접촉 붕괴). 물체를 처음부터 맞춰 생성
- **spawn 4cm 법칙**: 씬의 veg 위치는 taskdemo 중심 4cm 이내만 통과 (eval 실측 변동 반경 4.1~4.3cm에 맞춤)
- 최종 선별 (50/task): Cabinet tomato 40 (L2 33) + 기타 10 / Plate corn 40 (corn_1 31) + 기타 10 / Door·Faucet 50 — **traj-근접 랭킹** (소스 taskdemo 궤적과의 리샘플 L2)

## 3. pairs 법칙 — FINAL v3 ("t=0만 빼고 전부 DTW")
- **t=0 행만 강제** (taskdemo@0, 3 슬롯). 나머지 전부 위치 기반 DTW — pad 상수·lockstep 산술·approach 마스크·음수 w0 **전부 폐기** (eval이 pad를 알 수 없다는 unseen-호환 요구)
- **매칭 space (26-d, state57 실측)**: `[pos3, rot6D×0.25, grip×0.4, Δpos×20] × 양팔` — 위치 지배 + 속도 보조, grip 유지
- **top-5 슬롯**: task별 dthdemo 풀(100/task) 전체에서 윈도우 DTW, 4-그리드 계산 + 사이 시점 포인터 +1 슬라이드 (구 jitter 조건과 동일 내용, 결정론)
- **one-shot 슬롯**: 쿼리 에피소드 전체 ↔ source taskdemo **글로벌 DTW 정렬 경로 φ(t)** — 초반 신호 부족은 경로 연속성이 해결 (200 eps 전수: 단조위반 0, 시작앵커 200/200)
- **행 = 모든 timestep** (t=0..L−36, stride 1) + dataset jitter 삭제 → **1 에폭 = 전 에피소드×전 timestep 정확히 1회** (로봇 데이터셋 표준)
- 풀 민감도 실측: 매칭 품질 K=5에서 포화(풀 중복성), 슬롯 다양성만 K 비례

## 4. 학습 타깃 — abs_action57
- state57(실측) 대신 **명령 절대 타깃** 예측: `abs_action57` = env actions(site frame) 변환, **row t = act[t]** (act[t]→s[t+1])
- 근거: 실측 state를 명령으로 재실행하면 추적 지연이 이중으로 쌓임; 명령 예측은 demo의 추적 동역학을 그대로 재현 + site frame이라 실행 시 변환·오프셋 불요 (body/site flip 리스크 소멸)
- stats: **min/max+5% 마진 전 차원** (GR1 gen 데이터는 오염 없음 → 클리핑 0; dataloader 검증 10/10 PASS, 오차 0.000cm)
- state57 조건 정규화: percentile-hybrid (pos/rot p0.5/99.5 + hand15 min/max+10%), `robust_state_stats=False`

## 5. eval 정합 (2026-07-14 수술 완료)
- **매칭 space 26-d 통일**: 서버 온라인 descriptor = 학습 frames20과 비트 동일 (검증 0.00e+00)
- **설정 A (기본)**: 클라 `--oneshot-pool3` → 서버 `oneshot_task` → 그 태스크 taskdemo **3개 전체** 윈도우 합집합에서 within-episode 매칭 (매칭이 ep까지 선택)
- **설정 B (백업)**: `--pool-kind dth --play-pool dth_pool_1x_v6.npz` (augmented 100/task)
- **L2/L3 티어**: `taskdemo_pool_v6.npz`에 base+L2+L3 전 변형 12종 포함 — task 변형 이름만 넘기면 자동
- **실행 경로**: 서버 `--pred-target abs_action57` (기본) — 예측 denorm(abs stats) → env 액션 직행 (`_abs_to_env_action`, ctrl offset 없음)
- **t0**: 클라 브릿지/AlignerProc 제거 — home+settle = 학습 t0
- 오브젝트/spawn: 기존 one-shot 세팅 유지

## 6. 학습 구성
- config `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state` (manifest `egoverse_pairs_gr1_v6`: windows 76,991 / queries 48,493 / K=6 — 2026-07-14 재빌드: 전수조사에서 t=1..3 행 누락 발견(81c emit이 t=4 시작), 첫 그리드 역슬라이드+클램프 백필로 수정. 발사 전 전수조사 재검증 PASS: coverage 200/200, round-trip 0.0000cm, 이동차원 포화 0/400)
- yaml `/mnt/ddn/kube_yaml/cosmos-gr1-posttrain-v6-448-state-3node.yaml`: max_iter=8K=cycle, batch 4, base=**old state-mid 8K** — clean-mid 8K 비교 FINAL (2026-07-14, viz_v2 재실행·GT 비트동일 검증): in-dist 3-샘플 old 우세 + **EgoDex 6-샘플 derived-delta wrist old 0.59 vs clean 1.03 cm/step** (동일 config/시드/정규화, 모델만 다름) → old 확정. ("clean 압승"이던 이전 수치는 delta-era 렌더러 오용 산출물 — 무효)
- viz 스크립트 주의: v2/v3 체크포인트 report는 **반드시 `viz_v2.py`** (state 모드/sample_geometry/fps30) — `eval_egoverse_ret.py`는 v1 전용 (v2에 쓰면 t=0 검은 프레임 + delta 적분 skeleton + fps16 5.75초 재생)

## 7. 파이프라인 스크립트 (pair_builder)
79c(변환, abs_action57 포함) / 80c(dthdemo 풀) / 81c(pairs, 26-d space) / 85c(one-shot 글로벌 DTW) / 82c(manifest) / 86(3캠 stride pair 비디오) / 87(taskdemo abs 백필) / 88(비교 비디오) / 90(taskdemo_pool_v6)

# GR1 데이터·Eval 전면 재설계안 (v1 — 논의용 초안)

목표: **demo-conditioned policy가 seen ≥80%, unseen(L2/L3)에서도 높은 성능 유지.**
원칙: **데이터 생성과 eval이 같은 분포 위에 존재** (사용자 결정: 파이프라인을 env 순정 법칙에 맞춤).

근거가 된 정독: augment_door / gen_pnp_cabinet / replay_to_gr1 / eval_gr1_client /
kitchen.py·play_env.py reset 법칙 / TaskDemos·play 데이터 계보(HF MimicDroid, 인간 teleop) /
action_policy_gr1_posttrain.md·gr1_inference_retrieval_spec.md / knob×stage 전수 테이블.

---

## 0. 현 구조의 4대 결함 (코드로 확인된 사실)

| # | 결함 | 근거 |
|---|---|---|
| 1 | **분포가 3원화**: env 순정 reset 법칙 ≠ 생성기의 인위 법칙(힌지 U[-1.6,-0.7], spawn ±3.5cm) ≠ eval의 고정 3씬(reset_to) | augment_door:142, gen_pnp_cabinet:135, eval_gr1_client:103-113 |
| 2 | **replay 성공 필터가 4번째 왜곡**: GR1 성공만 keep(25-50%) → kept 분포가 "쉬운 씬"으로 편향, train·eval 비대칭 | replay_to_gr1:194 |
| 3 | **demo-conditioning이 학습 불가능한 데이터**: 모든 에피소드가 task당 소스 demo 3개의 리타겟 → 씬+proprio가 행동을 유일 결정 → 데모를 읽을 정보이론적 이유 없음 (실측 demo-blindness Δ0.006) | 생성기 구조 전체 |
| 4 | **행동 기저 3개/task**: 공간 교란만 커버, 전략 다양성 zero | TaskDemos 구조 |

구조적 배경: TaskDemos의 4개 (layout,style)는 play 8개 env의 부분집합이고, L2(=unseen
objects)/L3(=unseen obj+env)는 Style/LayoutMixin으로 이미 정의돼 있음 — unseen 축은
인프라에 존재, 지금까지 활용 안 됨.

---

## 1. 분포 통일: D의 정의 (결함 1·2 해결)

task별 씬 분포를 **env 자신의 reset 법칙**으로 정의:

```
D_seen(task)   = ep_meta(layout=canonical, style=canonical) + placement_initializer 순정 샘플
                 + play reset 법칙(문 0.9-1.0 open, faucet off, yaw π/2±π/8 등)
D_L2(task)     = 동일 layout + held-out style/objects (StyleMixin 경로)
D_L3(task)     = held-out layout(7/10) + 014_l3 style + test object pool (LayoutMixin 경로)
D_final(tier)  = D_tier ∩ GR1-feasible        ← 핵심
```

**GR1-feasible 필터를 train·eval에 동일 적용**: 씬 draw의 가해성(可解性)은 "소스 demo
리타겟 → GR1 replay 성공"으로 판정. 이 필터를
- train: 지금처럼 keep 조건으로,
- **eval: seed 셋 확정 시 1회 적용** (불가해 seed 제외, 문서화)
하면 결함 2(필터 비대칭)가 사라짐 — 성공 필터가 왜곡이 아니라 **분포 정의의 일부**가 됨.

생성기 수정: 인위 법칙 전부 제거. `env.reset()`이 그린 씬을 **읽어서**(drawn hinge angle,
drawn placement) 리타겟 — 씬 랜덤화는 env가, 생성기는 리타겟만.

## 2. Eval 프로토콜 (3-tier)

| tier | 씬 | N | 목적 |
|---|---|---|---|
| seen | D_seen, held-out seed 20개 (feasible-filtered, 영구 고정) | 20/task | **게이트 ≥80%** |
| legacy | 기존 3 teleop 씬 (연속성 체크) | 3/task | 과거 숫자와 교량 |
| L2 | D_L2 draw 10-20 | 10+/task | unseen objects — **ONE-SHOT**: 매칭 L2 taskdemo를 조건 데모로 제공 |
| L3 | D_L3 draw 10-20 | 10+/task | unseen env — **ONE-SHOT**: 매칭 L3 taskdemo를 조건 데모로 제공 |

**L2/L3 프로토콜 (확정)**: retrieval이 아니라 해당 unseen 세팅에서 기록된 taskdemo 1개를
명시적 조건 데모로 주입 — "시연 하나 보고 unseen에서 따라하기"가 본 시험. 이 데모들은
eval 전용 입력이며 학습에 절대 미사용. (서버에 explicit-demo 주입 모드 필요 — Phase 3.)

클라이언트 수정: `reset_to(고정씬)` → **seeded `env.reset()`** 경로 추가 (ep_meta 템플릿 +
seed; eval seed 리스트는 매니페스트로 freeze — t0 상태·렌더 포함 저장해 재현성 보장).
부수 효과: "demo_1/2/3 라벨별 성패"라는 RNG 아티팩트도 소멸.

## 3. Demo-conditioning 학습가능성 (결함 3 — unseen 일반화의 본체)

**멀티태스크-동일씬 데이터**: 같은 씬 draw(부엌엔 문·수도·오브젝트가 동시에 존재)에
**여러 task의 소스 demo를 각각 리타겟**해 에피소드 생성 →
같은 관측에서 "무엇을 할지"를 **데모만이 결정**하는 반사실 구조가 데이터에 생김.
데모를 읽지 않으면 loss를 낮출 수 없게 됨 — demo-blindness의 구조적 해소.

구현: 씬 draw 1회 → {close door, turn faucet, pnp…} 중 해당 씬에서 가해한 task들의
리타겟 에피소드를 각각 기록 (SinkEnvForPlay 계열이라 전 task가 같은 씬 패밀리).
pairing은 기존 78 기계 그대로 (query의 데모는 해당 task의 play/taskdemo 매치).

측정: 학습 중 demo-swap probe(Δ가 커져야 함)를 정기 게이트로.

## 4. 행동 기저 확장 (결함 4 — 중기)

- 단기: 멀티태스크-동일씬으로 "씬당 행동 수"를 1→k로.
- 중기: play 데이터의 subgoal 주석(annotate_subgoals)으로 세그먼트 pseudo-demo를 발굴해
  리타겟 소스 풀 확장 (인간 play 8시간은 전략 다양성의 보고).

## 5. 기지 오픈 갭 처리 (spec §7)

- 온라인 retrieval descriptor: cosmos_policy 서버는 78 완전 포팅 검증됨(100% 재현) — OK.
  cosmos3 서버(action_policy_server_gr1)는 구 z-score — cosmos3 재사용 시에만 수정.
- FUT=32→29 floor: 빌더 span 36으로 확장해 4프레임 낭비 제거 (재빌드 시 함께).
- t0→play 풀 스위치: cosmos_policy 서버는 wired — OK.

## 6. 실행 계획 (phased, 총 ~3-4일)

| Phase | 내용 | 산출/게이트 | 예상 |
|---|---|---|---|
| 1 | **eval-set 정의 도구**: D_seen/L2/L3 seed 샘플 → feasibility 필터 → 매니페스트 freeze (t0+렌더) | eval 매니페스트 3-tier | 반나절 |
| 2 | **생성기 재작성**(통합 1본): 순정 reset draw + 리타겟 + 멀티태스크-동일씬 + feasibility 북키핑 → 4 task 재생성 | init 분포 = eval 분포 검증(coverage nn≈internal), 씬당 다중 task 에피소드 | 1-2일 (12 EGL 워커) |
| 3 | zarr/pairs 재빌드(FUT 36 포함) + post-train (mid v3@448와 병행 가능) | demo-swap Δ 상승 게이트 | 1일 |
| 4 | 3-tier eval | seen ≥80% 게이트, L2/L3 기록 | 반나절 |

## 7. 논의 필요한 결정점

1. **멀티태스크-동일씬의 task 조합**: 4 task 전부 한 씬에서? (layout이 task별로 다름 —
   canonical layout이 다르므로 "씬당 가능한 task 부분집합"으로 자연 제한됨. L0 layout엔
   door+pnp, L4엔 faucet… → layout별 가능 task 매트릭스를 Phase 1에서 산출)
2. **seen 분포의 폭**: placement 법칙을 순정 그대로(넓음) vs 약간 조임 — 80% 게이트와
   trade-off. 제안: 순정 그대로 + feasibility 필터만.
3. **legacy 3씬 유지 여부** (연속성 vs 단순화).
4. **play(retrieval demo) 쪽 분포**: 그대로 사용(권장 — 인간 play는 조건 신호이지
   행동 타겟이 아님) vs 재검토.
5. mid 재학습(v3@448)을 Phase 2와 병행할지.

# Mamba: The Quant Compression

### 연속-상태 시퀀스 모델의 중요도 인지 용량 배분
**Importance-Aware Capacity Allocation in Continuous-State Sequence Models**
— KV-캐시 압축 프론티어의 Mamba로의 일반화 (Priority-Conditioned Retention)

> Technical Report · 2026-1 성균관대학교 AI 시스템 · 안예준 · 정찬희 · 김수윤
> 상태: 주제 구체화(scoping) — ICLR/NeurIPS 스타일 정식 논문의 선행 문서

---

## 요약 (Abstract)

상태 공간 모델(SSM, 예: Mamba)은 과거 전체를 **상수 크기의 연속 hidden state** 하나로 압축해 O(N) 선형 추론을 얻지만, 그 대가로 특정 정보를 손실 없이 복원하는 associative recall에서 Transformer에 크게 뒤진다. 한편 Transformer 진영의 **KV-캐시 압축** 연구는 "한정된 메모리를 *중요한 것*에 몰아준다"는 importance-aware 압축(mixed-precision quantization, importance-based eviction)으로 성숙한 프론티어를 구축했다. 그러나 이 패러다임은 **토큰별 슬롯을 가진 이산 캐시**를 전제하므로, 슬롯이 없고 모든 정보가 한 벡터에 중첩되는 연속-상태 SSM에는 직접 이식되지 않는다.

본 연구는 importance-aware 압축을 특정 아키텍처에 묶인 트릭이 아니라 **"중요도 → 보존되는 정보량"이라는 아키텍처 불문 원리**로 재정식화한다. 이 원리의 두 구현을 하나의 축 위에 둔다: (i) 이산 캐시의 *토큰당 비트 할당*, (ii) Mamba의 *Δ-게이트 기반 차등 write/decay*. 후자는 전자의 **연속 극한**으로 볼 수 있다. 우리는 데이터단 우선순위 문법 `[P1]>[P2]>[P3]`으로 Mamba의 **암묵적 recency 압축 정책을 task-defined importance 정책으로 대체**하고, 합성 *priority-coupled MQAR* 위에서 오프라인 최적(Belady) 대비 **retention efficiency**로 정량화한다.

## 1. 문제 정의 (Problem)

Mamba의 hidden state는 고정 크기다. Jelassi et al.(2024)은 이로부터 GSSM이 정확한 복사에 **근본적 용량 한계**를 가짐을 증명했다(상태를 키워도 복사 학습에 Transformer 대비 ~100배 데이터 필요). 이는 recall 실패가 두 원인의 혼합임을 시사한다.

- **(a) Selectivity 천장** — 무엇을 보존/망각할지 *못 고르는* 문제(파싱·게이팅·학습). 고칠 수 있는 부분.
- **(b) Capacity 임계점** — 저장할 비트 자체가 부족한 정보이론적 한계(K* ≈ c·d_state). 물리적 한계.

또한 SSM의 기본 압축 정책은 **recency 편향**이다: Ā = exp(ΔA), A < 0 이므로 오래된 상태는 지수적으로 감쇠한다. 즉 Mamba는 *최근 것*을 자동으로 우선 보존하지, *중요한 것*을 보존하지 않는다. 본 연구의 출발점은 이 기본 정책을 **명시적·과제 정의적 importance 정책으로 교체**하는 것이다.

## 2. 핵심 통찰 — 이산↔연속 통합 축 (Central Thesis)

KV-캐시 압축 프론티어의 본질은 한 줄로 요약된다: **중요도가 높은 정보일수록 더 많은 비트(정밀도/슬롯)를 차지한다.** ZipCache는 salient 토큰을 고정밀, 일반 토큰을 저정밀, 무의미 토큰을 0-bit(eviction)로 두고, KVmix·KVTuner·MixKVQ는 중요도/쿼리 인지로 비트폭을 차등 배분한다. 즉 이미 사실상 **다단계(tiered) 압축**이며, 우리의 `[P1]/[P2]/[P3]`은 그 정확한 대응물이다.

문제는 메커니즘이다. **이산 캐시는 "토큰당 비트"를 할당하지만, Mamba는 슬롯이 없다.** 모든 정보가 hidden state 안에 물감처럼 중첩된다. 따라서 "P3를 2-bit로 양자화"는 물리적으로 성립하지 않는다. 이식 가능한 것은 *원리*이고, Mamba에서의 실현은 **선택적 게이트를 통한 차등 보존(differential retention)**이다.

> **통합 축 (Unifying Axis).** 한 항목 i에 할당되는 "보존 정보량"을 rᵢ라 하자.
> - 이산 캐시: rᵢ = 토큰 i에 배정된 비트폭 bᵢ (0이면 eviction)
> - 연속 SSM: rᵢ = 쿼리 시점까지 살아남는 토큰 i의 기여, 즉 게이트 곱 ∏ Āτ 가 결정하는 *유효 보존량*
>
> 두 경우 모두 정책은 **rᵢ = f(importanceᵢ)** 형태다. Mamba는 per-item precision allocation의 **연속 극한**이며, 본 연구는 importance-aware 압축이 **아키텍처 불문**임을 이 축으로 보인다.

## 3. 방법 (Method) — Priority-Conditioned Retention

### 3.1 우선순위의 정의: Value-of-Information

우선순위는 임의 라벨이 아니라 기대 회상 가치로 정의한다.

```
priority(i) ∝ P(i가 질의될 확률) × value(정확한 회상)
```

이는 고정 용량에서의 최적 보존을 **knapsack / water-filling** 문제로 만든다.

### 3.2 문법 (Grammar) — 입력단에만 개입, 아키텍처 무수정

```
PLAIN : k_a=v_1  k_b=v_2  k_c=v_3 ... [QUERY] k_a?
PCR   : [P1] k_a=v_1  [P3] k_b=v_2  [P2] k_c=v_3 ... [QUERY] k_a?
```

`[P1]`(절대 보존) > `[P2]`(보통) > `[P3]`(최우선 망각). 이 태그들은 special token으로 등록해 Mamba의 Δ 게이트가 제어 신호로 학습하도록 한다.

### 3.3 메커니즘 가설

태그가 Δ를 통해 **유효 보존량 rᵢ를 차등 조절**한다고 본다. 단, Ā = exp(ΔA)이므로 "Δ spike = 보존"은 단순 성립하지 않는다. 따라서 메커니즘의 *직접* 측정량은 raw Δ가 아니라 **surviving contribution** rᵢ = ∏(τ>i) Āτ — 토큰 i의 기여가 쿼리 지점까지 얼마나 살아남는가 — 이며, raw Δ는 보조 지표로 둔다.

## 4. 이론 (Theory)

- **용량 바운드.** hidden state가 약 |h| 비트를 담고 항목당 b 비트가 필요하면, 무손실 recall 조건은 K·b ≤ |h| → 임계점 **K\* ≈ c·d_state**.
- **우선순위 부분집합.** 보존 대상이 고우선 부분집합 K_P1뿐이면 K_P1·b ≤ |h|. 따라서 **총 K가 K\*를 넘어도 [P1] 회상은 유지**될 수 있다.
- **용량 배분 법칙 (목표 결과).** 각 레벨의 붕괴점 K\*_Pj 가 d_state와 우선순위 점유율의 닫힌 형태(water-filling)를 갖는지 검증한다.
- **기준선 & 지표.** 오프라인 최적은 **Belady's MIN**(가장 늦게 쓰일 항목부터 eviction). 성능은

  ```
  Retention Efficiency = Recall_model / Recall_Belady-MIN
  ```

  로 측정하고, 전체 gap을 **(정책 최적성 gap) + (연속-state 구현 gap)** 2단으로 분해한다.

## 5. 실험 설계 (Experimental Protocol)

### 5.1 데이터 — Priority-Coupled Multi-Query MQAR

Zoology/MQAR 하니스 기반, 무작위 토큰 ID(의미 편향 배제). 핵심 수정 두 가지:

1. **쿼리 분포를 우선순위에 결합** — [P1] key를 [P3]보다 훨씬 높은 확률로 질의(예 0.7/0.25/0.05)하거나 multi-query + priority-weighted loss.
2. **Multi-query streaming** — 쿼리를 시퀀스 중간중간 분산해 동적 eviction 상황을 만든다. 이로써 Belady가 *legit한* 오프라인 최적이 된다.

**부하 그리드.** K ∈ {4,8,16,32,64,128,256} 로그 스윕으로 K\*를 충분히 초과. [P1] 개수는 작게 고정, [P3] 개수로 총 K를 키워 "P1 유지 / P3 탈락" 분기를 보장.

### 5.2 모델 & 통제

| | Vanilla Mamba | **Grammar Mamba** | Transformer |
|---|---|---|---|
| 역할 | 통제군(기본) | **제안 방법** | 상한선(induction head) |
| 토큰 | 일반 | + `[P1..P3][QUERY]` | + `[P1..P3][QUERY]` |
| 구조 | Mamba-1, 2-layer | Mamba-1, 2-layer | Transformer, 2-layer |

- 파라미터·d_state 완벽 매칭, d_state ∈ {8,16,32} 스윕, 시퀀스 길이 통제.

### 5.3 학습 & 통제 변인

- **Loss masking**: 각 `[QUERY]` 직후 정답 Value 토큰의 CE만 역전파.
- **K 커리큘럼**: K=2 sanity → 점진 증가 (MQAR 수렴 난이도 완화).
- **Shuffled-tag null**: 태그를 쿼리 분포와 무상관으로 만들면 P1=P3 가 나와야 통제 성공.
- 시드 ≥3 + 에러바.

### 5.4 평가 파이프라인

1. **Surviving-contribution / Δ-Gate Probe** — forward hook으로 Δₜ 및 게이트 곱 rᵢ 추출, [P1] vs [P3] vs 일반 토큰 분포 비교.
2. **우선순위 분기 곡선** — 총 K를 키우며 [P1]·[P3] 회상률을 분리 기록. K\* 이후 교차가 핵심 증거.
3. **Retention efficiency** — Belady-MIN 대비 비율 및 2단 gap 분해.

## 6. 가설 (Hypotheses)

- **H1** — grammar는 saturation 이전 recall을 끌어올린다(selectivity 실패분 제거).
- **H2** — K\* ≈ c·d_state 이며 grammar로 거의 불변(d_state 스윕으로 검증).
- **H3** — 우선순위 문법은 포화 시 [P1]을 보존하고 [P3]부터 보존도를 낮춘다.
- **H4** — surviving contribution rᵢ가 [P1] > 일반 > [P3] 로 차등(메커니즘 증거).
- **H5** — [P3]는 저부하에선 회상되고 포화 이후에만 탈락한다(eviction-under-pressure).
- **H6** — 이산 캐시의 비트폭을 정하는 동일한 importance 신호가 Mamba의 유효 보존량을 예측한다.

## 7. 포지셔닝 (Positioning)

- **이산 KV-캐시 압축(Transformer, 성숙)**: KIVI, KVQuant, ZipCache, KVmix·KVTuner·MixKVQ, H2O 등 — 전부 **슬롯 기반 캐시 + random access** 전제.
- **SSM 측(미개척)**: 압축은 주로 weight/구조 pruning·quantization, mechanistic 연구는 recency/primacy 편향 보고. **in-context 정보의 importance-tiered retention은 공백.**

**차별점 (한 줄):** *기존 importance-aware 압축은 슬롯 기반 캐시와 random access를 전제하지만, 우리는 random access가 없는 **중첩된 상수 상태**에서, 아키텍처 수정 없이 **데이터단**으로 이를 실현한다.*

## 8. 기대 기여 (Contributions)

1. importance-aware 압축을 연속-상태 SSM으로 **일반화**한 첫 시도(데이터단·무수정).
2. 이산(비트)↔연속(Δ-게이트 보존)을 잇는 **통합 축**과 **용량 배분 법칙**.
3. surviving-contribution 기반 **메커니즘 증거**로 차등 보존을 인과적으로 입증.
4. Belady 대비 **retention efficiency** 평가 프로토콜과 2단 gap 분해.

## 9. 로드맵 (Roadmap)

| Tier | 내용 |
|------|------|
| **Tier 0** (8주, 과제) | priority-coupled MQAR + Belady 기준선 → 분기 곡선·Δ probe |
| **Tier 1** | 용량 배분 법칙 + 간섭(interference) 프런티어 |
| **Tier 2** | 사전학습 Mamba-2 실모델 전이 (프롬프트 우선순위 포맷팅 → recall↑) |
| **Tier 3** | activation patching 인과 검증 + priority-weighted bound 이론 |

**투고 목표:** NeurIPS 2026 Workshop (non-archival) → 확장본 ICLR 2027.

## 10. 팀

성균관대학교 2026-1 AI 시스템

| 이름 | 역할 | 연락 |
|------|------|------|
| 안예준 | Lead · Theory & Framing | canceler94@gmail.com · [github.com/AnYejun](https://github.com/AnYejun) |
| 정찬희 | Experiments & Data Harness | yumme@skku.edu |
| 김수윤 | Mechanism & Evaluation | jonathankim0514@gmail.com |

**프로젝트 홈페이지:** https://anyejun.github.io/mamba-quant-compression/

## References (key)

- Gu & Dao (2023). *Mamba.* arXiv:2312.00752 · Dao & Gu (2024). *Mamba-2.*
- Jelassi et al. (2024). *Repeat After Me.* ICML. arXiv:2402.01032.
- Arora et al. (2023). *Zoology / MQAR* · Arora et al. (2024). *Just Read Twice.* arXiv:2407.05483.
- Mohtashami & Jaggi (2023). *Landmark Attention.* NeurIPS. arXiv:2305.16300.
- Liu et al. (2024). *KIVI.* ICML · Hooper et al. (2024). *KVQuant* · He et al. (2024). *ZipCache.* NeurIPS.
- *KVmix* (2506.08018) · *KVTuner* (2502.04420) · *MixKVQ* (2512.19206) · *PM-KVQ* (ICLR'26).
- *Mamba-Shedder* (2501.17088) · *Primacy & Recency in Mamba* (2506.15156).

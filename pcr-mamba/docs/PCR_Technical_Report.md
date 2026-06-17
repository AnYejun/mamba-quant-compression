# 연속-상태 시퀀스 모델의 중요도 인지 용량 배분
### Importance-Aware Capacity Allocation in Continuous-State Sequence Models
**— KV-캐시 압축 프론티어의 Mamba로의 일반화 (Priority-Conditioned Retention)**

> **Technical Report** · 2026-1 AI 시스템 · 안예준 · 김수윤 · 정찬희
> 상태: 주제 구체화(scoping) — ICLR/NeurIPS 스타일 정식 논문의 선행 문서

---

## 0. 요약 (Abstract)

상태 공간 모델(SSM, 예: Mamba)은 과거 전체를 **상수 크기의 연속 hidden state** 하나로 압축해 $O(N)$ 선형 추론을 얻지만, 그 대가로 특정 정보를 손실 없이 복원하는 associative recall에서 Transformer에 크게 뒤진다. 한편 Transformer 진영의 **KV-캐시 압축** 연구는 "한정된 메모리를 *중요한 것*에 몰아준다"는 importance-aware 압축(mixed-precision quantization, importance-based eviction)으로 성숙한 프론티어를 구축했다. 그러나 이 패러다임은 **토큰별 슬롯을 가진 이산 캐시**를 전제하므로, 슬롯이 없고 모든 정보가 한 벡터에 중첩되는 연속-상태 SSM에는 직접 이식되지 않는다.

본 연구는 importance-aware 압축을 특정 아키텍처에 묶인 트릭이 아니라 **"중요도 → 보존되는 정보량"이라는 아키텍처 불문 원리**로 재정식화한다. 이 원리의 두 구현을 하나의 축 위에 둔다: (i) 이산 캐시의 *토큰당 비트 할당*(mixed-precision/eviction), (ii) Mamba의 *Δ-게이트 기반 차등 write/decay*. 후자는 전자의 **연속 극한**으로 볼 수 있다. 우리는 데이터단 우선순위 문법 `[P1]>[P2]>[P3]`으로 Mamba의 **암묵적 recency 압축 정책을 task-defined importance 정책으로 대체**하고, 합성 *priority-coupled MQAR* 위에서 오프라인 최적(Belady) 대비 **retention efficiency**로 정량화한다. 핵심 검증은, 용량이 임계점 $K^\*$를 넘을 때 저우선([P3]) 정보가 먼저 탈락하고 고우선([P1])은 보존되는 **eviction-under-pressure** 분기다.

---

## 1. 문제 정의 (Problem)

Mamba의 hidden state $h_t \in \mathbb{R}^{d_\text{state}}$는 고정 크기다. Jelassi et al.(2024)은 이로부터 GSSM이 정확한 복사에 **근본적 용량 한계**를 가짐을 증명했고(상태를 키워도 복사 학습에 Transformer 대비 ~100배 데이터 필요), 이는 recall 실패가 두 원인의 혼합임을 시사한다.

- **(a) Selectivity 천장** — 무엇을 보존/망각할지 *못 고르는* 문제(파싱·게이팅·학습). 고칠 수 있는 부분.
- **(b) Capacity 임계점** — 저장할 비트 자체가 부족한 정보이론적 한계($K^\* \approx c\cdot d_\text{state}$). 물리적 한계.

또한 SSM의 기본 압축 정책은 **recency 편향**이다: $\bar A=\exp(\Delta A),\ A<0$이므로 오래된 상태는 지수적으로 감쇠한다(Mamba의 primacy/recency 분석, 2025). 즉 Mamba는 *최근 것*을 자동으로 우선 보존하지, *중요한 것*을 보존하지 않는다. 본 연구의 출발점은 이 기본 정책을 **명시적·과제 정의적 importance 정책으로 교체**하는 것이다.

---

## 2. 핵심 통찰 — 이산↔연속 통합 축 (Central Thesis)

KV-캐시 압축 프론티어의 본질은 한 줄로 요약된다: **중요도가 높은 정보일수록 더 많은 비트(정밀도/슬롯)를 차지한다.** ZipCache는 salient 토큰을 고정밀, 일반 토큰을 저정밀, 무의미 토큰을 0-bit(eviction)로 두고; KVmix·KVTuner·MixKVQ는 중요도/쿼리 인지로 비트폭을 차등 배분한다. 즉 이미 사실상 **다단계(tiered) 압축**이며, 우리의 `[P1]/[P2]/[P3]`은 그 정확한 대응물이다.

문제는 메커니즘이다. **이산 캐시는 "토큰당 비트"를 할당하지만, Mamba는 슬롯이 없다.** 모든 정보가 $h_t$ 안에 물감처럼 중첩된다. 따라서 "P3를 2-bit로 양자화"는 물리적으로 성립하지 않는다. 이식 가능한 것은 *원리*이고, Mamba에서의 실현은 **선택적 게이트를 통한 차등 보존(differential retention)**이다.

> **통합 축 (Unifying Axis).**
> 한 항목 $i$에 할당되는 "보존 정보량"을 $r_i$라 하자.
> - 이산 캐시: $r_i = $ 토큰 $i$에 배정된 비트폭 $b_i$ (0이면 eviction).
> - 연속 SSM: $r_i = $ 쿼리 시점까지 살아남는 토큰 $i$의 기여, 즉 게이트 곱 $\prod_{\tau>i}\bar A_\tau$에 의해 결정되는 *유효 보존량*.
>
> 두 경우 모두 정책은 $r_i = f(\text{importance}_i)$의 형태다. Mamba는 per-item precision allocation의 **연속 극한**이며, 본 연구는 importance-aware 압축이 **아키텍처 불문**임을 이 축으로 보인다.

이 관점에서 본 연구의 기여는 "Mamba판 ZipCache"가 아니라 **"importance-aware 압축의 연속-상태 일반화 + 그 용량 배분 법칙"**이다.

---

## 3. 방법 (Method) — Priority-Conditioned Retention

### 3.1 우선순위의 정의: Value-of-Information
우선순위는 임의 라벨이 아니라 기대 회상 가치로 정의한다.
$$\text{priority}(i) \ \propto\ P(\text{$i$가 질의될 확률}) \times \text{value}(\text{정확한 회상})$$
이는 최신 KV 압축(MixKVQ의 query-aware importance)과 동일한 정신이며, 고정 용량에서의 최적 보존을 **knapsack/water-filling** 문제로 만든다.

### 3.2 문법 (Grammar)
입력단에만 개입한다. 아키텍처는 무수정.
```
PLAIN     : k_a=v_1  k_b=v_2  k_c=v_3 ... [QUERY] k_a?
PCR       : [P1] k_a=v_1  [P3] k_b=v_2  [P2] k_c=v_3 ... [QUERY] k_a?
```
`[P1]`(절대 보존) > `[P2]`(보통) > `[P3]`(최우선 망각, = `[FORGET]`의 극단). 이 태그들은 special token으로 등록해 Mamba의 $\Delta$ 게이트가 제어 신호로 학습하도록 한다.

### 3.3 메커니즘 가설 (왜 되는가)
태그가 $\Delta$를 통해 **유효 보존량 $r_i$를 차등 조절**한다고 본다. 단, $\bar A=\exp(\Delta A)$이므로 "$\Delta$ spike = 보존"은 단순 성립하지 않는다(현재 입력을 강하게 쓰는 동시에 과거를 빨리 감쇠). 따라서 메커니즘의 *직접* 측정량은 raw $\Delta$가 아니라 **surviving contribution** $r_i=\prod_{\tau>i}\bar A_\tau$ — 토큰 $i$의 기여가 쿼리 지점까지 얼마나 살아남는가 — 이며, raw $\Delta$는 보조 지표로 둔다.

---

## 4. 이론 (Theory)

**용량 바운드.** $h$가 약 $|h|$ 비트를 담고 항목당 $b$ 비트가 필요하면, 무손실 recall 조건은 $K\cdot b \le |h|$ → 임계점 $K^\* \approx c\cdot d_\text{state}$.

**우선순위 부분집합.** 보존 대상이 고우선 부분집합 $K_{P1}$뿐이면 $K_{P1}\cdot b \le |h|$. 따라서 **총 $K$가 $K^\*$를 넘어도 [P1] 회상은 유지**될 수 있다(바운드 위반이 아니라 차등 할당).

**용량 배분 법칙 (목표 결과).** 우선순위 분포가 주어졌을 때 각 레벨의 붕괴점 $K^\*_{P_j}$가 $d_\text{state}$와 우선순위 점유율의 함수로 닫힌 형태(water-filling)를 갖는지 검증한다. *비자명한* 법칙이 나오면 본 연구의 핵심 contribution이 된다.

**정규적 기준선 & 지표.** 오프라인 최적은 **Belady's MIN**(가장 늦게 쓰일 항목부터 eviction)이며, 이는 *multi-query streaming* 설정에서만 유의미하다(단일 쿼리에선 자명해짐). 성능은
$$\text{Retention Efficiency} = \frac{\text{Recall}_\text{model}}{\text{Recall}_\text{Belady-MIN}}$$
로 측정하고, 전체 gap을 **(정책 최적성 gap) + (연속-state 구현 gap)** 2단으로 분해한다. 후자가 Mamba 고유의 손실이다.

---

## 5. 실험 설계 (Experimental Protocol)

### 5.1 데이터 — Priority-Coupled Multi-Query MQAR
Zoology/MQAR 하니스 기반, 무작위 토큰 ID(의미 편향 배제). **핵심 수정 두 가지:**
1. **쿼리 분포를 우선순위에 결합** — 균등 쿼리면 prioritization을 학습할 유인이 없다. 따라서 [P1] key를 [P3]보다 훨씬 높은 확률로 질의(예 0.7/0.25/0.05)하거나, multi-query + priority-weighted loss를 쓴다.
2. **Multi-query streaming** — 쿼리를 시퀀스 중간중간 분산해 동적 eviction 상황을 만든다. 이로써 Belady가 *legit한* 오프라인 최적이 되고 eviction-under-pressure가 실제 동적 현상이 된다.

**부하 그리드.** $K \in \{4,8,16,32,64,128,256\}$로 $K^\*$를 *충분히 초과*하도록 로그 스윕. **[P1] 개수는 작게 고정**하고 [P3] 개수로 총 $K$를 키워(부하는 주로 저우선에서) "P1 유지 / P3 탈락" 분기를 보장한다.

### 5.2 모델 & 통제
| | Vanilla Mamba | **Grammar Mamba** | Transformer |
|---|---|---|---|
| 역할 | 통제군(기본) | **제안 방법** | 상한선(induction head) |
| 토큰 | 일반 | + `[P1..P3][QUERY]` | + `[P1..P3][QUERY]` |
| 구조 | Mamba-1, 2-layer | Mamba-1, 2-layer | Transformer, 2-layer |

- **파라미터·$d_\text{state}$ 완벽 매칭** ("커서 이겼다" 차단).
- **$d_\text{state} \in \{8,16,32\}$ 스윕** — $K^\*\propto d_\text{state}$(H2) 검증의 필수 조건.
- **시퀀스 길이 통제** — 태그로 길어진 만큼 baseline 길이도 맞춤.

### 5.3 학습 & 통제 변인
- **Loss masking**: 각 `[QUERY]` 직후 정답 Value 토큰의 CE만 역전파.
- **K 커리큘럼**: $K{=}2$ sanity(loss→0) → 점진 증가(작은 $K$에서 큰 $K$로). MQAR는 악명 높게 학습이 어려워 커리큘럼 없이는 큰 $K$ 수렴 위험.
- **Shuffled-tag null**: 태그를 *쿼리 분포와 무상관*으로 만든 데이터. 이때 P1=P3가 나와야 통제 성공(태그가 위치가 아닌 우선순위 의미를 추종함을 확인).
- **시드 ≥3 + 에러바.**

### 5.4 평가 파이프라인
1. **Surviving-contribution / Δ-Gate Probe (H4)** — forward hook으로 $\Delta_t$ 및 게이트 곱 $r_i$를 추출, [P1] vs [P3] vs 일반 토큰의 분포를 통계적으로 비교(시각화 + 검정).
2. **우선순위 분기 곡선 (H3, H5)** — 총 $K$를 키우며 [P1]·[P3] 회상률을 *분리* 기록. K* 이후 [P3] 급락 / [P1] 유지의 **교차**가 핵심 증거.
3. **Retention efficiency (H6)** — Belady-MIN 대비 비율 및 2단 gap 분해.

---

## 6. 가설 (Hypotheses)

- **H1** — grammar는 saturation 이전 recall을 끌어올린다(selectivity 실패분 제거).
- **H2** — $K^\* \approx c\cdot d_\text{state}$이며 grammar로 거의 불변($d_\text{state}$ 스윕으로 검증).
- **H3** — 우선순위 문법은 포화 시 [P1]을 보존하고 [P3]부터 보존도를 낮춘다.
- **H4** — surviving contribution $r_i$가 [P1] > 일반 > [P3]로 차등(메커니즘 증거).
- **H5** — [P3]는 저부하에선 회상되고 포화 이후에만 탈락한다(eviction-under-pressure의 인과적 증거).
- **H6 (통합/법칙)** — 이산 캐시에서 비트폭을 정하는 동일한 importance 신호가 Mamba의 유효 보존량을 예측하며, 용량 배분 법칙이 두 레짐에 공통으로 성립한다.

> 어느 쪽 결과든 (a)/(b) 분해라는 기여는 성립하며, H6가 비자명하게 성립하면 본 연구는 단순 이식을 넘어선다.

---

## 7. 관련 연구 & 포지셔닝 (Positioning)

**이산 KV-캐시 압축(Transformer, 성숙).** mixed-precision quantization — KIVI(ICML'24), KVQuant, ZipCache(NeurIPS'24, salient=고정밀/일반=저정밀/0-bit=eviction), KVmix·KVTuner·MixKVQ(query-aware)·PM-KVQ(ICLR'26); importance eviction — H2O·Attention-Gate(2410.12876)·Cache What Lasts(2512.03324). → 전부 **슬롯 기반 캐시 + random access** 전제.

**SSM 측(미개척).** 압축은 주로 weight/구조 pruning·quantization(Mamba-Shedder, 2501.17088)이고, mechanistic 연구는 recency/primacy 편향을 보고(2506.15156). **in-context 정보의 importance-tiered retention은 공백.**

**Recall 한계 이론·벤치마크.** Jelassi 2024(2402.01032), Zoology/MQAR 2023, JRT 2024(2407.05483), Landmark Attention 2023(2305.16300).

**우리의 차별점 (한 줄).** *"기존 importance-aware 압축은 슬롯 기반 캐시와 random access를 전제하지만, 우리는 random access가 없는 **중첩된 상수 상태**에서, 아키텍처 수정 없이 **데이터단**으로 이를 실현한다."*

---

## 8. 기대 기여 (Contributions)

1. importance-aware 압축을 연속-상태 SSM으로 **일반화**한 첫 시도(데이터단·무수정).
2. 이산(비트)↔연속(Δ-게이트 보존)을 잇는 **통합 축**과 **용량 배분 법칙**.
3. surviving-contribution 기반 **메커니즘 증거**로 차등 보존을 인과적으로 입증.
4. Belady 대비 **retention efficiency** 평가 프로토콜과 2단 gap 분해.

---

## 9. 로드맵 & 투고 (Roadmap)

- **Tier 0 (8주, 과제)** — priority-coupled MQAR + Oracle regime + Belady 기준선 → 분기 곡선·Δ probe.
- **Tier 1** — 용량 배분 법칙 + 간섭(interference) 프런티어.
- **Tier 2** — 사전학습 Mamba-2 실모델 전이("프롬프트 우선순위 포맷팅 → recall↑").
- **Tier 3** — activation patching 인과 검증 + priority-weighted bound 이론.

**투고:** ICLR 스타일 템플릿으로 작성, 1차 목표 **NeurIPS 2026 Workshop**(논문 마감 ~9월, non-archival → 확장본 ICLR 2027 재투고). efficient/compression 워크샵 핏 강함.

---

## 10. 리스크 & 한계 (Risks)

- **프레이밍 규율** — "Mamba 양자화"로 서술하면 안 된다(연속 state엔 비트-슬롯이 없음). "차등 보존 / 용량 배분"으로 일관.
- **약효 위험** — $\Delta$의 dynamic range가 좁거나 recency 편향이 우세하면 우선순위 효과가 미약할 수 있음 → 그 경우 네거티브 결과(덜 화려하나 출판 가능).
- **혼잡한 인접 영역** — ZipCache 등과의 차별화 요구가 확실히 옴 → §7의 한 줄(중첩 상수 state·무 random access·데이터단)로 방어.
- **transfer 성격** — 본질이 이식+통합+법칙이라 "완전히 새로운 메커니즘"은 아님. 기대치는 강한 워크샵 ~ 조건부 본 트랙.

---

## References (key)
- Gu & Dao (2023). *Mamba.* arXiv:2312.00752.  · Dao & Gu (2024). *Mamba-2.*  · *Mamba-3* (2026). arXiv:2603.15569.
- Jelassi et al. (2024). *Repeat After Me.* ICML. arXiv:2402.01032.
- Arora et al. (2023). *Zoology / MQAR.*  · Arora et al. (2024). *Just Read Twice.* arXiv:2407.05483.
- Mohtashami & Jaggi (2023). *Landmark Attention.* NeurIPS. arXiv:2305.16300.
- Liu et al. (2024). *KIVI.* ICML.  · Hooper et al. (2024). *KVQuant.*  · He et al. (2024). *ZipCache.* NeurIPS.
- *KVmix* (2506.08018) · *KVTuner* (2502.04420) · *MixKVQ* (2512.19206) · *PM-KVQ* (ICLR'26).
- *Attention-Gate KV Eviction* (2410.12876) · *Cache What Lasts* (2512.03324).
- *Mamba-Shedder* (2501.17088) · *Primacy & Recency in Mamba* (2506.15156).

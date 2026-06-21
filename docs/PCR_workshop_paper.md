# Priority-Conditioned Retention: A Controllable Middle-Layer Memory Circuit in Pretrained Mamba

**안예준, 정찬희, 김수윤**
성균관대학교 · 2026-1 AI 시스템
*Workshop submission draft (non-archival) — single-model mechanistic study*

---

## Abstract

State-space models (SSMs) such as Mamba compress an entire history into a
fixed-size continuous hidden state, achieving linear-time inference at the cost
of associative recall: unlike attention, they cannot losslessly retrieve an
arbitrary past token. Prior work attributes this to a capacity bound, but leaves
open whether the practical failure is a *storage* limit or a *control* (which to
keep) limit. We show that a **pretrained** Mamba-790m already contains a
**controllable, spatially localized memory circuit**, accessible at inference
without any fine-tuning. By scaling the pre-softplus input to the Δ gate at
positions downstream of a target, we reduce the effective decay applied to that
target and recover associative recall on a capacity-pressured multi-query
associative-recall (MQAR) task (0.53 → 0.73 at the optimal operating point).
The effect is **dose-dependent, target-specific, and bidirectional**, and it is
**localized to layers 16–23** (≈33–50% relative depth). Critically, restricting
the intervention to these 8 of 48 layers recovers most of the recall gain at
**near-zero language-modeling cost (perplexity ×1.07)**, whereas the all-layer
intervention inflates perplexity ×1.81. We frame this as the continuous-state
analogue of importance-aware KV-cache compression, and report an honest
boundary: a literal write-quantization analogue of KV-cache bit allocation does
*not* transfer, because the superimposed state has no per-item slots to
reallocate. Our findings establish that Mamba's recall behavior is, in part, a
controllable property of a specific depth range rather than a fixed
architectural ceiling.

---

## 1. Introduction

Mamba and related selective SSMs match Transformers on many language tasks while
running in linear time, but they remain markedly weaker at *associative recall* —
retrieving a specific value bound to a key seen earlier in context. Jelassi et
al. (2024) show this stems from a fundamental capacity constraint: a fixed state
cannot losslessly copy arbitrarily long inputs. Yet a capacity bound does not, by
itself, explain *behavioral* recall failure at moderate load. The failure could
be (a) a **control** problem — the recurrence applies a default recency decay and
does not preferentially keep query-relevant items — or (b) a **storage** problem —
the state is genuinely out of bits. These have opposite implications: (a) is
fixable at inference; (b) is not.

This paper provides evidence for (a) in a regime where it matters. We do **not**
modify or retrain the model. Instead we ask: *does a pretrained Mamba already
expose a handle that lets us trade its default recency policy for a
target-preserving one?* We find that it does, that the handle is the Δ gate,
and — most importantly — that the handle is **localized to a contiguous band of
middle layers**. Operating only on that band recovers recall essentially for
free in terms of language-modeling quality, which is the behavior one expects of
a dedicated mechanism rather than a diffuse one.

**Contributions.**
1. A test-time intervention on the Δ gate that controllably preserves a target
   item in a capacity-pressured SSM, with dose-response, specificity, and
   bidirectionality controls (§4.2).
2. Localization of this control to **layers 16–23** of Mamba-790m, and a
   **surgical** result: the 8-layer intervention matches most of the full-model
   recall gain at perplexity ×1.07 vs. ×1.81 (§4.3–4.4). This is our central
   finding.
3. A framing that places this control on a single axis with importance-aware
   KV-cache compression, together with an honest negative result: a literal
   write-quantization analogue fails, bounding how far the cache analogy carries
   (§5).

We report scope honestly: all positive results are for Mamba-790m on synthetic
MQAR; we observe non-replication on Mamba-1.4b and fragility on natural language
(§6).

---

## 2. Background and Framing

**The Δ gate as a retention knob.** In a selective SSM the state updates as
`h_t = Ā_t h_{t-1} + B̄_t x_t`, with `Ā_t = exp(Δ_t A)` and `A < 0`, so
`Δ_t > 0 ⇒ Ā_t ∈ (0,1)`. The contribution of token *i* surviving to a query at
position *T* is therefore `B̄_i x_i · Π_{τ>i} Ā_τ`. Two facts follow. First, the
default policy is recency-biased: older contributions decay geometrically.
Second, the *survival* of an item is governed by the **downstream** decays
`Π_{τ>i} Ā_τ`, not by the item's own write alone. Reducing Δ at positions after a
target pushes those `Ā_τ → 1`, slowing the decay the target experiences — a
preservation lever that requires no weight change.

**Discrete ↔ continuous importance allocation.** The KV-cache compression
literature (KIVI; KVQuant; ZipCache; and rotation-based vector quantizers such as
TurboQuant) implements one principle: *spend more bits on more important tokens.*
This presumes a slotted cache with random access. Mamba has no slots — all items
are superimposed in one vector. We therefore treat "importance → retained
information" as architecture-agnostic and instantiate it in Mamba not as
per-token bit allocation but as **differential retention** through the Δ gate.
We name this framing Priority-Conditioned Retention (PCR). §5 reports where the
analogy holds and where it breaks.

---

## 3. Method

**Task.** Multi-query associative recall (MQAR) with random single-token
key/value words (no semantic priors). A prompt presents *N* key–value pairs
followed by a query key; the model must emit the bound value. Capacity pressure
is induced by increasing *N*. We query the first pair as the default target.

**Intervention.** We register forward hooks on each layer's `dt_proj` and
multiply its pre-softplus output by a per-position gain `g`. Setting `g < 1`
("other-gain") at positions downstream of the target's value reduces their
effective decay. A layer mask restricts the intervention to a chosen subset of
layers. The model runs on its pure-PyTorch sequential path (the fused kernels are
intentionally not installed), so the Δ pathway is hookable. No parameters are
updated.

**Model and metrics.** `state-spaces/mamba-790m-hf` (48 layers, fp32). We report
top-1 recall over 30 random seeds per condition. For the language-modeling
guardrail we report perplexity on a fixed held-out passage under the same gain,
relative to the unmodified model.

---

## 4. Results

### 4.1 Capacity curve

Base recall declines with load, giving a usable pressure regime:

| N | 16 | 24 | 32 | 48 | 64 |
|---|----|----|----|----|----|
| recall | 0.77 | 0.63 | 0.53 | 0.43 | 0.33 |

We fix **N = 32** (base 0.53) for the intervention experiments — enough pressure
to leave headroom without collapsing.

### 4.2 Test-time control (dose, specificity, direction)

Sweeping the other-gain at N = 32 over all layers yields a clear inverted-U with
an optimum at **og = 0.8**:

| og | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.4 |
|----|-----|-----|-----|-----|-----|-----|
| recall | 0.53 | 0.60 | **0.73** | 0.60 | 0.30 | 0.00 |

Too much suppression (og ≤ 0.6) halts integration and recall collapses. The
effect is **target-specific**: applying og = 0.8 to *downstream* positions yields
0.73, while applying the same budget to *random* positions yields only 0.50
(near base). This rules out a generic "more gain helps" artifact.

### 4.3 Layer localization

Restricting og = 0.8 to 8-layer bands (base 0.53) isolates the responsible depth:

| band | L0–7 | L8–15 | **L16–23** | L24–31 | L32–39 | L40–47 |
|------|------|-------|------------|--------|--------|--------|
| Δ recall | +0.00 | +0.00 | **+0.13** | +0.07 | +0.00 | +0.00 |

Recovery concentrates in **L16–23**, with a secondary contribution at L24–31 and
nothing elsewhere. The early and late thirds of the network are inert for this
control, consistent with a dedicated memory-handling depth range rather than a
diffuse property.

### 4.4 Surgical intervention (central result)

The localization is not merely descriptive — it is *cheaper*. Comparing the
all-layer intervention to the L16–23-only intervention (base perplexity ≈ 232):

| intervention | recall | perplexity |
|--------------|--------|------------|
| all layers, og = 0.8 | 0.73 | **×1.81** |
| **L16–23 only, og = 0.8** | 0.67 | **×1.07** |

The 8-layer intervention captures most of the recall gain at **near-zero
language-modeling cost**, while the all-layer version pays a large perplexity
tax. A targeted edit to the memory circuit is almost free; a global one is not.
This is the SSM analogue of operating on a specific circuit rather than perturbing
the whole network.

---

## 5. The discrete ↔ continuous boundary (a negative result)

If retention is one lever, *precision* — the literal KV-quant analogue — is a
candidate second lever: quantize the write of low-priority tokens to fewer bits.
We tested this directly by quantizing the per-position conv output (the token's
written representation) to *k* bits.

- **Single-item precision** is not a graceful knob: target recall is flat from
  16→8 bits, then cliffs (8-bit 0.53 → 4-bit 0.03 → 0). The activation tolerates
  8-bit but not 4-bit.
- **Reallocation fails**: quantizing the *distractor* positions to 4 bits drives
  target recall to **0.00** — it corrupts the shared computation rather than
  freeing capacity for the target.

This is informative, not merely a null. Because the state is a superposition with
no random-access slots, lowering one item's write precision either does nothing
(≥8-bit) or injects noise that the whole sequence shares (≤4-bit). There is no
per-item bit budget to reallocate. The importance-allocation principle thus
transfers to Mamba **only** through retention (§4), not through quantization — a
concrete boundary on how far the KV-cache analogy carries, and an empirical
instance of the "no slots" property that distinguishes continuous-state memory
from a discrete cache.

---

## 6. Limitations and scope

We state these plainly; they bound the claim and we believe reporting them
strengthens it.

- **Single model.** All positive results are for Mamba-790m. On Mamba-1.4b the
  same intervention shows no positive circuit at a comparable pressure regime
  (all layer-band Δ ≤ 0 despite ample headroom). The L16–23 circuit therefore
  appears **model-specific**; we do not claim a universal depth.
- **Synthetic task.** Effects are robust on random-token MQAR but fragile on
  natural-language multi-fact recall, where the same intervention can hurt.
- **Graded priority is not established.** A three-level graded variant appeared
  monotone but is confounded with query position; we do not claim multi-level
  allocation. The SSM constraint (`Δ>0 ⇒ Ā<1`, shared decay) predicts that
  truly item-selective allocation is limited, consistent with §5.

---

## 7. Related work

**SSM capacity.** Jelassi et al. (2024) establish the copying/capacity limit;
work on primacy/recency in Mamba documents the default decay policy. We add a
*controllability* result on top of the capacity picture. **KV-cache
compression.** KIVI, KVQuant, ZipCache and rotation-based quantizers
(TurboQuant, Zandieh et al., 2026; ICLR 2026) realize importance-aware bit
allocation on slotted caches; we ask whether the principle survives the move to a
slotless continuous state, and find it survives as retention but not as
quantization. **Mechanistic interpretability.** Localizing a behavior to a
contiguous layer band parallels circuit-style findings in Transformers; ours is,
to our knowledge, an early such localization of a *controllable* memory mechanism
in a pretrained SSM.

---

## 8. Conclusion

A pretrained Mamba's associative recall is, in part, a **controllable property of
a specific middle-layer band**, not a fixed ceiling. A surgical, training-free
intervention on layers 16–23 recovers recall at negligible language-modeling
cost. The importance-aware compression principle from the KV-cache frontier
transfers to continuous state through *retention*, but not through
*quantization* — the superimposed state has no slots to reallocate. We hope the
localization result, and the honest boundary around it, are useful to ongoing
work on what fixed-state sequence models can and cannot remember.

---

### References (key)

- Gu & Dao (2023). *Mamba.* arXiv:2312.00752. · Dao & Gu (2024). *Mamba-2.*
- Jelassi et al. (2024). *Repeat After Me.* ICML. arXiv:2402.01032.
- Arora et al. (2023). *Zoology / MQAR.*
- Liu et al. (2024). *KIVI.* · Hooper et al. (2024). *KVQuant.* · He et al. (2024). *ZipCache.*
- Zandieh et al. (2026). *TurboQuant.* ICLR 2026. arXiv:2504.19874.

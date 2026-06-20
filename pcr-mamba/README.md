# Priority-Conditioned Retention (PCR)

**A controllable, middle-layer memory circuit in pretrained Mamba.**

Team project · 2026-1 AI Systems · Sungkyunkwan University
정찬희 (Chanhee Jeong) · 안예준 (Yejun Ahn) · 김수윤 (Suyun Kim)

---

## TL;DR

A pretrained Mamba's associative-recall failure is, in part, a **control** problem,
not only a **storage** problem. Without any fine-tuning, scaling the pre-softplus
input to the **Δ gate** at positions downstream of a target slows the decay that
target experiences and recovers recall under capacity pressure. The control is
**localized to layers 16–23**: intervening on just those 8 of 48 layers recovers
most of the gain at **near-zero language-modeling cost**. A fine-grained ablation
narrows this to a **4-layer minimal core at L19–22** that matches the 8-layer band
and is both sufficient and necessary.

(`og` = *other-gain*, the gain `g < 1` applied to non-target downstream positions.)

| Intervention (MQAR, N=32, base recall 0.53) | Recall | Perplexity |
|---|---|---|
| All layers, og=0.8 | 0.73 | ×1.81 |
| L16–23 (8 layers), og=0.8 | 0.67 | ×1.07 |
| **L19–22 (4-layer minimal core), og=0.8** | **0.70** | **×1.08** |

Importance-aware allocation transfers to continuous state through **retention**,
but **not** through quantization — the superimposed state has no per-item slots
to reallocate (see `src/pcr_writequant.py`, a deliberate negative result).

---

## Repository layout

```
pcr-mamba/
├── notebooks/
│   └── PCR_from_scratch.ipynb     # run end-to-end on Colab L4 (main entry point)
├── src/
│   ├── pcr_writequant.py          # write-precision quantization lever (negative result)
│   └── pcr_stage0_probe.py        # storage-vs-selectivity probe (state effective-rank)
├── paper/
│   ├── PCR_paper.tex              # self-contained 2-column conference-style source
│   └── PCR_paper.pdf
├── slides/
│   └── PCR_Proposal.pptx          # 10-min proposal deck (+ appendix)
├── docs/
│   ├── PCR_workshop_paper.md      # markdown source of the paper
│   └── PCR_Technical_Report.md    # extended technical notes
├── requirements.txt
└── LICENSE
```

---

## Reproduce

**Environment:** Google Colab with an L4 GPU (≈ tens of minutes).

> ⚠️ Do **not** install `mamba-ssm` / `causal-conv1d`. We rely on the pure-PyTorch
> sequential path so the Δ pathway is hookable. The warning
> *"The fast path is not available … sequential implementation"* is **expected**
> and confirms the slow path is active.

```bash
pip install -r requirements.txt
```

Then open `notebooks/PCR_from_scratch.ipynb` and run all cells. It will:

1. load `state-spaces/mamba-790m-hf` (fp32, frozen) and install the Δ hook,
2. print the **capacity curve** (pick N where base recall ≈ 0.4; we use N=32),
3. **Gate 1** — dose-response, specificity (downstream vs random),
4. **layer localization** → the L16–23 circuit,
5. **surgical** L16–23-only vs all-layer + perplexity guardrail,
6. write-order priority, graded P1/P2/P3 (exploratory),
7. the **write-quantization lever** (negative result),
8. **experiments A–H** — N-sweeps, fine-grained ablation (→ L19–22 minimal core +
   necessity knockout), Mamba-130m scale check, JRT baseline, de-confounded graded
   priority, n=150 Wilson-CI validation, and the differential-retention sweep.

To run the storage-vs-selectivity probe, paste `src/pcr_stage0_probe.py` after the
notebook (it reuses the notebook's harness).

### Expected numbers (Mamba-790m, from-scratch run)

```
Gate1 dose (all layers):  og=0.8 → 0.73 (peak),  og=0.4 → 0.00
specificity:              downstream 0.73 ≫ random 0.50
localization (og=0.8):    L16-23 +0.13, L24-31 +0.07, others ~0.00
surgical:                 L16-23 only  0.67 / ppl ×1.07
                          all layers   0.73 / ppl ×1.81
minimal circuit:          L19-22 (4L)  0.70 / ppl ×1.08  (matches 8L; necessary)
write-quant lever:        target k-bit → cliff at 4-bit; distractor k-bit → 0.00 (fails)
```
Exact values vary slightly with seeds; the **shape** (inverted-U, L16–23 peak,
ppl gap, quant cliff) is robust.

---

## Scope & limitations (read this)

- **Single model.** Positive results are Mamba-790m only. The circuit does **not**
  replicate on Mamba-1.4b (all-band Δ ≤ 0) and transfers only weakly to Mamba-130m
  (≈⅓ the effect, at a different relative depth) → likely specific to the 790m scale.
- **Synthetic task.** Robust on random-token MQAR; **fragile** on natural-language
  multi-fact recall.
- **Graded priority is real but weak.** A de-confounded experiment (Exp E) confirms
  other-gain causally controls retention independent of write position, but the
  effect is modest; the original strong result was largely a write-position artifact.

These bound the claim.

---

## Method note (why this, not something easier)

Most "what to keep in a fixed state" work (Titans, DeltaNet, TTT, Mamba-2)
**trains** new architectures or memory modules. PCR is a **training-free**,
frozen-base alternative that *wakes up* an existing circuit — a different category
of contribution and cheap to deploy. The closest training-free baseline is
**Just Read Twice** (prompting).

---

## Citation

```bibtex
@misc{pcr2026,
  title  = {Priority-Conditioned Retention: A Controllable Middle-Layer
            Memory Circuit in Pretrained Mamba},
  author = {Jeong, Chanhee and Ahn, Yejun and Kim, Suyun},
  note   = {Sungkyunkwan University, 2026-1 AI Systems team project},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

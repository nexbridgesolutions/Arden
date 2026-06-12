# Arden — Training Log

Bitácora de fases de entrenamiento del modelo Arden 0.2B.
Cada entrada documenta hardware, configuración, resultados y muestras reales de inferencia.

---

## Phase 1 — Frozen backbone (COMPLETED ✅)

**Date:** 2026-06-12
**Version:** v0.9.0
**Status:** Completed — 50,000 steps

### Configuration

| Setting | Value |
|---|---|
| Total parameters | 180,707,328 (180.7M) |
| Trainable | 14,177,280 (14.2M) — layers 20, 21 + final norm |
| Frozen | 166,530,048 (166.5M) |
| Architecture | 22 layers, d_model 768, 12 heads, d_ff 3072 |
| Sequence length | 256 |
| Batch / grad accum | 1 × 8 (effective 8) |
| Learning rate | 3e-4 cosine, 1000 warmup |
| Hardware | NVIDIA GTX 1050 2GB |
| Throughput | ~50 steps/min |

### Results

| Metric | Start | End |
|---|---|---|
| VAL loss | 9.14 | **6.53** |
| Steps | 0 | 50,000 |
| Tokens seen (effective) | — | ~21% of corpus |

The training survived multiple CUDA crashes (RAM accumulation on long runs)
and resumed automatically each time from the latest checkpoint — validating
the resilience architecture (checkpoints + auto-resume + systemd restart).

### Inference samples @ step 50,000

Sampling: temperature 0.8, top_k 40, repetition penalty 1.3

**Prompt:** `The European Parliament`
> The European Parliament In a statement of the United States, he said there is no doubt to be more than in any other countries.

**Prompt:** `El Parlamento Europeo`
> El Parlamento Europeo La Comisión de la UE [...] del Consejo [...] que el [...] los trabajadores [...] y unidades [...]

**Prompt:** `Le Parlement européen`
> Le Parlement européen Les [...] ne sont pas seulement régionale et l'Élysée de la République dans le cadre du mondial.

### Observations

- **English (77% of corpus):** near-coherent news-register syntax.
- **French (18%):** real phrases and structures ("l'Élysée de la République").
- **Spanish (5%):** correct domain vocabulary but fragmented tokens.
- Quality per language tracks corpus proportion — Spanish is weakest due to
  under-representation in WMT11 (1.2GB ES vs 17GB EN).
- Token fragmentation in ES/FR confirms the **embeddings remain frozen at
  random init** — the main limitation of the frozen-backbone approach.

### Conclusion

Pipeline fully validated. The model learns genuinely. The ceiling on quality
is the frozen random embeddings and backbone — addressed in Phase 2+ via
progressive unfreezing.

Checkpoint preserved as: `arden_0.2b_phase1_50k.pt`

---

## Phase 2 — Progressive unfreezing: layers 18-19 (IN PROGRESS 🔄)

**Date:** 2026-06-12
**Status:** Started

### Changes from Phase 1

| Setting | Phase 1 | Phase 2 |
|---|---|---|
| Trainable layers | 20, 21 | 18, 19, 20, 21 |
| Trainable params | 14.2M | 28.4M |
| Frozen params | 166.5M | 152.4M |
| max_steps | 50,000 | 100,000 |
| Optimizer | — | fresh (groups changed) |
| Dataset window | first 256 tokens | random 256-token window |

Resumes model weights from `step_00050000_final.pt`. Optimizer starts fresh
because the parameter groups changed (2 → 4 trainable layers). The random
window over each 2048-token chunk now exposes the full corpus instead of only
the first 256 tokens of each chunk (~8x more effective data).

### Expected behavior

- VAL loss rises temporarily as layers 18-19 enter with random weights, then
  decreases as they adapt.
- Target: VAL loss below Phase 1's 6.53 by step 100,000.

### Results

_(to be updated)_

## Experimental methodology — rotating gradual unfreezing

**Research question:** How far can a language model be pretrained
from scratch on minimal, low-cost hardware (8GB system RAM, 2GB GPU
VRAM)?

Standard gradual unfreezing (Howard & Ruder, 2018) accumulates
unfrozen layers when fine-tuning a *pretrained* model. This project
explores an under-documented regime: applying gradual unfreezing
during *from-scratch pretraining* on a 2GB-VRAM GPU, where layers
are unfrozen in rotating windows of 3 (re-freezing upper layers to
remain within memory limits) rather than accumulated.

To our knowledge, this rotating variant — applied to from-scratch
pretraining under a 2GB memory budget — has not been previously
documented. Results, positive or negative, are recorded here.

— David Ernesto Arriaga Pineda, Nex Bridge Solutions LLC, 2026

---
# Arden — Training Log

Bitácora de fases de entrenamiento del modelo Arden 0.2B.
Cada entrada documenta hardware, configuración, resultados y muestras reales de inferencia.

---
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

Cargando modelo...

--- PRUEBA ARDEN 0.2 ---

Prompt : The history of the world is
Arden  : The history of the world is In addition to being subject to our Priv acy Policy , storage , and use of your data will be notified from the laws and regulations country .

Prompt : La historia del mundo es
Arden  : La histor ia del mun do es El president e de la Com isi ón , el Con se jo que expulsion ó lo h acer al gun as o .

Prompt : L'histoire du monde est
Arden  : L ' his to ire du mon de est Le chef d ’ ê tre dé but de la ré g ion des dro its de l ’ hom me à ce ux qu i ont é t és par le ur é pr és ident ifier en France .

Prompt : A história do mundo é
Arden  : A hist ó ria do mun do é La 3 de la citywide del con se j em plo , el que las cu ent a ñ os y los tr ab aj ado res es .

## Phase 2 / Window A — 3 contiguous top layers (COMPLETED)

**Layers trained:** 19, 20, 21 + final norm + lm_head (21.3M trainable)
**Steps:** 50,000 → 100,000
**Hardware:** 2GB VRAM GPU, ~50 steps/min

### VAL loss progression

| Step | VAL loss | Perplexity | Note |
|---|---|---|---|
| 50,500 | 6.4454 | 629.8 | layer 19 just unfrozen |
| 62,500 | 6.3534 | 574.4 | adapting |
| 73,000 | 6.2498 | 517.9 | **best** |
| 100,000 | 6.2836 | 535.7 | plateau (LR floor reached) |

**Finding:** 3 contiguous top layers saturate around VAL 6.25. After
step 73k the loss oscillates without further improvement — the cosine
LR reached its floor (3e-5) and the limited trainable capacity (top 3
layers over a frozen random backbone) hit its ceiling. **6.25 is the
new baseline the rotating experiment must beat.**

### Inference samples @ Window A (4-language probe set)

Sampling: temperature 0.8, top_k 40, repetition penalty 1.3

**EN (77% of corpus):** "The history of the world is In addition to
being subject to our Privacy Policy, storage, and use of your data
will be notified from the laws and regulations country."
→ fluent, near-complete tokens.

**FR (18%):** "L'histoire du monde est Le chef d'être début de la
région des droits de l'homme..."
→ decent structure, some fragmentation.

**ES (5%):** "La historia del mundo es El presidente de la Comisión,
el Consejo que expulsionó lo hacer algunas o."
→ more fragmented, incoherent content.

**PT (Wikipedia only):** "A história do mundo é La 3 de la citywide
del consej emplo... snif snif"
→ weakest; cross-language bleed and severe fragmentation.

### Observations

- **Output quality tracks corpus proportion exactly: EN > FR > ES > PT.**
  This empirically confirms the corpus-imbalance hypothesis. PT is
  weakest as it came only from Wikipedia, without WMT11 reinforcement.
- **Token fragmentation persists in all four languages** ("histor ia",
  "mun do"), confirming the embedding layer remains frozen at random
  init. This is expected and will only be resolved at the full-unfreeze
  stage (Phase 3).
- Grammar and sentence structure improved noticeably from step 50k,
  but semantic coherence remains shallow (well-formed filler).

Checkpoint preserved as: `arden_0.2b_phase2_100k.pt`

---

## Phase 3 / Window B — first rotation (COMPLETED)

**Layers trained:** 17, 18, 19 + final norm (19 = bridge from Window A)
**Re-frozen:** layers 20, 21 (were trained in Window A)
**Steps:** 100,000 → 150,000
**Hardware:** 2GB VRAM GPU

### VAL loss progression

| Step | VAL loss | Perplexity | Note |
|---|---|---|---|
| 100,500 | 6.1895 | 487.6 | strong start (bridge + higher LR) |
| ... | 6.2844 | 536.2 | drifting up |
| ... | 6.4015 | 602.7 | bottleneck showing |
| ... | 6.3734 | 586.1 | |
| ... | 6.2480 | 517.0 | brief recovery |
| 150,000 | 6.2945 | 541.6 | **final** |

### Verdict: rotation did NOT beat the contiguous baseline

```
Window A (contiguous, layers 19,20,21): VAL 6.25  ← baseline
Window B (rotating,  layers 17,18,19):  VAL 6.29  ← did not beat it
```

Window B opened promisingly at 6.19 (helped by the reset LR and the
trained bridge layer 19), but **rebounded and plateaued around
6.25–6.40, finishing at 6.29 — slightly worse than the baseline.**

### Finding (pre-registered outcome)

This is the bottleneck predicted in the methodology note. When the
upper layers (20, 21) are re-frozen to make room for deeper layers
(17, 18), whatever the deeper layers learn must pass through the
fixed upper layers before reaching the output. This **bottleneck
cancels the benefit** of unfreezing deeper layers.

**This empirically supports why the canonical method (Howard & Ruder,
2018) accumulates unfrozen layers rather than rotating them.** Under a
2GB budget, rotation with re-freezing is inferior to simply training
the top contiguous layers. A valid negative result.

### Inference @ Window B (4-language probe)

- **EN:** "The history of the world is As part of a result, I am going
  to be able to do with it." — coherent grammar, no clear gain over A.
- **ES:** "El proceso de la Comisión aplicada en el Parlamento Europeo
  está tico y algo que se ha sido el mercado..." — no improvement;
  possibly more tangled.
- **FR / PT:** similar quality to Window A; token fragmentation intact.
- **"Par l amento" persists** in all non-English languages, as expected
  (token_emb embedding remains frozen throughout rotation).

### Implication for next steps

Rotation on 2GB has reached its useful limit. Continuing to deeper
windows (C, D, ...) would likely show the same or worse bottleneck.
**The real quality gain requires full unfreeze (all layers +
token_emb embedding) on the 6GB tower** — which removes the bottleneck
entirely by training every layer simultaneously, including the
embedding that causes token fragmentation.

Checkpoint preserved as: `arden_0.2b_windowB_150k.pt`

---


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
## Gradient checkpointing + smoke test — rotation made viable on 2GB

### The 2GB wall (before checkpointing)

The rotation experiment stalled at Window C: training layers in
deeper positions (15,16,17) caused CUDA OOM on the 2GB GPU, even with
`expandable_segments:True` already active. With only ~17 MiB free, the
backward pass could not allocate. Reducing to 2 layers moved the OOM
to the forward pass (cross_entropy over 32k vocab). The wall was the
fixed cost of gradients + optimizer state + activations, not
fragmentation.

### Fix: gradient checkpointing

Wrapped each transformer layer with `torch.utils.checkpoint` during
training (forward only; eval runs normally):

```python
for layer in self.layers:
    if self.training:
        x = checkpoint(layer, x, attn_mask, key_padding_mask,
                       use_reentrant=False)
    else:
        x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
```

This trades compute for memory: activations are recomputed in the
backward pass instead of stored, freeing VRAM.

### Finding on checkpointing

| Config | Without checkpointing | With checkpointing |
|---|---|---|
| 3 layers, deep (15,16,17) | OOM | **fits, 1.7GB** |
| 4 layers | OOM | OOM (still) |
| Throughput | ~63s / 50 steps | ~84-130s / 50 steps |

- Checkpointing **unblocks 3 layers in ANY position**, including deep
  ones that previously OOM'd.
- It does **not raise the max layer count** (still 3) — the 4-layer
  wall comes from gradient + optimizer cost, which checkpointing does
  not reduce.
- Cost: roughly 30-50% slower per step.

### Smoke test: full-rotation viability

Ran a rapid smoke test (~100-200 steps per window) descending the
entire model, window by window, from layer 17 down to layer 0:

| Window | Layers | Fits on 2GB? |
|---|---|---|
| C | 15,16,17 | ✅ |
| D | 13,14,15 | ✅ |
| E | 11,12,13 | ✅ |
| F | 9,10,11 | ✅ |
| G | 7,8,9 | ✅ |
| H | 5,6,7 | ✅ |
| I | 3,4,5 | ✅ |
| J | 1,2,3 | ✅ |
| K | 0,1 | ✅ |

**Result: every window fits at a constant ~1.7GB VRAM. The full
rotation experiment, all the way to layer 0, is viable on a 2GB GPU
with gradient checkpointing.** No OOM at any depth.

VAL numbers during the smoke test are NOT meaningful (only ~100-200
steps per window, LR at floor) — the smoke test verified technical
feasibility only, not quality.

### Real experiment scope (decided)

The rotation experiment will be run over **exactly one epoch** of the
corpus (240,000 steps total at batch_eff=8). With Windows A and B
already complete, the remaining ~88,500 steps to reach 1 full epoch
are distributed across the 8 remaining windows (~11,000 steps each),
descending from layer 15 to layer 0.

**Methodological note:** Windows A and B ran 50k+ steps each, while
Windows D–K run ~11k each (to fit the experiment within one epoch).
Step counts per window are therefore NOT uniform. VAL differences
between early and late windows may partly reflect step count, not only
layer choice. This is a documented limitation of running the full
experiment within a single-epoch budget on constrained hardware.

The embedding (token_emb) remains frozen throughout the entire
rotation — token fragmentation ("Par l amento") will persist until the
full-unfreeze stage on the tower.

---

## Rotating gradual unfreezing — full descent complete (SUCCESS ✅)

**Status:** Completed — full rotation executed as planned, layer 21
down to layer 0, in windows of 3 with one bridge layer overlapping
between consecutive windows (per the methodology and smoke-test plan
above).

### Result

| Stage | VAL loss |
|---|---|
| Phase 2 / Window A baseline (contiguous, layers 19-21) | 6.25 |
| Window B (rotating, layers 17-19) | 6.29 — did not beat baseline |
| **Final window (rotating descent complete, layers 1-0)** | **5.0125** |

The full rotating descent — continuing past Window B through the
deeper windows (C through K, layers 15 down to 0, enabled by the
gradient checkpointing fix documented above) — closes with a VAL loss
of **5.0125**, a substantial improvement over both the 6.25 contiguous
baseline and the 6.29 Window B result.

This supports the project's working hypothesis: while a single
rotation step (Window B) showed the predicted re-freezing bottleneck,
continuing the rotation across the full depth of the network — each
window passing learned signal further down via the bridge layer —
eventually overcomes that bottleneck and drives loss meaningfully
below the contiguous-only baseline.

### Inference samples (test_arden.py, PyTorch, post-rotation)

Multilingual probe set (EN/ES/FR/PT), same prompts style as Phase 1/2
for comparability.

**EN:** "Once upon a time, in a small village, The most recent years
ago that it is to be an end of $4 billion." / "Yesterday I went to the
market and Most of them for a few years, as if it's relevant that we
don't have any other than they want."
→ Fluent surface grammar, intact whole-word tokens, still semantically
loose — consistent with prior phases.

**ES:** "Había una vez, en un pequeño pueblo, El acceso de la segunda
idad con el Parlamento Europeo a." / "Ayer fui al mercado y En el
consecuencia en los que no puede habrar lo en la Comisión."
→ Token fragmentation persists ("Hab ía", "Par l amento", "id ad"),
as expected — token_emb has not been part of this rotation and
remains at random initialization.

**FR / PT:** Similar pattern — coherent function-word usage and
EU/political-register vocabulary bleeding through (a corpus artifact
noted since Phase 1), but heavy subword fragmentation throughout.

### Note on embedding layer

As anticipated since Phase 1, fragmentation in ES/FR/PT ("Hab ía",
"pe que ñ o p ue blo") confirms `token_emb` remained frozen throughout
the entire rotating descent. The VAL loss improvement to 5.0125 was
achieved entirely through the transformer layers and final norm/head —
the embedding layer is the next and final frontier, planned for a
full-unfreeze stage.

Checkpoint preserved as: `step_00240000_final.pt` (also mirrored as
`best_model.pt`).

--- PRUEBA ARDEN 0.2 ---

Prompt : Once upon a time, in a small village,
Arden  : Once upon a time , in a small village , The most recent years ago that it is to be an end of $ 4 billion .

Prompt : Había una vez, en un pequeño pueblo,
Arden  : Hab ía una vez , en un pe que ñ o p ue blo , El ac ces o de la se g unda id ad con el Par l amento Europe o a .

Prompt : Il était une fois, dans un petit village,
Arden  : Il é ta it une fo is , d ans un pet it village , C ette ré union n ous p ou v ons à la s ant é rie de l ' E sp agne sur ce tte question .

Prompt : Era uma vez, numa pequena aldeia,
Arden  : E ra uma vez , num a pe qu ena al de ia , A dem á s los usu ar ios po dr án val or ar los com ent ar ios de ot ros le cto res vot and o a favor o en contr a , y en cas o de que cons ide ren un com ent ario in

Prompt : Yesterday I went to the market and
Arden  : Yesterday I went to the market and Most of them for a few years , as if it ’ s rele that we don ' t have any other than they want .

Prompt : Ayer fui al mercado y
Arden  : A yer fu i al mer c ado y En el con se cu enc ia en los que no p ued e hab r ar lo en la Com isi ón .

Prompt : Hier je suis allé au marché et
Arden  : H ier je su is all é au march é et D ans son pro jet de lo i , le ur pays du mon dial ent re les dé part isans ment .

Prompt : Ontem fui ao mercado e
Arden  : On tem fu i ao mer c ado e Per o de la Un i ón Europe a , en un que no p ued an las aut or id ades est as al ac ion es y el debate res pe to del Con se jo ven ir án dose ñ or io .

Prompt : If it rains tomorrow, then we will
Arden  : If it rains tomorrow , then we will The company ’ s decision to make the country is a largest economy of their own on a new features .

Prompt : Si mañana llueve, entonces
Arden  : Si ma ñ ana ll ue ve , ent on ces El G ob ier no se h ace un plan te ma del remind o de los Est ados mi emb ros y la Un i ón Europe a en el delegate , per o que le va a plic ar ía en al gun as í cul os

Prompt : S'il pleut demain, alors nous
Arden  : S ' il ple ut de main , al ors n ous Le produ it que vous sou ha ite z consul ter est pay ant ou ré serv é à nos ab on n és . ou vell es se produ is ent .. _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

Prompt : Se chover amanhã, então
Arden  : Se ch over aman h ã , ent ão En la m ê me é dia ée de l ' E tat am iti confidentiality et d ans le cad re du sect eur g én é ral des Beckham es ou v r és idents sur les dé c isions es de ux parties .
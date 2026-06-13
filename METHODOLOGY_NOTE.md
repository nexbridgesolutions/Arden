# Arden — Experimental Methodology Note

**To be appended to TRAINING_LOG.md**

---

## Experimental methodology — rotating gradual unfreezing

**Research question:** How far can a decoder-only language model be
pretrained *from scratch* on minimal, low-cost hardware (≈8GB system
RAM, 2GB GPU VRAM)?

### Background

Gradual unfreezing (Howard & Ruder, 2018, *ULMFiT*) is an established
technique in which a model's layers are unfrozen progressively, from
the output layer toward the input layer, to retain learned features
and avoid catastrophic forgetting. In its original and common form it
is applied to **fine-tuning a model that is already pretrained**, and
unfrozen layers are **accumulated** (once a layer is unfrozen it stays
trainable as deeper layers join).

### What this project explores

Arden applies the same top-to-bottom unfreezing direction, but in a
regime we have not found documented:

1. **From-scratch pretraining**, not fine-tuning. Layers begin at
   random initialization rather than from pretrained weights.
2. **Severe memory constraint** (2GB VRAM), which makes accumulating
   unfrozen layers infeasible beyond ~3 layers.
3. **Rotating windows** instead of accumulation: layers are unfrozen
   in overlapping groups of three, re-freezing the upper layers to
   stay within the memory budget. One layer is shared between
   consecutive windows as a "bridge" to carry learned signal downward.

```
Window A: layers 21, 20, 19
Window B: layers 19, 18, 17   (19 is the bridge)
Window C: layers 17, 16, 15   (17 is the bridge)
Window D: layers 15, 14, 13   (15 is the bridge)
...
```

### Honest framing

This is an **experiment in progress**, not a validated result. The
rotating variant has a known theoretical trade-off versus the
canonical accumulating method: when upper layers are re-frozen, the
signal learned by deeper layers must pass through fixed upper layers,
which may bottleneck learning. Whether the "bridge" overlap mitigates
this enough to be useful on minimal hardware is exactly the open
question being tested.

Results — **positive or negative** — will be recorded in this log.
A negative result (the rotating variant underperforming the simpler
3-contiguous-layer approach) is a valid and useful outcome.

### To our knowledge

We have not found prior documentation of this specific combination —
gradual unfreezing applied to *from-scratch* pretraining under a 2GB
memory budget with *rotating, re-freezing* windows. If prior work
exists, this note should be read as an independent reproduction rather
than a first.

— David Ernesto Arriaga Pineda
Nex Bridge Solutions LLC, 2026

---

### Baseline to beat

| Approach | Layers | VAL loss | Notes |
|---|---|---|---|
| Phase 1 (contiguous) | 20, 21 | 6.53 | 2 top layers |
| Phase 2 (contiguous) | 19, 20, 21 | 6.44 | 3 top layers — current best |
| Phase 3 (rotating) | rotating ×3 | _TBD_ | this experiment |

The rotating approach is considered successful **only if** it drives
VAL loss meaningfully below the 6.44 achieved by simply training the
3 top contiguous layers. Otherwise, the simpler contiguous approach
wins and that is documented as the finding.

## Rotating gradual unfreezing — experiment tracking

Each window trains for 50,000 steps on a 2GB-VRAM GPU. Windows of 3
layers overlap by one "bridge" layer, which stays trainable across two
consecutive windows to carry learned signal downward. The embedding
layer is the final frontier and is only reached under full unfreeze
(6GB GPU).

**Pre-registered success criterion:** the rotating approach is
considered useful only if VAL loss drops meaningfully below the
baseline of **6.44** (achieved by simply training the 3 top contiguous
layers, no rotation). A negative result is a valid, documented finding.

### Baseline (contiguous, no rotation)

| Phase | Trainable layers | Steps | VAL loss |
|---|---|---|---|
| Phase 1 | 20, 21 | 50,000 | 6.53 |
| Phase 2 | 19, 20, 21 | 50,000 | **6.44** |

### Rotating windows

| Window | Trainable layers | Bridge | Steps (+50k each) | VAL loss | Δ vs baseline |
|---|---|---|---|---|---|
| A | 21, 20, 19 | — | 50,000 | 6.44 | baseline |
| B | 19, 18, 17 | 19 | 100,000 | _pending_ | _–_ |
| C | 17, 16, 15 | 17 | 150,000 | _pending_ | _–_ |
| D | 15, 14, 13 | 15 | 200,000 | _pending_ | _–_ |
| E | 13, 12, 11 | 13 | 250,000 | _pending_ | _–_ |
| F | 11, 10, 9 | 11 | 300,000 | _pending_ | _–_ |
| G | 9, 8, 7 | 9 | 350,000 | _pending_ | _–_ |
| H | 7, 6, 5 | 7 | 400,000 | _pending_ | _–_ |
| I | 5, 4, 3 | 5 | 450,000 | _pending_ | _–_ |
| J | 3, 2, 1 | 3 | 500,000 | _pending_ | _–_ |
| K | 1, 0 | 1 | 550,000 | _pending_ | _–_ |

### Final stage (full unfreeze, 6GB GPU)

| Stage | Trainable | VAL loss | Notes |
|---|---|---|---|
| Full unfreeze | all 22 layers + embedding | _pending_ | embedding finally trained |

### How to read this table

- **VAL loss falling window after window** → the rotating variant is
  learning as it descends; the bridge overlap is doing its job.
- **VAL loss stalling or rising as windows go deeper** → the
  re-freezing of upper layers bottlenecks deep-layer learning, as
  predicted by theory. Still a valid finding.
- The embedding layer is never trained during rotation (it sits below
  layer 0 and does not fit alongside transformer layers in 2GB). Token
  fragmentation (e.g. "Par l amento" instead of "Parlamento") is
  expected to persist until the full-unfreeze stage.

### Notes per window

_(record observations here as each window completes — inference
samples, anomalies, crashes, throughput)_

- **Window A (21,20,19):** in progress. First eval VAL 6.4454 at step
  50,500. Throughput ~50 steps/min on 2GB GPU. VRAM ~1.7GB.
  ### Hypothesis (to be tested)

We hypothesize that warming the transformer layers via rotating
unfreezing reduces the steps required for the full-unfreeze stage
to reach a given VAL loss, compared to full unfreeze from random
initialization.

The magnitude of this saving (if any) can only be measured by
running both conditions:
  (1) full unfreeze from scratch  [control]
  (2) full unfreeze after rotating warm-up  [experiment]

Until both are measured, no percentage is claimed.
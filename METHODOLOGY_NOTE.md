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
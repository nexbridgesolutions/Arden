<div align="center">

# ARDEN AI

### A multilingual language model built from scratch — ES / EN / PT / FR

**English** | [Español](#)

[![License](https://img.shields.io/badge/License-Arden%20Community%20License%20v1.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org)
[![Parameters](https://img.shields.io/badge/Parameters-180M-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Training-yellow.svg)]()

</div>

---

The Arden project aims to pretrain a family of multilingual language models (Spanish, English, Portuguese, French) **built entirely from scratch in PyTorch** — no pretrained weights, no model forks. The current model, **ARDEN 0.2B (180.7M parameters)**, is being pretrained on consumer hardware: a single NVIDIA GTX 1050 with 2GB of VRAM. 🚀

Yes, you read that right — 2GB. Arden is proof that you don't need a datacenter to start building language models. The project scales progressively: each phase trains a larger model on better hardware, reusing the same codebase, tokenizer, and corpus pipeline.

Arden is developed and owned by **[Nex Bridge Solutions LLC](https://nexbridgesolutions.com)** (Arizona, USA), created by **David Ernesto Arriaga Pineda**.

## News

- **2026-06**: Arden 0.2B v0.9.0 — GPU training pipeline stabilized: automatic resume from checkpoint, offset-indexed dataset loader (23GB corpus on 1.5GB RAM), eval loop fixes.
- **2026-06**: Migrated training from CPU (Xeon, Dell PowerEdge T110 II) to GPU (GTX 1050). ~200x speedup.
- **2026-06**: Full corpus processed — 2.2M chunks from WMT11 (ES/EN/FR) + Wikipedia (ES/EN/PT/FR) + identity dataset, tokenized with a custom 32k BPE tokenizer.
- **2026-06**: Initial release of the Arden architecture, tokenizer, data pipeline and training loop. Licensed under the Arden Community License v1.0.

## Releases Schedule

We will roll out intermediate checkpoints as training progresses, in the spirit of open development.

**Base models:**

| Phase | Checkpoint | Params | Hardware | Status |
|---|---|---|---|---|
| 1 | arden-0.2b-step-50k | 180.7M | GTX 1050 2GB (frozen backbone) | ✅ training |
| 2 | arden-0.2b-progressive | 180.7M | GTX 1050 2GB (gradual unfreezing) | 🔄 IN training |
| 3 | arden-0.2b-full | 180.7M | GTX 1660 Super 6GB (full training) | 📋 Planned |
| 4 | arden-0.5b | ~514M | GTX 1660 Super 6GB | 📋 Planned |
| 5 | arden-1.1b | ~1.18B | 12GB+ VRAM GPU | 📋 Planned |

**Chat models:**

| Checkpoint | Base | Finetuning | Status |
|---|---|---|---|
| arden-0.2b-chat-v0.1 | arden-0.2b-full | Identity + conversational dataset | 📋 Planned |

## Potential Use Cases

Small multilingual models are useful for many applications:

- **Local, private conversational AI** in Spanish, English, Portuguese and French — no internet connection, no data leaving your machine.
- **Deployment on modest hardware** — a 4-bit quantized Arden 0.2B weighs roughly 120MB.
- **A reference codebase** for anyone who wants to pretrain a language model from scratch in pure PyTorch — without diving into Megatron-LM, FSDP or multi-node complexity. Every component (attention, tokenizer, training loop, data pipeline) is readable in a single afternoon.
- **A foundation for Latin American AI** — Arden treats Spanish and Portuguese as first-class languages, not afterthoughts.

## Training Details

| Setting | Description |
|---|---|
| Parameters | 180,707,328 (180.7M) |
| Architecture | Decoder-only Transformer (GPT-style), pre-LayerNorm, GELU |
| Model Size | Layers: 22, Heads: 12, Embedding Size: 768, Intermediate Size: 3072 |
| Tied Embeddings | Yes |
| Sequence Length | 256 |
| Tokenizer | Custom BPE, 32,000 vocab, trained on the multilingual corpus |
| Batch Size | 1 × 8 gradient accumulation (effective batch 8) |
| Learning Rate | 3e-4, cosine decay with 1,000 warmup steps |
| Training Data | WMT11 monolingual (Europarl + News Crawl + News Commentary, ES/EN/FR) + Wikipedia (ES/EN/PT/FR) + identity dataset |
| Combined Dataset Size | ~23GB tokenized — 2.2M chunks of 2,048 tokens |
| Precision | float32 (AMP-ready) |
| Hardware | 1× NVIDIA GTX 1050 2GB 😄 |
| Checkpointing | Time-based (every 4h) + best-model on every eval, automatic resume |

### Training strategy: progressive unfreezing

Because the full 180M model does not fit in 2GB of VRAM for training, Arden 0.2B uses a staged approach:

1. **Phase 1 (current):** last 2 transformer layers + final norm trainable (14.2M params), rest frozen.
2. **Phase 2:** progressively unfreeze additional layers on the GTX 1050.
3. **Phase 3:** full-parameter training on a GTX 1660 Super 6GB.

The codebase supports CPU-only training, single-GPU CUDA training with AMP, offset-indexed streaming datasets (train on corpora larger than RAM), and automatic resume from the most recent checkpoint.

## Project Structure

```
arden/
├── core/
│   ├── attention.py      # Multi-head self-attention
│   ├── config.py         # Master configuration (model / training / data)
│   ├── model.py          # ArdenModel — decoder-only Transformer
│   └── tokenizer.py      # BPE tokenizer ES/EN/PT/FR
├── data/
│   ├── dataset_loader.py # WMT11 + Wikipedia corpus preparation
│   ├── preprocessor.py   # Tokenization & train/val/test splits
│   └── identity_dataset.jsonl
├── train.py              # Training loop (resume, AMP, freeze control)
├── test_arden.py         # Quick inference sanity check
├── arden_train_runner.sh # systemd runner
├── arden-train.service   # systemd unit
└── LICENSE               # Arden Community License v1.0
```

## Pretrain

```bash
# 1. Environment
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers datasets psutil

# 2. Prepare the corpus
python3 data/dataset_loader.py
python3 data/preprocessor.py

# 3. Train (resumes automatically from the latest checkpoint)
python3 train.py
```

To run as a service that survives reboots:

```bash
sudo cp arden-train.service /etc/systemd/system/
sudo systemctl enable --now arden-train
```

## License

Arden is released under the **Arden Community License v1.0**:

- ✅ Free for personal use, research, and education
- ✅ Free for internal evaluation
- ❌ Hosting Arden as a service requires a commercial agreement
- ❌ Embedding Arden in commercial products requires a commercial agreement

For commercial licensing: **legal@nexbridgesolutions.com**

## TODO

This project is under active development by a very small team. Community feedback is highly appreciated.

- [x] Architecture, tokenizer and data pipeline from scratch
- [x] CPU and single-GPU CUDA training with automatic resume
- [x] Multilingual corpus (WMT11 + Wikipedia, 4 languages)
- [x] Complete Phase 1 training (50k steps)
- [ ] Progressive layer unfreezing (Phase 2)
- [ ] Full-parameter training on 6GB GPU (Phase 3)
- [ ] Identity & conversational finetuning
- [ ] GGUF conversion for Ollama / llama.cpp
- [ ] Hugging Face publication
- [ ] Evaluation on multilingual benchmarks
- [ ] Arden 0.5B

## Frequently Asked Questions

#### 1. Why train such a small model on such modest hardware?

Because the alternative was not training at all. Arden is built in Central America, self-funded, with no institutional backing — and we believe the barrier to entry for language model research should be a $100 used GPU, not a GPU cluster. Every optimization in this codebase (offset-indexed datasets, staged unfreezing, time-based checkpointing for unreliable power) exists because it had to.

#### 2. Is Arden competitive with TinyLlama / Qwen / Gemma?

Not yet, and we won't pretend otherwise. Those models were trained on trillions of tokens with hundreds of GPUs. Arden's value is different: a fully owned, fully transparent, multilingual-first model with a readable codebase, growing in public — checkpoint by checkpoint.

#### 3. Why ES/EN/PT/FR?

Over 900 million people speak Spanish, Portuguese or French as a first language, yet most small open models treat them as secondary. Arden was designed multilingual from the tokenizer up.

## About

**Arden** is created by **David Ernesto Arriaga Pineda**, founder of **Nex Bridge Solutions LLC** (Arizona, USA).

- 🌐 [nexbridgesolutions.com](https://nexbridgesolutions.com)
- 📧 legal@nexbridgesolutions.com

## Citation

If you find this work valuable:

```bibtex
@misc{arriaga2026arden,
  title  = {Arden: A Multilingual Language Model Built From Scratch on Consumer Hardware},
  author = {Arriaga Pineda, David Ernesto},
  year   = {2026},
  url    = {https://github.com/nexbridgesolutions/Arden}
}
```

---

<div align="center">

**Copyright 2026 David Ernesto Arriaga Pineda — Nex Bridge Solutions LLC**

*"Building AI from scratch, one token at a time."*

</div>
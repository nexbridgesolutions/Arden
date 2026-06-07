<div align="center">

# 🤖 Arden 1.0

### General Purpose Bilingual LLM — ES / EN / PT / FR

**Built from scratch with PyTorch by Nex Bridge Solutions LLC**

[![License](https://img.shields.io/badge/License-Arden%20Community%20License%20v1.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org)
[![Parameters](https://img.shields.io/badge/Parameters-1.1B-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Training-yellow.svg)]()

</div>

---

## 📌 Overview

**Arden** is a 1.1 billion parameter decoder-only Transformer language model built entirely from scratch using PyTorch — no pre-trained weights, no model forks. Arden is designed for general-purpose bilingual inference in Spanish, English, Portuguese, and French.

Arden is the foundational AI model of **Nex Bridge Solutions LLC**, developed and owned by **David Ernesto Arriaga Pineda**.

---

## ✨ Key Features

- **1.1B parameters** — competitive scale with TinyLlama and similar open models
- **4 languages** — Spanish, English, Portuguese, French (ES/EN/PT/FR)
- **Built from scratch** — custom Transformer architecture in pure PyTorch
- **Custom BPE tokenizer** — 32,000 token vocabulary trained on multilingual corpus
- **CPU-first design** — trains on CPU, scales to GPU when available
- **Checkpoint every 4h** — automatic resume from last checkpoint
- **Freeze backbone** — efficient CPU training with only last layers active
- **Ollama compatible** — designed for local deployment via Ollama

---

## 🧠 Architecture
Model Type     : Decoder-only Transformer (GPT-style)
Parameters     : 1,177,616,384 (~1.1B)
Layers         : 22
Attention heads: 16
d_model        : 2,048
d_ff           : 8,192
d_head         : 128
Max context    : 2,048 tokens
Vocabulary     : 32,000 (custom BPE)
Activation     : GELU
Normalization  : Pre-LayerNorm
Embeddings     : Tied input/output

---

## 🗂️ Project Structure
arden/
├── core/
│   ├── attention.py      # Multi-head self-attention
│   ├── config.py         # Full model & training configuration
│   ├── model.py          # ArdenModel — main Transformer
│   └── tokenizer.py      # BPE tokenizer ES/EN/PT/FR
├── data/
│   ├── dataset_loader.py # Wikipedia multilingual downloader
│   ├── preprocessor.py   # Tokenization & train/val/test splits
│   └── processed/        # Preprocessed JSONL chunks
├── train.py              # Main training loop
├── LICENSE               # Arden Community License v1.0
└── README.md

---

## 🚀 Quick Start

### Requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install datasets tokenizers psutil
```

### Download & Preprocess Data

```bash
python3 data/dataset_loader.py
python3 data/preprocessor.py
```

### Train

```bash
python3 train.py
```

### Resume from checkpoint

In `core/config.py` set:
```python
resume_from: Optional[str] = "/opt/arden/checkpoints/step_XXXXXXXX.pt"
```

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Corpus | Wikipedia ES + EN + PT + FR |
| Batch size | 1 |
| Grad accumulation | 8 |
| Effective batch | 8 |
| Learning rate | 3e-4 (cosine decay) |
| Warmup steps | 2,000 |
| Max steps | 50,000 |
| Checkpoint | Every 4 hours |
| Hardware | CPU-only (scales to GPU) |
| Framework | PyTorch 2.5+ |

---

## 📜 License

Arden is released under the **Arden Community License v1.0**.

- ✅ Free for personal use, research, and education
- ✅ Free for internal commercial use
- ❌ Hosting as a service requires a commercial agreement
- ❌ Building commercial products requires a commercial agreement

For commercial licensing: **legal@nexbridgesolutions.com**

---

## 🏢 About

**Arden** is developed and owned by **David Ernesto Arriaga Pineda**, founder of **Nex Bridge Solutions LLC** (Arizona, USA).

- 🌐 [nexbridgesolution.com](https://nexbridgesolution.com)
- 📧 legal@nexbridgesolutions.com
- 🏢 Nex Bridge Solutions LLC — Arizona, USA

---

## 🗺️ Roadmap

- [x] Architecture design — 1.1B Transformer
- [x] Custom BPE tokenizer ES/EN/PT/FR
- [x] Data pipeline — Wikipedia 4 languages
- [x] Training loop with checkpoint & resume
- [x] GitHub repository
- [ ] First checkpoint
- [ ] Full Wikipedia corpus (6M+ articles)
- [ ] GPU training migration
- [ ] GGUF conversion for Ollama
- [ ] Hugging Face publication
- [ ] Arden 2.0 — expanded corpus & fine-tuning

---

<div align="center">

**Copyright 2026 David Ernesto Arriaga Pineda — Nex Bridge Solutions LLC**

*"Building AI from scratch, one token at a time."*

</div>

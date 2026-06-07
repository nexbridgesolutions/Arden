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

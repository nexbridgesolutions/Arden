"""
ARDEN 0.2B — Query CLI (single-shot)
Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

Single-prompt text generation against ARDEN, via command line.

Usage:
    python3 command.py "Once upon a time, in a small village,"

Unlike ARDENWARD (which CLASSIFIES security events), ARDEN is a
general-purpose LLM that CONTINUES text. This script takes the prompt,
tokenizes it, and generates the continuation token by token.

Loads the model on each invocation — it is not a persistent service.
For real volume, a long-running service is preferable (see roadmap).
"""

import sys
import argparse
import torch
sys.path.insert(0, '/opt/arden')
from core.config import ArdenConfig
from core.model import ArdenModel
from core.tokenizer import ArdenTokenizer
from pathlib import Path


def generate(model, tokenizer, prompt: str,
             max_new_tokens: int = 60,
             temperature: float = 0.8,
             top_k: int = 40,
             repetition_penalty: float = 1.3) -> str:
    """Generates the continuation of a prompt using top-k sampling +
    repetition penalty (the same settings that worked in test_arden.py)."""

    ids = tokenizer.encode(prompt, add_special_tokens=True)
    x   = torch.tensor([ids], dtype=torch.long)

    eos_id = getattr(tokenizer, "eos_id", None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out    = model(input_ids=x)
            logits = out["lm_logits"][:, -1, :]

            # Repetition penalty — discourages already-used tokens
            for tok in set(x[0].tolist()):
                if logits[0, tok] > 0:
                    logits[0, tok] /= repetition_penalty
                else:
                    logits[0, tok] *= repetition_penalty

            # Temperatura
            logits = logits / temperature

            # Top-k — limita el muestreo a los k tokens más probables
            if top_k > 0:
                kth = torch.topk(logits, top_k)[0][:, -1:]
                logits = logits.masked_fill(logits < kth, float('-inf'))

            probs   = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

    return tokenizer.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser(
        description="ARDEN 0.2B — generación single-shot por CLI")
    parser.add_argument("prompt", nargs="+",
                        help="Texto inicial a continuar")
    parser.add_argument("--tokens", type=int, default=60,
                        help="Máximo de tokens nuevos (default 60)")
    parser.add_argument("--temp", type=float, default=0.8,
                        help="Temperatura de sampling (default 0.8)")
    parser.add_argument("--topk", type=int, default=40,
                        help="Top-k para el sampling (default 40)")
    parser.add_argument("--reppen", type=float, default=1.3,
                        help="Penalización por repetición (default 1.3)")
    parser.add_argument("--ckpt", type=str,
                        default="/opt/arden/checkpoints/best_model.pt",
                        help="Ruta al checkpoint")
    args = parser.parse_args()

    prompt = " ".join(args.prompt)

    cfg       = ArdenConfig()
    vocab_dir = Path('/opt/arden/data/vocab')
    ckpt_path = Path(args.ckpt)

    print("Cargando tokenizer...")
    tokenizer = ArdenTokenizer.load(vocab_dir)

    print("Cargando modelo...")
    model = ArdenModel(cfg.model)
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f"Parámetros: {model.num_parameters():,}\n")
    print(f"Prompt : {prompt}")

    salida = generate(
        model, tokenizer, prompt,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.topk,
        repetition_penalty=args.reppen,
    )

    print(f"Arden  : {salida}")


if __name__ == "__main__":
    main()
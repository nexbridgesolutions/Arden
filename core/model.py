"""
ARDEN 1.0 — General Purpose Bilingual LLM ES/EN
Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

Decoder-only Transformer ~0.2B parameters.
CPU-only (torch.float32). No external dependencies beyond PyTorch.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.attention import MultiHeadAttention
from core.config import ModelConfig


class FeedForward(nn.Module):
    _ACTIVATIONS = {
        "gelu" : nn.GELU(),
        "relu" : nn.ReLU(),
        "swish": nn.SiLU(),
    }

    def __init__(self, config: ModelConfig):
        super().__init__()
        act = self._ACTIVATIONS.get(config.activation)
        if act is None:
            raise ValueError(f"Unknown activation: {config.activation}")
        self.fc1  = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias_in_ffn)
        self.act  = act
        self.fc2  = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias_in_ffn)
        self.drop = nn.Dropout(config.ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pre_norm = config.use_pre_norm
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.attn  = MultiHeadAttention(
            d_model  = config.d_model,
            n_heads  = config.n_heads,
            dropout  = config.attention_dropout,
            use_bias = config.use_bias_in_attn,
        )
        self.ffn  = FeedForward(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.drop(self.attn(self.norm1(x), attn_mask, key_padding_mask))
            x = x + self.drop(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop(self.attn(x, attn_mask, key_padding_mask)))
            x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class ArdenModel(nn.Module):
    """
    ARDEN 1.0 — Decoder-only Transformer ~110M params.
    Single objective: Causal Language Modeling (next token prediction).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop  = nn.Dropout(config.dropout)

        # Transformer stack
        self.layers     = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.n_layers)]
        )
        self.norm_final = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # Weight init
        self.apply(self._init_weights)
        scale = (2.0 * config.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "fc2.weight")):
                nn.init.normal_(p, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : Optional[torch.Tensor] = None,
        labels         : Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        return_hidden  : bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        attn_mask        = self._causal_mask(T, device) if use_causal_mask else None
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.eq(0)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        hidden    = self.norm_final(x)
        lm_logits = self.lm_head(hidden)

        out: Dict[str, torch.Tensor] = {"lm_logits": lm_logits}

        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        if return_hidden:
            out["hidden_states"] = hidden

        return out

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids         : torch.Tensor,
        max_new_tokens    : int   = 512,
        temperature       : float = 0.7,
        top_k             : int   = 50,
        top_p             : float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id      : Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            ctx    = generated[:, -self.config.max_seq_len:]
            out    = self.forward(ctx, use_causal_mask=True)
            logits = out["lm_logits"][:, -1, :].float()

            if repetition_penalty != 1.0:
                for tok in set(generated[0].tolist()):
                    if logits[0, tok] < 0:
                        logits[0, tok] *= repetition_penalty
                    else:
                        logits[0, tok] /= repetition_penalty

            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                kth_val = torch.topk(logits, k)[0][:, -1:]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs  = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        return generated


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from core.config import ModelConfig

    cfg   = ModelConfig()
    model = ArdenModel(cfg)
    total = model.num_parameters()
    print(f"ARDEN 1.0 — Parameters: {total:,}  ({total/1e6:.1f} M)")

    x    = torch.randint(0, cfg.vocab_size, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.long)
    out  = model(x, attention_mask=mask)
    print(f"lm_logits : {out['lm_logits'].shape}")
    print("OK")
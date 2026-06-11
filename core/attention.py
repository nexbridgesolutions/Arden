# ARDENWARD 2.2 - Nex Bridge Solutions LLC | TX 9-539-096 | David Arriaga
"""
Multi-Head Self-Attention — CPU-only, pure PyTorch, no fused kernels.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product multi-head attention.

    Supports:
    - Causal (autoregressive) masking via attn_mask (T, T) float tensor
    - Padding masking via key_padding_mask (B, T) bool tensor (True = ignore)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.q_proj   = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj   = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj   = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        self.attn_drop = nn.Dropout(dropout)

    # ── helpers ──────────────────────────────────────────────────────
    def _split_heads(self, t: torch.Tensor, B: int, T: int) -> torch.Tensor:
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # → (B, H, T, d_head)

    def _merge_heads(self, t: torch.Tensor, B: int, T: int) -> torch.Tensor:
        return t.transpose(1, 2).contiguous().view(B, T, self.d_model)

    # ── forward ──────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d_model)
        attn_mask: Optional[torch.Tensor] = None, # (T, T) additive float mask
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) bool, True=pad
    ) -> torch.Tensor:                            # (B, T, d_model)
        B, T, _ = x.shape

        Q = self._split_heads(self.q_proj(x), B, T)  # (B, H, T, dh)
        K = self._split_heads(self.k_proj(x), B, T)
        V = self._split_heads(self.v_proj(x), B, T)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Add causal / positional mask (additive: 0 or -inf)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (T, T) → (1, 1, T, T) broadcast over B and H
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                scores = scores + attn_mask

        # Zero-out padding positions (-inf before softmax)
        if key_padding_mask is not None:
            # (B, T) → (B, 1, 1, T)
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        weights = F.softmax(scores, dim=-1)

        # Avoid NaN when a row is all -inf (pure-padding sequence edge case)
        weights = torch.nan_to_num(weights, nan=0.0)

        weights = self.attn_drop(weights)

        out = torch.matmul(weights, V)              # (B, H, T, dh)
        out = self._merge_heads(out, B, T)          # (B, T, d_model)
        return self.out_proj(out)
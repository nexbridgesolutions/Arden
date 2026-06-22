#!/usr/bin/env python3
"""
ARDEN 1.0 -> GGUF Converter
Copyright 2026 Nex Bridge Solutions LLC - David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

Convierte un checkpoint PyTorch de Arden (arquitectura GPT-2-like) a formato
GGUF para uso con llama.cpp / Ollama.

Uso:
    python3 arden_to_gguf.py \
        --checkpoint /opt/arden/checkpoints/step_00240000_final.pt \
        --tokenizer /opt/arden/data/vocab/tokenizer.json \
        --tokenizer-meta /opt/arden/data/vocab/tokenizer_meta.json \
        --outfile /opt/arden/gguf/arden-0.2b-f16.gguf \
        --ftype f16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import gguf


# =============================================================================
# Arquitectura Arden confirmada (core/config.py - ModelConfig)
# =============================================================================
ARDEN_CONFIG = {
    "vocab_size": 32_000,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 22,
    "d_ff": 3_072,
    "d_head": 64,
    "max_seq_len": 256,
    "norm_eps": 1e-6,
}

FTYPE_MAP = {
    "f32": (gguf.GGMLQuantizationType.F32, np.float32),
    "f16": (gguf.GGMLQuantizationType.F16, np.float16),
}


def load_checkpoint(path: Path) -> dict:
    """Carga el checkpoint .pt y devuelve el state_dict del modelo."""
    print(f"[1/6] Cargando checkpoint: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ck, dict) and "model_state" in ck:
        sd = ck["model_state"]
        print(f"      step={ck.get('step', '?')}")
    elif isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
    else:
        sd = ck  # asume que ya es el state_dict plano

    n_params = sum(v.numel() for v in sd.values())
    print(f"      Tensores: {len(sd)} | Parametros totales: {n_params:,}")
    return sd


def validate_shapes(sd: dict, cfg: dict) -> None:
    """Verifica que las shapes del checkpoint coincidan con la config esperada."""
    print("[2/6] Validando shapes contra configuracion esperada...")

    checks = [
        ("token_emb.weight", (cfg["vocab_size"], cfg["d_model"])),
        ("pos_emb.weight", (cfg["max_seq_len"], cfg["d_model"])),
        ("lm_head.weight", (cfg["vocab_size"], cfg["d_model"])),
        ("norm_final.weight", (cfg["d_model"],)),
        ("layers.0.attn.q_proj.weight", (cfg["d_model"], cfg["d_model"])),
        ("layers.0.ffn.fc1.weight", (cfg["d_ff"], cfg["d_model"])),
    ]
    errors = []
    for name, expected_shape in checks:
        if name not in sd:
            errors.append(f"  FALTA: {name}")
            continue
        actual = tuple(sd[name].shape)
        if actual != expected_shape:
            errors.append(f"  SHAPE MISMATCH en {name}: esperado {expected_shape}, real {actual}")

    detected_layers = len({
        int(k.split(".")[1]) for k in sd if k.startswith("layers.")
    })
    if detected_layers != cfg["n_layers"]:
        errors.append(
            f"  N_LAYERS MISMATCH: config dice {cfg['n_layers']}, "
            f"checkpoint tiene {detected_layers}"
        )

    if errors:
        print("      ERRORES DE VALIDACION:")
        for e in errors:
            print(e)
        raise SystemExit(
            "\nAbortando: el checkpoint no coincide con ARDEN_CONFIG. "
            "Ajusta el diccionario ARDEN_CONFIG en este script antes de reintentar."
        )
    print("      OK - todas las shapes coinciden.")


def build_qkv_fused(sd: dict, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fusiona q_proj/k_proj/v_proj (separados en Arden) en un solo tensor
    c_attn estilo GPT-2: weight [d_model, 3*d_model], bias [3*d_model].

    IMPORTANTE: GGUF/llama.cpp espera el tensor en orientacion (out, in)
    para weight, igual que PyTorch nn.Linear.weight ya esta.
    """
    prefix = f"layers.{layer_idx}.attn"
    q_w = sd[f"{prefix}.q_proj.weight"].numpy()  # [d_model, d_model]
    k_w = sd[f"{prefix}.k_proj.weight"].numpy()
    v_w = sd[f"{prefix}.v_proj.weight"].numpy()
    q_b = sd[f"{prefix}.q_proj.bias"].numpy()
    k_b = sd[f"{prefix}.k_proj.bias"].numpy()
    v_b = sd[f"{prefix}.v_proj.bias"].numpy()

    # Concatenar en dim 0 -> [3*d_model, d_model], igual que GPT-2 c_attn
    # transpuesto (GPT-2 original usa Conv1D con shape [in, out], pero
    # llama.cpp convert para GPT-2 normaliza esto internamente via la
    # convencion nn.Linear: [out_features, in_features]).
    qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)   # [3*d_model, d_model]
    qkv_b = np.concatenate([q_b, k_b, v_b], axis=0)   # [3*d_model]

    return qkv_w, qkv_b


def add_tensor(writer: gguf.GGUFWriter, name: str, tensor: np.ndarray,
                np_dtype, gguf_dtype) -> None:
    """
    Anade un tensor al writer, convirtiendo dtype segun ftype solicitado.

    IMPORTANTE: ciertos tensores DEBEN permanecer en F32 sin importar el
    --ftype elegido, porque el backend de computo de ggml (ggml_mul,
    ggml_norm, lookup de embeddings) no tiene kernels para mezclar F32
    con F16 en esas operaciones especificas. Forzar F16 en estos tensores
    es la causa raiz del error:
        "binary_op: unsupported types: dst: f32, src0: f32, src1: f16"

    Tensores que SIEMPRE van en F32, independientemente de --ftype:
      - Todas las normas (attn_norm, ffn_norm, output_norm) y sus bias
      - Todos los bias (attn_qkv.bias, attn_output.bias, ffn_up.bias,
        ffn_down.bias) -- llama.cpp espera bias en F32 por convencion
      - token_embd / position_embd -- el lookup de embeddings en ggml
        para arquitectura GPT2 opera en F32

    Solo las matrices de pesos "grandes" (attn_qkv.weight, attn_output.weight,
    ffn_up.weight, ffn_down.weight, output.weight) respetan el --ftype
    elegido (f16 para ahorrar espacio).
    """
    FORCE_F32_SUFFIXES = (
        "_norm.weight", "_norm.bias",
        ".bias",                       # cualquier bias, en cualquier tensor
        "token_embd.weight",
        "position_embd.weight",
    )

    force_f32 = any(name.endswith(suf) or name == suf for suf in FORCE_F32_SUFFIXES)

    if force_f32:
        arr = tensor.astype(np.float32)
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
    else:
        arr = tensor.astype(np_dtype)
        writer.add_tensor(name, arr, raw_dtype=gguf_dtype)


def convert(args: argparse.Namespace) -> None:
    cfg = ARDEN_CONFIG
    gguf_dtype, np_dtype = FTYPE_MAP[args.ftype]

    sd = load_checkpoint(Path(args.checkpoint))
    validate_shapes(sd, cfg)

    # ===== Tokenizer metadata =====
    print("[3/6] Cargando metadata del tokenizer...")
    with open(args.tokenizer_meta) as f:
        tok_meta = json.load(f)

    from tokenizers import Tokenizer
    hf_tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = hf_tok.get_vocab_size()
    if vocab_size != cfg["vocab_size"]:
        print(f"      AVISO: tokenizer vocab_size={vocab_size} "
              f"!= config vocab_size={cfg['vocab_size']}")

    vocab = hf_tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, f"<UNUSED_{i}>") for i in range(cfg["vocab_size"])]

    print(f"      Vocab cargado: {len(tokens)} tokens")
    print(f"      BOS={tok_meta['bos_id']} EOS={tok_meta['eos_id']} "
          f"PAD={tok_meta['pad_id']} UNK={tok_meta['unk_id']}")

    # ===== GGUF Writer =====
    print(f"[4/6] Iniciando GGUFWriter -> {args.outfile}")
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(args.outfile, gguf.MODEL_ARCH.GPT2.name.lower()
                              if hasattr(gguf.MODEL_ARCH.GPT2, "name") else "gpt2")

    # --- Metadata general ---
    writer.add_name("ARDEN-0.2B")
    writer.add_description(
        "ARDEN 1.0 - General Purpose Bilingual LLM ES/EN. "
        "Copyright 2026 Nex Bridge Solutions LLC - David Ernesto Arriaga Pineda. "
        "License: Arden Community License v1.0"
    )
    writer.add_vocab_size(cfg["vocab_size"])
    writer.add_context_length(cfg["max_seq_len"])
    writer.add_embedding_length(cfg["d_model"])
    writer.add_block_count(cfg["n_layers"])
    writer.add_feed_forward_length(cfg["d_ff"])
    writer.add_head_count(cfg["n_heads"])
    writer.add_layer_norm_eps(cfg["norm_eps"])
    writer.add_file_type(gguf_dtype)

    # --- Tokenizer ---
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types([gguf.TokenType.NORMAL] * len(tokens))
    writer.add_bos_token_id(tok_meta["bos_id"])
    writer.add_eos_token_id(tok_meta["eos_id"])
    writer.add_pad_token_id(tok_meta["pad_id"])
    writer.add_unk_token_id(tok_meta["unk_id"])

    # BPE merges (requerido por el tokenizer gpt2 de llama.cpp)
    tok_json = json.loads(Path(args.tokenizer).read_text())
    merges_raw = tok_json.get("model", {}).get("merges", [])
    merges = []
    for m in merges_raw:
        if isinstance(m, list):
            merges.append(" ".join(m))
        else:
            merges.append(m)
    writer.add_token_merges(merges)
    print(f"      Merges BPE cargados: {len(merges)}")

    # ===== Tensores: embeddings =====
    print("[5/6] Escribiendo tensores...")
    add_tensor(writer, "token_embd.weight", sd["token_emb.weight"].numpy(),
               np_dtype, gguf_dtype)
    add_tensor(writer, "position_embd.weight", sd["pos_emb.weight"].numpy(),
               np_dtype, gguf_dtype)

    # ===== Tensores: por capa =====
    for i in range(cfg["n_layers"]):
        qkv_w, qkv_b = build_qkv_fused(sd, i)
        add_tensor(writer, f"blk.{i}.attn_qkv.weight", qkv_w, np_dtype, gguf_dtype)
        add_tensor(writer, f"blk.{i}.attn_qkv.bias", qkv_b, np_dtype, gguf_dtype)

        add_tensor(writer, f"blk.{i}.attn_output.weight",
                   sd[f"layers.{i}.attn.out_proj.weight"].numpy(), np_dtype, gguf_dtype)
        add_tensor(writer, f"blk.{i}.attn_output.bias",
                   sd[f"layers.{i}.attn.out_proj.bias"].numpy(), np_dtype, gguf_dtype)

        add_tensor(writer, f"blk.{i}.attn_norm.weight",
                   sd[f"layers.{i}.norm1.weight"].numpy(), np_dtype, gguf_dtype)
        add_tensor(writer, f"blk.{i}.attn_norm.bias",
                   sd[f"layers.{i}.norm1.bias"].numpy(), np_dtype, gguf_dtype)

        add_tensor(writer, f"blk.{i}.ffn_norm.weight",
                   sd[f"layers.{i}.norm2.weight"].numpy(), np_dtype, gguf_dtype)
        add_tensor(writer, f"blk.{i}.ffn_norm.bias",
                   sd[f"layers.{i}.norm2.bias"].numpy(), np_dtype, gguf_dtype)

        add_tensor(writer, f"blk.{i}.ffn_up.weight",
                   sd[f"layers.{i}.ffn.fc1.weight"].numpy(), np_dtype, gguf_dtype)
        add_tensor(writer, f"blk.{i}.ffn_up.bias",
                   sd[f"layers.{i}.ffn.fc1.bias"].numpy(), np_dtype, gguf_dtype)

        add_tensor(writer, f"blk.{i}.ffn_down.weight",
                   sd[f"layers.{i}.ffn.fc2.weight"].numpy(), np_dtype, gguf_dtype)
        add_tensor(writer, f"blk.{i}.ffn_down.bias",
                   sd[f"layers.{i}.ffn.fc2.bias"].numpy(), np_dtype, gguf_dtype)

        if (i + 1) % 5 == 0 or i == cfg["n_layers"] - 1:
            print(f"      Capa {i + 1}/{cfg['n_layers']} escrita")

    # ===== Tensores: salida =====
    add_tensor(writer, "output_norm.weight", sd["norm_final.weight"].numpy(),
               np_dtype, gguf_dtype)
    add_tensor(writer, "output_norm.bias", sd["norm_final.bias"].numpy(),
               np_dtype, gguf_dtype)
    add_tensor(writer, "output.weight", sd["lm_head.weight"].numpy(),
               np_dtype, gguf_dtype)

    # ===== Finalizar =====
    print("[6/6] Escribiendo archivo GGUF a disco...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size_mb = Path(args.outfile).stat().st_size / (1024 * 1024)
    print(f"\nCompletado: {args.outfile} ({out_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convierte checkpoint Arden a GGUF")
    parser.add_argument("--checkpoint", required=True, help="Ruta al .pt")
    parser.add_argument("--tokenizer", required=True, help="Ruta a tokenizer.json")
    parser.add_argument("--tokenizer-meta", required=True, help="Ruta a tokenizer_meta.json")
    parser.add_argument("--outfile", required=True, help="Ruta de salida .gguf")
    parser.add_argument("--ftype", choices=["f32", "f16"], default="f16",
                         help="Precision de salida (f16 recomendado, luego cuantizar con llama-quantize)")
    args = parser.parse_args()

    convert(args)


if __name__ == "__main__":
    main()
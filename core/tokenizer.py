"""
ARDEN 1.0 — Bilingual BPE Tokenizer ES/EN
Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

General purpose tokenizer for Spanish and English.
Built on HuggingFace tokenizers library.
Languages: Spanish, English, Portuguese, French.

Special tokens:
  <PAD> <UNK> <BOS> <EOS> <MASK>
  <USER> <ASSISTANT> <SYSTEM>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence as NormSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


# ──────────────────────────────────────────────────────────────────────────────
#  Special tokens
# ──────────────────────────────────────────────────────────────────────────────

SPECIAL_TOKENS: List[str] = [
    "<PAD>",        # 0 — padding
    "<UNK>",        # 1 — unknown
    "<BOS>",        # 2 — beginning of sequence
    "<EOS>",        # 3 — end of sequence
    "<MASK>",       # 4 — masked token (MLM future use)
    "<USER>",       # 5 — user turn in conversation
    "<ASSISTANT>",  # 6 — assistant turn
    "<SYSTEM>",     # 7 — system prompt
]


# ──────────────────────────────────────────────────────────────────────────────
#  ArdenTokenizer
# ──────────────────────────────────────────────────────────────────────────────

class ArdenTokenizer:
    """
    BPE tokenizer optimized for bilingual ES/EN general purpose text.

    Training:
        tok = ArdenTokenizer(vocab_size=32_000)
        tok.train(files=["data/corpus_es.txt", "data/corpus_en.txt"])
        tok.save("data/vocab/")

    Inference:
        tok = ArdenTokenizer.load("data/vocab/")
        ids = tok.encode("Hola mundo, hello world")
        text = tok.decode(ids)
    """

    def __init__(self, vocab_size: int = 32_000):
        self.vocab_size  = vocab_size
        self._tokenizer  : Optional[Tokenizer] = None

        # Token IDs
        self.pad_id  : int = 0
        self.unk_id  : int = 1
        self.bos_id  : int = 2
        self.eos_id  : int = 3
        self.mask_id : int = 4
        self.user_id : int = 5
        self.asst_id : int = 6
        self.sys_id  : int = 7

    # ── Build untrained tokenizer ─────────────────────────────────────
    def _build_tokenizer(self) -> Tokenizer:
        tok = Tokenizer(BPE(unk_token="<UNK>"))
        tok.normalizer   = NormSequence([NFKC()])
        tok.pre_tokenizer = Whitespace()
        tok.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[("<BOS>", 2), ("<EOS>", 3)],
        )
        return tok

    # ── Train from files ──────────────────────────────────────────────
    def train(
        self,
        files: List[str],
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> None:
        """
        Train BPE from plain text files.
        Each file: one sentence per line or raw text.
        Mix ES and EN files for bilingual vocabulary.
        """
        self._tokenizer = self._build_tokenizer()

        trainer = BpeTrainer(
            vocab_size     = self.vocab_size,
            min_frequency  = min_frequency,
            special_tokens = SPECIAL_TOKENS,
            show_progress  = show_progress,
            initial_alphabet = list(
                "abcdefghijklmnopqrstuvwxyz"
                "áéíóúüñàèìòùâêîôûäëïöüç"  # Spanish
                "ãõâêîôûàèìòùç"             # Portuguese
                "æœÿ"                        # French extended
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "ÁÉÍÓÚÜÑÃÕÀÈÌÒÙÂÊÎÔÛ"
                "0123456789"
                " .,;:!?-_'\"()[]{}/@#$%&*+"
            ),
        )

        self._tokenizer.train(files=files, trainer=trainer)
        self._cache_special_ids()

    # ── Train from iterator ───────────────────────────────────────────
    def train_from_iterator(
        self,
        iterator: Sequence[str],
        min_frequency: int = 2,
        length: Optional[int] = None,
    ) -> None:
        self._tokenizer = self._build_tokenizer()

        trainer = BpeTrainer(
            vocab_size     = self.vocab_size,
            min_frequency  = min_frequency,
            special_tokens = SPECIAL_TOKENS,
            show_progress  = True,
        )

        self._tokenizer.train_from_iterator(
            iterator, trainer=trainer, length=length
        )
        self._cache_special_ids()

    # ── Cache special token IDs ───────────────────────────────────────
    def _cache_special_ids(self) -> None:
        assert self._tokenizer is not None
        self.pad_id  = self._tokenizer.token_to_id("<PAD>")  or 0
        self.unk_id  = self._tokenizer.token_to_id("<UNK>")  or 1
        self.bos_id  = self._tokenizer.token_to_id("<BOS>")  or 2
        self.eos_id  = self._tokenizer.token_to_id("<EOS>")  or 3
        self.mask_id = self._tokenizer.token_to_id("<MASK>") or 4
        self.user_id = self._tokenizer.token_to_id("<USER>") or 5
        self.asst_id = self._tokenizer.token_to_id("<ASSISTANT>") or 6
        self.sys_id  = self._tokenizer.token_to_id("<SYSTEM>") or 7

    # ── Encode ────────────────────────────────────────────────────────
    def encode(
        self,
        text              : str,
        add_special_tokens: bool = True,
        max_length        : Optional[int] = None,
        padding           : bool = False,
        truncation        : bool = True,
    ) -> List[int]:
        assert self._tokenizer is not None, "Tokenizer not trained/loaded"

        if truncation and max_length:
            self._tokenizer.enable_truncation(max_length=max_length)
        else:
            self._tokenizer.no_truncation()

        if padding and max_length:
            self._tokenizer.enable_padding(
                pad_id=self.pad_id, pad_token="<PAD>", length=max_length
            )
        else:
            self._tokenizer.no_padding()

        if not add_special_tokens:
            old_pp = self._tokenizer.post_processor
            self._tokenizer.post_processor = None
            enc = self._tokenizer.encode(text)
            self._tokenizer.post_processor = old_pp
        else:
            enc = self._tokenizer.encode(text)

        return enc.ids

    def encode_batch(
        self,
        texts      : List[str],
        max_length : Optional[int] = None,
        padding    : bool = True,
        truncation : bool = True,
    ) -> List[List[int]]:
        assert self._tokenizer is not None

        if truncation and max_length:
            self._tokenizer.enable_truncation(max_length=max_length)
        if padding and max_length:
            self._tokenizer.enable_padding(
                pad_id=self.pad_id, pad_token="<PAD>", length=max_length
            )

        return [e.ids for e in self._tokenizer.encode_batch(texts)]

    # ── Decode ────────────────────────────────────────────────────────
    def decode(
        self,
        ids                : List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        assert self._tokenizer is not None
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        batch              : List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        assert self._tokenizer is not None
        return self._tokenizer.decode_batch(
            batch, skip_special_tokens=skip_special_tokens
        )

    # ── Chat format helper ────────────────────────────────────────────
    def encode_chat(
        self,
        system   : Optional[str] = None,
        user     : Optional[str] = None,
        assistant: Optional[str] = None,
        max_length: Optional[int] = 512,
    ) -> List[int]:
        """
        Encode a conversation turn with role tokens.
        Format: <SYSTEM>system<EOS><USER>user<EOS><ASSISTANT>
        """
        assert self._tokenizer is not None
        parts = []
        if system:
            parts.append(f"<SYSTEM>{system}<EOS>")
        if user:
            parts.append(f"<USER>{user}<EOS>")
        if assistant:
            parts.append(f"<ASSISTANT>{assistant}<EOS>")
        else:
            parts.append("<ASSISTANT>")

        text = "".join(parts)
        return self.encode(text, add_special_tokens=False, max_length=max_length)

    # ── Vocabulary info ───────────────────────────────────────────────
    @property
    def vocab(self) -> Dict[str, int]:
        assert self._tokenizer is not None
        return self._tokenizer.get_vocab()

    def __len__(self) -> int:
        assert self._tokenizer is not None
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        assert self._tokenizer is not None
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id_: int) -> Optional[str]:
        assert self._tokenizer is not None
        return self._tokenizer.id_to_token(id_)

    # ── Save / Load ───────────────────────────────────────────────────
    def save(self, directory: Union[str, Path]) -> None:
        assert self._tokenizer is not None, "Nothing to save — not trained"
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self._tokenizer.save(str(directory / "tokenizer.json"))

        meta = {
            "vocab_size" : self.vocab_size,
            "pad_id"     : self.pad_id,
            "unk_id"     : self.unk_id,
            "bos_id"     : self.bos_id,
            "eos_id"     : self.eos_id,
            "mask_id"    : self.mask_id,
            "user_id"    : self.user_id,
            "asst_id"    : self.asst_id,
            "sys_id"     : self.sys_id,
            "special_tokens": SPECIAL_TOKENS,
        }
        with open(directory / "tokenizer_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "ArdenTokenizer":
        directory = Path(directory)
        tok_path  = directory / "tokenizer.json"
        meta_path = directory / "tokenizer_meta.json"

        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {directory}")

        obj = cls()
        obj._tokenizer = Tokenizer.from_file(str(tok_path))

        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            obj.vocab_size = meta.get("vocab_size", 32_000)
            obj.pad_id     = meta.get("pad_id",  0)
            obj.unk_id     = meta.get("unk_id",  1)
            obj.bos_id     = meta.get("bos_id",  2)
            obj.eos_id     = meta.get("eos_id",  3)
            obj.mask_id    = meta.get("mask_id", 4)
            obj.user_id    = meta.get("user_id", 5)
            obj.asst_id    = meta.get("asst_id", 6)
            obj.sys_id     = meta.get("sys_id",  7)

        return obj

    def __repr__(self) -> str:
        n = len(self) if self._tokenizer else "untrained"
        return f"ArdenTokenizer(vocab_size={n})"


# ──────────────────────────────────────────────────────────────────────────────
#  Bootstrap — smoke test sin datos reales
# ──────────────────────────────────────────────────────────────────────────────

def build_bootstrap_tokenizer(
    vocab_size: int = 1_000,
    save_dir  : Optional[Path] = None,
) -> ArdenTokenizer:
    """
    Trains a small tokenizer on synthetic ES/EN corpus.
    Useful for model smoke tests before real data is available.
    """
    corpus = [
        # Español
        "Hola, ¿cómo estás? Estoy bien, gracias.",
        "El modelo de inteligencia artificial aprende de los datos.",
        "La red neuronal procesa texto en español e inglés.",
        "Buenos días, ¿en qué puedo ayudarte hoy?",
        "El aprendizaje automático transforma la industria moderna.",
        "Por favor escribe una historia corta sobre el mar.",
        "La tecnología avanza a pasos agigantados cada año.",
        "¿Cuál es la capital de El Salvador? Es San Salvador.",
        "El software libre permite a todos colaborar y mejorar.",
        "Necesito ayuda con mi proyecto de programación en Python.",
        # English
        "Hello, how are you? I am fine, thank you.",
        "The artificial intelligence model learns from data.",
        "The neural network processes text in Spanish and English.",
        "Good morning, how can I help you today?",
        "Machine learning is transforming the modern industry.",
        "Please write a short story about the ocean.",
        "Technology advances at a rapid pace every year.",
        "What is the capital of Arizona? It is Phoenix.",
        "Open source software allows everyone to collaborate.",
        "I need help with my Python programming project.",
        # Português
        "Olá, como vai você? Estou bem, obrigado.",
        "O modelo de inteligência artificial aprende com os dados.",
        "A rede neural processa texto em quatro idiomas diferentes.",
        "Bom dia, como posso ajudá-lo hoje?",
        "A tecnologia avança rapidamente a cada ano que passa.",
        # Français
        "Bonjour, comment allez-vous? Je vais bien, merci.",
        "Le modèle d'intelligence artificielle apprend des données.",
        "Le réseau de neurones traite le texte en plusieurs langues.",
        "Bonjour, comment puis-je vous aider aujourd'hui?",
        "La technologie progresse rapidement chaque année.",
    ] * 100

    tok = ArdenTokenizer(vocab_size=vocab_size)
    tok.train_from_iterator(corpus, min_frequency=1, length=len(corpus))

    if save_dir:
        tok.save(save_dir)

    return tok


# ──────────────────────────────────────────────────────────────────────────────
#  Main — smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    print("Training bootstrap tokenizer...")
    tok = build_bootstrap_tokenizer(vocab_size=500)
    print(tok)

    # Test ES
    text_es = "Hola, soy Arden, un asistente de propósito general."
    ids_es  = tok.encode(text_es)
    back_es = tok.decode(ids_es)
    print(f"\nES Input : {text_es}")
    print(f"ES IDs   : {ids_es[:10]} ...")
    print(f"ES Decode: {back_es}")

    # Test EN
    text_en = "Hello, I am Arden, a general purpose assistant."
    ids_en  = tok.encode(text_en)
    back_en = tok.decode(ids_en)
    print(f"\nEN Input : {text_en}")
    print(f"EN IDs   : {ids_en[:10]} ...")
    print(f"EN Decode: {back_en}")

    # Test chat format
    chat_ids = tok.encode_chat(
        system="Eres un asistente útil.",
        user="¿Qué es Python?",
    )
    print(f"\nChat IDs : {chat_ids[:15]} ...")
    print("OK")
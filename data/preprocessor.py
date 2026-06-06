"""
ARDEN 1.0 — Data Preprocessor
Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

Tokenizes raw JSONL corpus and creates train/val/test splits.
Output: data/processed/train.jsonl, val.jsonl, test.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tokenizer import ArdenTokenizer
from core.config import ArdenConfig


# ──────────────────────────────────────────────────────────────────────────────
#  ArdenPreprocessor
# ──────────────────────────────────────────────────────────────────────────────

class ArdenPreprocessor:
    """
    1. Trains tokenizer on raw corpus (if not already trained)
    2. Tokenizes all texts into token ID sequences
    3. Chunks sequences to max_seq_len
    4. Splits into train / val / test
    5. Saves to data/processed/
    """

    def __init__(self, config: ArdenConfig):
        self.config       = config
        self.data_cfg     = config.data
        self.model_cfg    = config.model

        self.raw_dir      = Path(self.data_cfg.data_dir)
        self.processed_dir = Path(self.data_cfg.processed_dir)
        self.vocab_dir    = Path(self.data_cfg.vocab_dir)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer: Optional[ArdenTokenizer] = None

    # ── Step 1: Train or load tokenizer ──────────────────────────────
    def prepare_tokenizer(self) -> ArdenTokenizer:
        tok_path = self.vocab_dir / "tokenizer.json"

        if tok_path.exists():
            print("Loading existing tokenizer...")
            self.tokenizer = ArdenTokenizer.load(self.vocab_dir)
            print(f"  Loaded — vocab size: {len(self.tokenizer):,}")
            return self.tokenizer

        print("Training tokenizer from corpus...")
        self.tokenizer = ArdenTokenizer(
            vocab_size=self.model_cfg.vocab_size
        )

        # Collect all raw JSONL files
        jsonl_files = list(self.raw_dir.glob("wikipedia_*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No JSONL files found in {self.raw_dir}\n"
                "Run dataset_loader.py first."
            )

        print(f"  Found {len(jsonl_files)} corpus files:")
        for f in jsonl_files:
            print(f"    {f.name}")

        # Stream texts for tokenizer training
        def text_iterator():
            for jsonl in jsonl_files:
                with open(jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        yield record["text"]

        self.tokenizer.train_from_iterator(
            text_iterator(),
            min_frequency=2,
        )

        self.tokenizer.save(self.vocab_dir)
        print(f"  Tokenizer saved to {self.vocab_dir}")
        print(f"  Vocab size: {len(self.tokenizer):,}")
        return self.tokenizer

    # ── Step 2: Tokenize and chunk ────────────────────────────────────
    def _tokenize_file(self, jsonl_path: Path) -> List[List[int]]:
        assert self.tokenizer is not None
        max_len = self.model_cfg.max_seq_len
        chunks  = []
        buffer  = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text   = record["text"]

                ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )

                buffer.extend(ids)

                # Chunk buffer into max_len sequences
                while len(buffer) >= max_len:
                    chunks.append(buffer[:max_len])
                    buffer = buffer[max_len:]

        # Last partial chunk
        if len(buffer) >= self.data_cfg.min_length:
            # Pad to max_len
            pad_id = self.tokenizer.pad_id
            buffer = buffer + [pad_id] * (max_len - len(buffer))
            chunks.append(buffer[:max_len])

        return chunks

    # ── Step 3: Process all files ─────────────────────────────────────
    def process(self) -> None:
        assert self.tokenizer is not None, "Call prepare_tokenizer() first"

        jsonl_files = list(self.raw_dir.glob("wikipedia_*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files in {self.raw_dir}")

        print(f"\nTokenizing {len(jsonl_files)} corpus files...")
        all_chunks = []

        for jsonl in jsonl_files:
            print(f"  Processing {jsonl.name}...")
            chunks = self._tokenize_file(jsonl)
            all_chunks.extend(chunks)
            print(f"    {len(chunks):,} chunks")

        print(f"\nTotal chunks: {len(all_chunks):,}")

        # Shuffle
        random.seed(self.config.training.seed)
        random.shuffle(all_chunks)

        # Split
        n      = len(all_chunks)
        n_train = int(n * self.data_cfg.train_split)
        n_val   = int(n * self.data_cfg.val_split)

        train = all_chunks[:n_train]
        val   = all_chunks[n_train:n_train + n_val]
        test  = all_chunks[n_train + n_val:]

        print(f"  Train : {len(train):,}")
        print(f"  Val   : {len(val):,}")
        print(f"  Test  : {len(test):,}")

        # Save splits
        self._save_split(train, self.processed_dir / "train.jsonl")
        self._save_split(val,   self.processed_dir / "val.jsonl")
        self._save_split(test,  self.processed_dir / "test.jsonl")

        print(f"\nSaved to {self.processed_dir}")

    # ── Save split to JSONL ───────────────────────────────────────────
    def _save_split(self, chunks: List[List[int]], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps({"input_ids": chunk}) + "\n")
        size = path.stat().st_size / (1024 ** 2)
        print(f"  Saved {path.name}: {len(chunks):,} chunks ({size:.1f} MB)")

    # ── Full pipeline ─────────────────────────────────────────────────
    def run(self) -> None:
        print("=" * 60)
        print("  ARDEN 1.0 — Data Preprocessor")
        print("=" * 60)
        self.prepare_tokenizer()
        self.process()
        print("\nPreprocessing complete.")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core.config import ArdenConfig

    cfg = ArdenConfig()
    preprocessor = ArdenPreprocessor(cfg)
    preprocessor.run()
"""
ARDEN 1.0 — Dataset Loader
Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

Downloads and loads Wikipedia ES, EN, PT, FR for training.
Uses HuggingFace datasets library (streaming — no full download needed).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterator, List, Optional
from datasets import load_dataset


# ──────────────────────────────────────────────────────────────────────────────
#  Language sources
# ──────────────────────────────────────────────────────────────────────────────

WIKIPEDIA_SOURCES = {
    "es": ("wikimedia/wikipedia", "20231101.es"),
    "en": ("wikimedia/wikipedia", "20231101.en"),
    "pt": ("wikimedia/wikipedia", "20231101.pt"),
    "fr": ("wikimedia/wikipedia", "20231101.fr"),
}

# ──────────────────────────────────────────────────────────────────────────────
#  ArdenDatasetLoader
# ──────────────────────────────────────────────────────────────────────────────

class ArdenDatasetLoader:
    """
    Streams Wikipedia articles in ES, EN, PT, FR.
    Uses streaming mode — no need to download full dataset.
    Saves processed text to JSONL files for training.
    """

    def __init__(
        self,
        data_dir      : Path,
        languages     : List[str] = ["es", "en", "pt", "fr"],
        min_length    : int = 100,
        max_length    : int = 2_048,
        max_per_lang  : Optional[int] = None,  # None = all available
    ):
        self.data_dir    = Path(data_dir)
        self.languages   = languages
        self.min_length  = min_length
        self.max_length  = max_length
        self.max_per_lang = max_per_lang

        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Stream articles from one language ────────────────────────────
    def _stream_language(self, lang: str) -> Iterator[str]:
        source = WIKIPEDIA_SOURCES.get(lang)
        if source is None:
            raise ValueError(f"Language not supported: {lang}")

        print(f"  Streaming Wikipedia {lang.upper()}...")

        dataset = load_dataset(
            source[0],
            source[1],
            split="train",
            streaming=True,
        )

        count = 0
        for article in dataset:
            text = article.get("text", "").strip()

            # Filter by length
            if len(text) < self.min_length:
                continue

            # Truncate if too long
            if len(text) > self.max_length * 6:  # ~6 chars per token
                text = text[:self.max_length * 6]

            yield text
            count += 1

            if self.max_per_lang and count >= self.max_per_lang:
                break

        print(f"  Done {lang.upper()}: {count:,} articles")

    # ── Save one language to JSONL ────────────────────────────────────
    def download_language(self, lang: str) -> Path:
        out_path = self.raw_dir / f"wikipedia_{lang}.jsonl"

        if out_path.exists():
            print(f"  {lang.upper()} already exists — skipping")
            return out_path

        print(f"\nDownloading Wikipedia {lang.upper()}...")
        count = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for text in self._stream_language(lang):
                record = {"lang": lang, "text": text}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                if count % 10_000 == 0:
                    print(f"    {count:,} articles saved...")

        print(f"  Saved: {out_path} ({count:,} articles)")
        return out_path

    # ── Download all languages ────────────────────────────────────────
    def download_all(self) -> List[Path]:
        paths = []
        for lang in self.languages:
            path = self.download_language(lang)
            paths.append(path)
        return paths

    # ── Iterator for training ─────────────────────────────────────────
    def iter_texts(self, split_file: Optional[Path] = None) -> Iterator[str]:
        """
        Iterate over all texts from saved JSONL files.
        Used by tokenizer training and dataset preprocessing.
        """
        if split_file and split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    yield record["text"]
            return

        for lang in self.languages:
            jsonl_path = self.raw_dir / f"wikipedia_{lang}.jsonl"
            if not jsonl_path.exists():
                print(f"  Warning: {jsonl_path} not found — run download_all() first")
                continue
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    yield record["text"]

    # ── Stats ─────────────────────────────────────────────────────────
    def stats(self) -> None:
        print("\nARDEN Dataset Stats:")
        print("=" * 40)
        total = 0
        for lang in self.languages:
            path = self.raw_dir / f"wikipedia_{lang}.jsonl"
            if path.exists():
                count = sum(1 for _ in open(path, encoding="utf-8"))
                size  = path.stat().st_size / (1024 ** 2)
                print(f"  {lang.upper()}: {count:>8,} articles  ({size:.1f} MB)")
                total += count
            else:
                print(f"  {lang.upper()}: not downloaded")
        print(f"  {'TOTAL':>4}: {total:>8,} articles")
        print("=" * 40)


# ──────────────────────────────────────────────────────────────────────────────
#  Main — smoke test with small sample
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data"

    # Test with 100 articles per language
    loader = ArdenDatasetLoader(
        data_dir    = data_dir,
        languages   = ["es", "en", "pt", "fr"],
        max_per_lang= 100,
    )

    print("ARDEN 1.0 — Dataset Loader Test")
    print("Downloading 100 articles per language...")
    loader.download_all()
    loader.stats()

    # Show sample
    print("\nSample texts:")
    for i, text in enumerate(loader.iter_texts()):
        print(f"\n[{i+1}] {text[:200]}...")
        if i >= 3:
            break
"""
ARDEN 1.0 — Training Loop
Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

Main training script for ARDEN 1.1B.
Supports CPU-only and CUDA (auto-detected).
Resume from checkpoint supported.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import ArdenConfig, TrainingConfig, DEFAULT_CONFIG
from core.model import ArdenModel
from core.tokenizer import ArdenTokenizer

# ──────────────────────────────────────────────────────────────────────────────
#  Checkpoint interval — igual que Ardenward
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL_HOURS = 4
CHECKPOINT_INTERVAL_SECS  = CHECKPOINT_INTERVAL_HOURS * 3600

# ──────────────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ArdenDataset(Dataset):
    """Loads preprocessed JSONL chunks for training."""

    def __init__(self, jsonl_path: Path, max_seq_len: int):
        self.samples    : List[List[int]] = []
        self.max_seq_len = max_seq_len

        print(f"  Loading {jsonl_path.name}...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                ids    = record["input_ids"]
                if len(ids) >= 2:
                    self.samples.append(ids)

        print(f"  Loaded {len(self.samples):,} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids    = self.samples[idx][:self.max_seq_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels    = input_ids.clone()
        attention_mask = torch.ones(len(ids), dtype=torch.long)

        return {
            "input_ids"     : input_ids,
            "labels"        : labels,
            "attention_mask": attention_mask,
        }


def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad batch to same length."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids      = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels         = torch.full((len(batch), max_len), -100,   dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len,          dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L]      = b["input_ids"]
        labels[i, :L]         = b["labels"]
        attention_mask[i, :L] = b["attention_mask"]

    return {
        "input_ids"     : input_ids,
        "labels"        : labels,
        "attention_mask": attention_mask,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Learning rate scheduler
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainingConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(cfg.warmup_steps, 1)

    progress = (step - cfg.warmup_steps) / max(
        cfg.max_steps - cfg.warmup_steps, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + (cfg.learning_rate - cfg.min_learning_rate) * cosine


# ──────────────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────────────

class ArdenTrainer:

    def __init__(self, config: ArdenConfig):
        self.config   = config
        self.cfg_t    = config.training
        self.cfg_m    = config.model
        self.cfg_ck   = config.checkpoint
        self.cfg_log  = config.logging

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # CPU threading
        if self.device.type == "cpu":
            torch.set_num_threads(self.cfg_t.num_threads)
            torch.set_num_interop_threads(self.cfg_t.num_interop_threads)
            print(f"  CPU threads: {self.cfg_t.num_threads}")

        # Reproducibility
        torch.manual_seed(self.cfg_t.seed)
        random.seed(self.cfg_t.seed)

        # Paths
        self.processed_dir  = Path(config.data.processed_dir)
        self.checkpoint_dir = Path(self.cfg_ck.checkpoint_dir)
        self.log_dir        = Path(self.cfg_log.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.step       = 0
        self.best_loss  = float("inf")
        self.log_file   = open(self.log_dir / "arden_training.log", "a", encoding="utf-8")

    # ── Build model ───────────────────────────────────────────────────   
    def _build_model(self) -> ArdenModel:
        print("\nBuilding ARDEN model...")
        model = ArdenModel(self.cfg_m).to(self.device)
        params = model.num_parameters()
        print(f"  Parameters: {params:,}  ({params/1e6:.1f} M)")

        # Freeze backbone — solo entrena últimas 2 capas + lm_head
        # Igual que Ardenward freeze_backbone — mantiene RAM bajo
        print("  Freezing backbone (last 2 layers + lm_head trainable)...")
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Descongelar solo últimas 2 capas y lm_head
        for name, param in model.named_parameters():
            if any(x in name for x in [
                "layers.20.", "layers.21.",  # últimas 2 capas
                "norm_final",
                "lm_head",
            ]):
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = model.num_parameters()
        print(f"  Trainable : {trainable:,}  ({trainable/1e6:.1f} M)")
        print(f"  Frozen    : {total-trainable:,}  ({(total-trainable)/1e6:.1f} M)")
        return model

    # ── Build optimizer ───────────────────────────────────────────────   
    def _build_optimizer(self, model: ArdenModel) -> torch.optim.Optimizer:
        # Solo parámetros entrenables
        decay    = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() < 2]

        groups = [
            {"params": decay,    "weight_decay": self.cfg_t.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            groups,
            lr   = self.cfg_t.learning_rate,
            betas= (self.cfg_t.beta1, self.cfg_t.beta2),
            eps  = self.cfg_t.epsilon,
        )


    # ── Load datasets ─────────────────────────────────────────────────
    def _load_datasets(self, pad_id: int) -> Tuple[DataLoader, DataLoader]:
        train_path = self.processed_dir / "train.jsonl"
        val_path   = self.processed_dir / "val.jsonl"

        if not train_path.exists():
            raise FileNotFoundError(
                f"train.jsonl not found in {self.processed_dir}\n"
                "Run preprocessor.py first."
            )

        train_ds = ArdenDataset(train_path, self.cfg_m.max_seq_len)
        val_ds   = ArdenDataset(val_path,   self.cfg_m.max_seq_len)

        _collate = lambda b: collate_fn(b, pad_id=pad_id)

        train_loader = DataLoader(
            train_ds,
            batch_size  = self.cfg_t.batch_size,
            shuffle     = True,
            num_workers = self.cfg_t.num_workers,
            pin_memory  = self.cfg_t.pin_memory,
            collate_fn  = _collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = self.cfg_t.batch_size,
            shuffle     = False,
            num_workers = self.cfg_t.num_workers,
            pin_memory  = self.cfg_t.pin_memory,
            collate_fn  = _collate,
        )

        return train_loader, val_loader

    # ── Save checkpoint ───────────────────────────────────────────────
    def _save_checkpoint(
        self,
        model    : ArdenModel,
        optimizer: torch.optim.Optimizer,
        loss     : float,
        tag      : str = "",
    ) -> None:
        name = f"step_{self.step:08d}{tag}.pt"
        path = self.checkpoint_dir / name

        epoch = (self.step * self.cfg_t.batch_size * self.cfg_t.gradient_accumulation_steps) // max(1, self.cfg_t.batch_size * self.cfg_t.gradient_accumulation_steps * 10)
        torch.save({
            "step"           : self.step,
            "epoch"          : epoch,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss"           : loss,
            "config"         : self.config.to_dict() if hasattr(self.config, "to_dict") else {},
        }, path)

        print(f"  Checkpoint saved: {name}")

        # Keep only last N checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        while len(checkpoints) > self.cfg_ck.keep_last_n_checkpoints:
            checkpoints[0].unlink()
            checkpoints = checkpoints[1:]

    # ── Save best model ───────────────────────────────────────────────
    def _save_best(self, model: ArdenModel, loss: float) -> None:
        torch.save({
            "step"       : self.step,
            "model_state": model.state_dict(),
            "loss"       : loss,
        }, self.cfg_ck.best_model_path)
        print(f"  Best model saved (loss={loss:.4f})")

    # ── Resume from checkpoint ────────────────────────────────────────
    def _resume(
        self,
        model    : ArdenModel,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        resume = self.cfg_ck.resume_from
        if not resume:
            return

        path = Path(resume)
        if not path.exists():
            print(f"  Warning: checkpoint {path} not found — starting fresh")
            return

        print(f"  Resuming from {path}...")
        ck = torch.load(path, map_location=self.device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        self.step = ck.get("step", 0)
        print(f"  Resumed at step {self.step}")

    # ── Evaluate ──────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(
        self,
        model     : ArdenModel,
        val_loader: DataLoader,
    ) -> float:
        model.eval()
        total_loss = 0.0
        steps      = 0

        for batch in val_loader:
            input_ids      = batch["input_ids"].to(self.device)
            labels         = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            out = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )

            total_loss += out["loss"].item()
            steps      += 1

            if steps >= self.cfg_t.eval_steps:
                break

        model.train()
        return total_loss / max(steps, 1)

    # ── Log ───────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    # ── Main training loop ────────────────────────────────────────────
    def train(self) -> None:
        print("=" * 60)
        print("  ARDEN 1.0 — Training")
        print("=" * 60)

        # Load tokenizer for pad_id
        vocab_dir = Path(self.config.data.vocab_dir)
        if (vocab_dir / "tokenizer.json").exists():
            tok = ArdenTokenizer.load(vocab_dir)
            pad_id = tok.pad_id
        else:
            pad_id = 0

        model        = self._build_model()
        optimizer    = self._build_optimizer(model)
        self._resume(model, optimizer)

        train_loader, val_loader = self._load_datasets(pad_id)

        model.train()
        optimizer.zero_grad()

        accum_loss  = 0.0
        accum_steps = 0
        t0          = time.time()

        self._log(f"\n{'─'*60}")
        self._log(f"  ARDEN 1.0 — Nex Bridge Solutions LLC")
        self._log(f"  General Purpose Bilingual LLM ES/EN/PT/FR")
        self._log(f"{'─'*60}")
        self._log(f"  Device         : {self.device}")
        self._log(f"  Parameters     : 1,177,616,384 (1.1B)")
        self._log(f"  Trainable      : 100,720,640 (100.7M — frozen backbone)")
        self._log(f"  Max steps      : {self.cfg_t.max_steps:,}")
        self._log(f"  Batch size     : {self.cfg_t.batch_size}")
        self._log(f"  Grad accum     : {self.cfg_t.gradient_accumulation_steps}")
        self._log(f"  Effective batch: {self.cfg_t.batch_size * self.cfg_t.gradient_accumulation_steps}")
        self._log(f"  Learning rate  : {self.cfg_t.learning_rate}")
        self._log(f"  Checkpoint     : cada {CHECKPOINT_INTERVAL_HOURS}h (dentro del loop)")
        self._log(f"  Resume         : automático desde último checkpoint")
        self._log(f"  Log            : {self.log_dir / 'arden_training.log'}")
        self._log(f"{'─'*60}")
        self._log(f"  INICIANDO ENTRENAMIENTO — step {self.step}")
        self._log(f"{'─'*60}\n")

        # Infinite dataloader iterator
        def cycle(loader):
            while True:
                for batch in loader:
                    yield batch

        data_iter      = cycle(train_loader)
        last_ckpt_time = time.time()

        while self.step < self.cfg_t.max_steps:
            batch = next(data_iter)

            input_ids      = batch["input_ids"].to(self.device)
            labels         = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Forward
            out  = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
            loss = out["loss"] / self.cfg_t.gradient_accumulation_steps
            loss.backward()

            accum_loss  += loss.item()
            accum_steps += 1

            # Gradient accumulation step
            if accum_steps % self.cfg_t.gradient_accumulation_steps == 0:
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    model.parameters(), self.cfg_t.grad_clip
                )

                # Update LR
                lr = get_lr(self.step, self.cfg_t)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()
                self.step += 1

                # Logging
                # Logging
                if self.step % self.cfg_t.log_interval == 0:
                    elapsed  = time.time() - t0
                    loss_val = accum_loss * self.cfg_t.gradient_accumulation_steps
                    ppl      = math.exp(min(loss_val, 20))
                    ram_gb   = self._ram_usage()
                    epoch    = (self.step * self.cfg_t.batch_size * self.cfg_t.gradient_accumulation_steps) // max(len(train_loader.dataset), 1) + 1

                    self._log(
                        f"epoch {epoch:>3} | "
                        f"step {self.step:>6} | "
                        f"loss {loss_val:.4f} | "
                        f"ppl {ppl:.1f} | "
                        f"lr {lr:.2e} | "
                        f"RAM {ram_gb:.1f}GB | "
                        f"time {elapsed:.0f}s"
                    )
                    accum_loss = 0.0

                # Evaluation
                if self.step % self.cfg_t.eval_interval == 0:
                    val_loss = self._evaluate(model, val_loader)
                    self._log(f"\n  VAL loss: {val_loss:.4f}  ppl: {math.exp(min(val_loss,20)):.1f}\n")

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self._save_best(model, val_loss)

                # Checkpoint cada 4 horas — igual que Ardenward
                now = time.time()
                if now - last_ckpt_time >= CHECKPOINT_INTERVAL_SECS:
                    self._save_checkpoint(model, optimizer, accum_loss)
                    last_ckpt_time = now

        # Final save
        self._save_checkpoint(model, optimizer, accum_loss, tag="_final")
        self._log(f"\nTraining complete at step {self.step}")
        self.log_file.close()

    # ── RAM usage ─────────────────────────────────────────────────────
    def _ram_usage(self) -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e9
        except ImportError:
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg     = ArdenConfig()
    trainer = ArdenTrainer(cfg)
    trainer.train()
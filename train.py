"""
ARDEN 0.2B — Training Loop
Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

Training script for ARDEN 0.2B on GTX 1050 2GB with float16 AMP.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import TrainingConfig
from core.config import ArdenConfig
from core.model import ArdenModel
from core.tokenizer import ArdenTokenizer

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CHECKPOINT_INTERVAL_HOURS = 4
CHECKPOINT_INTERVAL_SECS  = CHECKPOINT_INTERVAL_HOURS * 3600


class ArdenDataset(Dataset):
    def __init__(self, jsonl_path: Path, max_seq_len: int):
        self.jsonl_path  = jsonl_path
        self.max_seq_len = max_seq_len
        self._cache      : dict = {}
        self._cache_size : int  = 20_000

        print(f"  Indexing {jsonl_path.name}...")
        self.offsets: List[int] = []

        with open(jsonl_path, "rb") as f:
            offset = 0
            for line in f:
                if len(line.strip()) > 2:
                    self.offsets.append(offset)
                offset += len(line)

        print(f"  Indexed {len(self.offsets):,} samples")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self.offsets[idx]
        if offset in self._cache:
            record = self._cache[offset]
        else:
            with open(self.jsonl_path, "rb") as f:
                f.seek(offset)
                record = json.loads(f.readline())
                if len(self._cache) < self._cache_size:
                    self._cache[offset] = record

        full_ids = record["input_ids"]
        if len(full_ids) > self.max_seq_len:
            start = random.randint(0, len(full_ids) - self.max_seq_len)
            ids = full_ids[start:start + self.max_seq_len]
        else:
            ids = full_ids
        input_ids      = torch.tensor(ids, dtype=torch.long)
        labels         = input_ids.clone()
        attention_mask = torch.ones(len(ids), dtype=torch.long)
        return {
            "input_ids"     : input_ids,
            "labels"        : labels,
            "attention_mask": attention_mask,
        }


def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len        = max(b["input_ids"].size(0) for b in batch)
    input_ids      = torch.full((len(batch), max_len), pad_id,  dtype=torch.long)
    labels         = torch.full((len(batch), max_len), -100,    dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len,           dtype=torch.long)

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


def get_lr(step: int, cfg: TrainingConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + (cfg.learning_rate - cfg.min_learning_rate) * cosine


class ArdenTrainer:

    def __init__(self, config):
        self.config  = config
        self.cfg_t   = config.training
        self.cfg_m   = config.model
        self.cfg_ck  = config.checkpoint
        self.cfg_log = config.logging

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU   : {torch.cuda.get_device_name(0)}")
            print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        torch.manual_seed(self.cfg_t.seed)
        random.seed(self.cfg_t.seed)

        self.processed_dir  = Path(config.data.processed_dir)
        self.checkpoint_dir = Path(self.cfg_ck.checkpoint_dir)
        self.log_dir        = Path(config.logging.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.step      = 0
        self.best_loss = float("inf")
        self.log_file  = open(self.log_dir / "arden_05b_training.log", "a", encoding="utf-8")

    def _build_model(self) -> ArdenModel:
        print("\nBuilding ARDEN 0.2B model...")
        model = ArdenModel(self.cfg_m)

        # Freeze backbone — entrena solo últimas 2 capas + lm_head
        print("  Freezing backbone...")
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(x in name for x in ["layers.0.", "layers.1.", "norm_final", "lm_head"]):
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = model.num_parameters()
        print(f"  Parameters: {total:,} ({total/1e6:.1f}M)")
        print(f"  Trainable : {trainable:,} ({trainable/1e6:.1f}M)")
        print(f"  Frozen    : {total-trainable:,} ({(total-trainable)/1e6:.1f}M)")

        return model.to(self.device)

    def _build_optimizer(self, model: ArdenModel) -> torch.optim.Optimizer:
        decay    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg_t.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg_t.learning_rate,
            betas=(self.cfg_t.beta1, self.cfg_t.beta2),
            eps=self.cfg_t.epsilon,
        )

    def _load_datasets(self, pad_id: int) -> Tuple[DataLoader, DataLoader]:
        train_path = self.processed_dir / "train.jsonl"
        val_path   = self.processed_dir / "val.jsonl"

        if not train_path.exists():
            raise FileNotFoundError(f"train.jsonl not found — run preprocessor.py first")

        train_ds = ArdenDataset(train_path, self.cfg_m.max_seq_len)
        val_ds   = ArdenDataset(val_path,   self.cfg_m.max_seq_len)

        _collate = lambda b: collate_fn(b, pad_id=pad_id)

        train_loader = DataLoader(train_ds, batch_size=self.cfg_t.batch_size,
                                  shuffle=True, num_workers=self.cfg_t.num_workers,
                                  pin_memory=self.cfg_t.pin_memory, collate_fn=_collate)
        val_loader   = DataLoader(val_ds, batch_size=self.cfg_t.batch_size,
                                  shuffle=False, num_workers=self.cfg_t.num_workers,
                                  pin_memory=self.cfg_t.pin_memory, collate_fn=_collate)
        return train_loader, val_loader

    def _save_checkpoint(self, model, optimizer, loss, tag=""):
        name = f"step_{self.step:08d}{tag}.pt"
        path = self.checkpoint_dir / name
        torch.save({
            "step"           : self.step,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss"           : loss,
        }, path)
        print(f"  Checkpoint saved: {name}")

        # Mantener solo los últimos N checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        while len(checkpoints) > self.cfg_t.keep_last_n_checkpoints:
            checkpoints[0].unlink()
            checkpoints = checkpoints[1:]

    def _save_best(self, model, loss):
        torch.save({"step": self.step, "model_state": model.state_dict(), "loss": loss},
                   self.cfg_ck.best_model_path)
        print(f"  Best model saved (loss={loss:.4f})")

    @torch.no_grad()
    def _evaluate(self, model, val_loader) -> float:
        model.eval()
        total_loss = 0.0
        steps      = 0
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(self.device)
            labels         = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            with torch.amp.autocast('cuda', enabled=self.cfg_t.use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out["loss"].item()
            steps      += 1
            if steps >= self.cfg_t.eval_steps:
                break
        model.train()
        return total_loss / max(steps, 1)

    def _log(self, msg: str) -> None:
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def _ram_usage(self) -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e9
        except:
            return 0.0

    def train(self) -> None:
        print("=" * 60)
        print("  ARDEN 1.0.1 — Training")
        print("=" * 60)

        vocab_dir = Path(self.config.data.vocab_dir)
        pad_id    = ArdenTokenizer.load(vocab_dir).pad_id if (vocab_dir / "tokenizer.json").exists() else 0

        model     = self._build_model()
        optimizer = self._build_optimizer(model)
        scaler    = torch.amp.GradScaler('cuda', enabled=self.cfg_t.use_amp)

        # Resume automático desde el checkpoint
        best_path   = self.checkpoint_dir / "best_model.pt"
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))

        # Reúne candidatos con su step real, elige el MÁS AVANZADO
        candidates = []
        if checkpoints:
            last_ckpt = checkpoints[-1]
            try:
                step_ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=True)["step"]
                candidates.append((step_ckpt, last_ckpt))
            except Exception:
                pass
        if best_path.exists():
            try:
                step_best = torch.load(best_path, map_location="cpu", weights_only=True)["step"]
                candidates.append((step_best, best_path))
            except Exception:
                pass

        resume_path = max(candidates, key=lambda c: c[0])[1] if candidates else None

        if resume_path is not None:
            print(f"\n  Resuming from {resume_path.name}...")
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                    print("  Optimizer state restored")
                except ValueError:
                    print("  Optimizer state skipped (freeze config changed) — fresh optimizer")
            self.step = ckpt["step"]
            print(f"  Resumed at step {self.step}")
        else:
            print("\n  No checkpoint found — starting from scratch")

        train_loader, val_loader = self._load_datasets(pad_id)

        model.train()
        optimizer.zero_grad()

        accum_loss     = 0.0
        accum_steps    = 0
        t0             = time.time()
        last_ckpt_time = time.time()

        self._log(f"\n{'─'*60}")
        self._log(f"  ARDEN 0.2B — Nex Bridge Solutions LLC")
        self._log(f"  General Purpose Bilingual LLM ES/EN/PT/FR")
        self._log(f"{'─'*60}")
        self._log(f"  Device         : {self.device}")
        self._log(f"  GPU            : {torch.cuda.get_device_name(0)}")
        total_params = sum(p.numel() for p in model.parameters())
        self._log(f"  Parameters     : {total_params:,} ({total_params/1e6:.0f}M)")
        self._log(f"  AMP float16    : {self.cfg_t.use_amp}")
        self._log(f"  Batch size     : {self.cfg_t.batch_size}")
        self._log(f"  Grad accum     : {self.cfg_t.gradient_accumulation_steps}")
        self._log(f"  Effective batch: {self.cfg_t.batch_size * self.cfg_t.gradient_accumulation_steps}")
        self._log(f"  Learning rate  : {self.cfg_t.learning_rate}")
        self._log(f"  Checkpoint     : cada {CHECKPOINT_INTERVAL_HOURS}h")
        self._log(f"{'─'*60}")
        self._log(f"  INICIANDO ENTRENAMIENTO — step {self.step}")
        self._log(f"{'─'*60}\n")

        def cycle(loader):
            while True:
                for batch in loader:
                    yield batch

        data_iter = cycle(train_loader)

        while self.step < self.cfg_t.max_steps:
            batch          = next(data_iter)
            input_ids      = batch["input_ids"].to(self.device)
            labels         = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.cfg_t.use_amp):
                out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"] / self.cfg_t.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accum_loss  += loss.item()
            accum_steps += 1

            if accum_steps % self.cfg_t.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg_t.grad_clip)

                lr = get_lr(self.step, self.cfg_t)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                self.step += 1

                if self.step % self.cfg_t.log_interval == 0:
                    elapsed  = time.time() - t0
                    loss_val = accum_loss * self.cfg_t.gradient_accumulation_steps
                    ppl      = math.exp(min(loss_val, 20))
                    vram_gb  = torch.cuda.memory_allocated() / 1e9
                    ram_gb   = self._ram_usage()
                    epoch    = (self.step * self.cfg_t.batch_size * self.cfg_t.gradient_accumulation_steps) // max(len(train_loader.dataset), 1) + 1

                    self._log(
                        f"epoch {epoch:>3} | step {self.step:>6} | "
                        f"loss {loss_val:.4f} | ppl {ppl:.1f} | "
                        f"lr {lr:.2e} | VRAM {vram_gb:.1f}GB | "
                        f"RAM {ram_gb:.1f}GB | time {elapsed:.0f}s"
                    )
                    accum_loss = 0.0

                if self.step % self.cfg_t.eval_interval == 0 and self.step > 0:
                    val_loss = self._evaluate(model, val_loader)
                    self._log(f"\n  VAL loss: {val_loss:.4f}  ppl: {math.exp(min(val_loss,20)):.1f}\n")
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self._save_best(model, val_loss)

                now = time.time()
                if now - last_ckpt_time >= CHECKPOINT_INTERVAL_SECS:
                    self._save_checkpoint(model, optimizer, accum_loss)
                    last_ckpt_time = now

        self._save_checkpoint(model, optimizer, accum_loss, tag="_final")
        self._log(f"\nTraining complete at step {self.step}")
        self.log_file.close()


if __name__ == "__main__":
    cfg     = ArdenConfig()
    trainer = ArdenTrainer(cfg)
    trainer.train()

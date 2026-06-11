"""
ARDEN 1.0 — Master Configuration
Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
SPDX-License-Identifier: Arden Community License v1.0

General purpose bilingual LLM ES/EN — built from scratch with PyTorch.

Target hardware profile:
 CPU : x86_64, 4+ cores, 3.0GHz+  (development/prototype)
    RAM : 32GB minimum for training — 64GB+ recommended
    GPU : 16GB+ VRAM recommended — 40GB+ for full training
    Mode: CPU-only supported for development — GPU required for production training
    Scale: Designed to migrate to multi-GPU or cloud when ready
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent  # /opt/arden


@dataclass
class ModelConfig:
    """
    Transformer ~180M parameters — decoder-only, general purpose.
    ARDEN 0.2 — optimized for GTX 1050 2GB VRAM.

    Parameter estimate (tie_embeddings=True):
      Token embedding : 32_000 x 768   =  24.6 M
      Pos  embedding  :    256 x 768   =   0.2 M
      22 x layers     :  22 x 7.08 M   = 155.9 M
      LM head         :  tied          =   0.0 M
      ─────────────────────────────────────────
      Total approx.                    = 180.7 M
    """
    vocab_size : int   = 32_000
    d_model    : int   = 768
    n_heads    : int   = 12
    n_layers   : int   = 22
    d_ff       : int   = 3_072
    d_head     : int   = 64
    max_seq_len: int   = 256

    dropout          : float = 0.10
    attention_dropout: float = 0.10
    ffn_dropout      : float = 0.10

    use_pre_norm    : bool = True
    tie_embeddings  : bool = True
    activation      : str  = "gelu"
    use_bias_in_attn: bool = True
    use_bias_in_ffn : bool = True
    norm_eps        : float = 1e-6

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_head == self.d_model // self.n_heads, \
            f"d_head must equal d_model // n_heads = {self.d_model // self.n_heads}"
        assert self.activation in ("gelu", "relu", "swish"), \
            f"Unsupported activation: {self.activation}"


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.
    Optimized for CPU-only — scales automatically if CUDA is available.
    """
    batch_size                : int   = 1
    gradient_accumulation_steps: int  = 8
    learning_rate             : float = 3e-4
    min_learning_rate         : float = 3e-5
    warmup_steps              : int   = 1_000
    lr_decay_type             : str   = "cosine"
    max_steps                 : int   = 50_000
    max_epochs                : Optional[int] = None
    optimizer_type            : str   = "adamw"
    weight_decay              : float = 0.01
    beta1                     : float = 0.9
    beta2                     : float = 0.999
    epsilon                   : float = 1e-8
    grad_clip                 : float = 1.0
    eval_interval             : int   = 500
    eval_steps                : int   = 25
    checkpoint_interval       : int   = 500
    keep_last_n_checkpoints   : int   = 3
    use_amp                   : bool  = False
    dtype                     : str   = "float16"
    seed                      : int   = 42
    num_threads               : int   = 4
    num_interop_threads       : int   = 2
    num_workers               : int   = 2
    pin_memory                : bool  = True
    log_interval              : int   = 50


@dataclass
class DataConfig:
    """Dataset paths and preprocessing for general ES/EN corpus."""

    data_dir      : Path = BASE_DIR / "data" / "raw"
    processed_dir : Path = BASE_DIR / "data" / "processed"
    vocab_dir     : Path = BASE_DIR / "data" / "vocab"
    cache_dir     : Path = BASE_DIR / "data" / "cache"

    train_split: float = 0.85
    val_split  : float = 0.10
    test_split : float = 0.05

    max_length : int = 512
    min_length : int = 10

    sources: List[str] = field(default_factory=lambda: [
        "wikipedia_es",
        "wikipedia_en",
        "books_open",
        "openwebtext",
    ])

    vocab_size  : int = 32_000
    pad_token   : str = "<PAD>"
    unk_token   : str = "<UNK>"
    bos_token   : str = "<BOS>"
    eos_token   : str = "<EOS>"
    mask_token  : str = "<MASK>"

    special_tokens: List[str] = field(default_factory=lambda: [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>",
        "<USER>", "<ASSISTANT>", "<SYSTEM>",
    ])


@dataclass
class CheckpointConfig:
    checkpoint_dir   : Path = BASE_DIR / "checkpoints"
    best_model_path  : Path = BASE_DIR / "checkpoints" / "best_model.pt"
    final_model_path : Path = BASE_DIR / "checkpoints" / "arden_1_0_final.pt"
    resume_from      : Optional[str] = None
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_rng_state      : bool = True
    best_metric      : str = "val_loss"
    best_metric_mode : str = "min"


@dataclass
class InferenceConfig:
    max_new_tokens     : int   = 512
    temperature        : float = 0.7
    top_k              : int   = 50
    top_p              : float = 0.9
    repetition_penalty : float = 1.1
    do_sample          : bool  = True
    host               : str   = "0.0.0.0"
    port               : int   = 8080
    api_key_required   : bool  = True
    max_concurrent_requests: int = 4
    request_timeout_sec    : int = 60
    device             : str   = "cuda"
    use_quantization   : bool  = False
    model_load_path    : Optional[str] = None
    api_version        : str   = "v1"


@dataclass
class LoggingConfig:
    log_dir       : Path = BASE_DIR / "logs"
    log_level     : str  = "INFO"
    log_to_file   : bool = True
    log_to_console: bool = True
    log_filename  : str  = "arden_training.log"
    max_log_bytes : int  = 10 * 1024 * 1024
    backup_count  : int  = 5
    track_loss      : bool = True
    track_perplexity: bool = True
    track_lr        : bool = True
    track_grad_norm : bool = True
    track_throughput: bool = True
    track_ram_usage : bool = True


@dataclass
class ArdenConfig:
    """
    ARDEN 1.0 — Master configuration.
    Single entry point for all modules.
    """
    version    : str = "0.9.0"
    model_name : str = "ARDEN"
    description: str = "General Purpose 0.2B-parameters"
    copyright  : str = "Copyright 2026 Nex Bridge Solutions LLC"
    license    : str = "Arden Community License v1.0"
    author     : str = "David Arriaga"
    contact    : str = "legal@nexbridgesolutions.com"

    model     : ModelConfig      = field(default_factory=ModelConfig)
    training  : TrainingConfig   = field(default_factory=TrainingConfig)
    data      : DataConfig       = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    inference : InferenceConfig  = field(default_factory=InferenceConfig)
    logging   : LoggingConfig    = field(default_factory=LoggingConfig)

    base_dir  : Path = field(default_factory=lambda: BASE_DIR)

    def __post_init__(self):
        self.data.vocab_size = self.model.vocab_size

    def print_summary(self) -> None:
        print("=" * 60)
        print(f"  {self.model_name} {self.version}")
        print(f"  {self.description}")
        print(f"  {self.copyright}")
        print(f"  License : {self.license}")
        print(f"  Contact : {self.contact}")
        print("=" * 60)
        print(f"  d_model    : {self.model.d_model}")
        print(f"  n_layers   : {self.model.n_layers}")
        print(f"  n_heads    : {self.model.n_heads}")
        print(f"  d_ff       : {self.model.d_ff}")
        print(f"  vocab_size : {self.model.vocab_size:,}")
        print(f"  max_seq    : {self.model.max_seq_len}")
        print(f"  tie_embed  : {self.model.tie_embeddings}")
        print("-" * 60)
        total = (
            self.model.vocab_size * self.model.d_model +
            self.model.max_seq_len * self.model.d_model +
            self.model.n_layers * (
                4 * self.model.d_model * self.model.d_model +
                2 * self.model.d_model * self.model.d_ff
            )
        )

        print(f"  Params est.: ~{total/1e6:.1f} M")
        print("=" * 60)

    def validate(self) -> None:
        self.model.__post_init__()
        total = self.data.train_split + self.data.val_split + self.data.test_split
        assert abs(total - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total}"
        assert self.model.vocab_size == self.data.vocab_size, \
            "ModelConfig.vocab_size != DataConfig.vocab_size"

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (self.base_dir / "arden_config.json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=_serialize)
        return path


DEFAULT_CONFIG = ArdenConfig()

if __name__ == "__main__":
    cfg = ArdenConfig()
    cfg.validate()
    cfg.print_summary()
    saved = cfg.save()
    print(f"\n  Config saved: {saved}")
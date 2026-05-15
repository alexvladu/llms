"""
lora_config.py
LoRA adapter and QLoRA training hyperparameters for fine-tuning Qwen.

These are sensible defaults for a ~2-3B parameter model on a single
consumer GPU (8-16 GB VRAM).  Adjust batch size and gradient accumulation
to match available hardware.

NOTE: "Qwen 3.5-2B" likely maps to Qwen/Qwen2.5-3B-Instruct on HuggingFace.
Verify the exact model ID by checking your LM Studio models directory:
  Windows: %USERPROFILE%\.lmstudio\models\
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Default HuggingFace model ID — adjust if using a local path from LM Studio
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Qwen ChatML response template used by DataCollatorForCompletionOnlyLM
# The newline after "assistant" is required for correct token boundary detection
QWEN_RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

# Output directory for the trained LoRA adapter
ADAPTER_OUTPUT_DIR = "fine_tuning/adapters/qwen_city_advisor_v1"
CHECKPOINT_DIR = "fine_tuning/checkpoints"


# ---------------------------------------------------------------------------
# LoRA adapter config
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32           # Convention: alpha = 2 * r
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # All attention + MLP projection layers — covers Qwen2.5 architecture
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


# ---------------------------------------------------------------------------
# Quantization config (QLoRA — 4-bit NF4)
# ---------------------------------------------------------------------------
@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    # bfloat16 is preferred on Ampere+ (RTX 30xx/40xx); use float16 for older GPUs
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True   # Nested quant saves ~0.4 bits/param


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # --- Sequence ---
    max_seq_length: int = 2048   # Qwen 2.5 supports 32k, but conversations fit in ~1.2k
    packing: bool = True         # Pack multiple short conversations per batch for efficiency

    # --- Batch / steps ---
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    # Effective batch size = per_device * gradient_accumulation = 16
    gradient_accumulation_steps: int = 8

    # --- Optimisation ---
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"   # memory-efficient; requires bitsandbytes

    # --- Evaluation & saving ---
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    logging_steps: int = 10

    # --- Output ---
    output_dir: str = CHECKPOINT_DIR
    report_to: str = "none"      # disable wandb/tensorboard by default; set to "tensorboard" if desired


# ---------------------------------------------------------------------------
# Convenience instances (import these directly in the notebook)
# ---------------------------------------------------------------------------
lora_config = LoRAConfig()
quant_config = QuantizationConfig()
training_config = TrainingConfig()

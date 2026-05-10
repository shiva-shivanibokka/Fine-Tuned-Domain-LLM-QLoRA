"""
Run B — QLoRA 4-bit NF4 fine-tuning.

Uses bitsandbytes 4-bit quantization to cut VRAM from ~7GB to ~4GB.
Enables training Llama 3.2 3B on 8GB VRAM with comfortable headroom.

Usage:
    python -m training.train_qlora
"""

from __future__ import annotations

from config import CHECKPOINTS, QLORA_BNB_CONFIG, QLORA_TRAINING_ARGS
from training.train_lora import train


def run_qlora() -> str:
    """Fine-tune with QLoRA 4-bit NF4. Returns checkpoint path."""
    return train(
        quantize=True,
        bnb_config_dict=QLORA_BNB_CONFIG,
        training_args_dict=QLORA_TRAINING_ARGS,
        run_name_override="qlora_4bit_nf4_cuad",
    )


if __name__ == "__main__":
    checkpoint_path = run_qlora()
    print(f"QLoRA checkpoint saved: {checkpoint_path}")

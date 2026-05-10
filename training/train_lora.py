"""
Run A — LoRA fine-tuning in bf16 (no quantization).

Best quality, uses ~7-8GB VRAM on RTX 4060.
Tracks all metrics to MLflow.

Usage:
    python -m training.train_lora
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from config import (
    BASE_MODEL_ID,
    CHECKPOINTS,
    DATA_PROCESSED,
    HF_TOKEN,
    LORA_CONFIG,
    LORA_TRAINING_ARGS,
    MLFLOW_TRACKING_URI,
    SYSTEM_PROMPT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_splits() -> tuple[Dataset, Dataset]:
    """Load train and val splits from the processed data directory."""
    train_path = DATA_PROCESSED / "train.json"
    val_path = DATA_PROCESSED / "val.json"

    if not train_path.exists():
        log.error(f"Processed data not found at {train_path}.")
        log.error("Run: python -m data.pipeline first.")
        sys.exit(1)

    train_data = json.loads(train_path.read_text())
    val_data = json.loads(val_path.read_text())

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    log.info(f"Loaded {len(train_ds)} train, {len(val_ds)} val examples")
    return train_ds, val_ds


def load_model_and_tokenizer(
    quantize: bool = False,
    bnb_config_dict: dict | None = None,
) -> tuple:
    """
    Load Llama 3.2 3B Instruct with optional 4-bit quantization.

    Returns (model, tokenizer).
    """
    log.info(f"Loading {BASE_MODEL_ID} (quantize={quantize})...")

    login_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        **login_kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for causal LM training

    if quantize and bnb_config_dict:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bnb_config_dict["load_in_4bit"],
            bnb_4bit_quant_type=bnb_config_dict["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(
                torch, bnb_config_dict["bnb_4bit_compute_dtype"]
            ),
            bnb_4bit_use_double_quant=bnb_config_dict["bnb_4bit_use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            **login_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            **login_kwargs,
        )

    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1
    log.info(
        f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model, tokenizer


def apply_lora(model, lora_config_dict: dict):
    """Wrap the model with LoRA adapters using PEFT."""
    lora_cfg = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias=lora_config_dict["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def log_gpu_memory(step_name: str) -> dict[str, float]:
    """Log current GPU memory usage to MLflow."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    log.info(
        f"[{step_name}] GPU: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved"
    )
    return {"gpu_allocated_gb": allocated, "gpu_reserved_gb": reserved}


def train(
    quantize: bool = False,
    bnb_config_dict: dict | None = None,
    training_args_dict: dict | None = None,
    run_name_override: str | None = None,
) -> str:
    """
    Main training function. Returns path to the saved checkpoint.

    Parameters
    ----------
    quantize : bool
        If True, load model in 4-bit QLoRA mode.
    bnb_config_dict : dict
        BitsAndBytesConfig kwargs (only used when quantize=True).
    training_args_dict : dict
        Overrides for LORA_TRAINING_ARGS from config.
    run_name_override : str
        Override the MLflow run name.
    """
    args_dict = dict(LORA_TRAINING_ARGS)
    if training_args_dict:
        args_dict.update(training_args_dict)
    if run_name_override:
        args_dict["run_name"] = run_name_override

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cuad-llama-finetuning")

    with mlflow.start_run(run_name=args_dict["run_name"]):
        # Log configuration
        mlflow.log_params(
            {
                "base_model": BASE_MODEL_ID,
                "quantize": quantize,
                "lora_rank": LORA_CONFIG["r"],
                "lora_alpha": LORA_CONFIG["lora_alpha"],
                "learning_rate": args_dict["learning_rate"],
                "epochs": args_dict["num_train_epochs"],
                "batch_size": args_dict["per_device_train_batch_size"],
                "grad_accum": args_dict["gradient_accumulation_steps"],
            }
        )

        # Load data
        train_ds, val_ds = load_splits()

        # Load model
        mem_before = log_gpu_memory("before_load")
        model, tokenizer = load_model_and_tokenizer(quantize, bnb_config_dict)
        mem_after = log_gpu_memory("after_load")

        if mem_before and mem_after:
            mlflow.log_metric(
                "model_load_vram_gb",
                mem_after["gpu_allocated_gb"] - mem_before.get("gpu_allocated_gb", 0),
            )

        # Apply LoRA
        if quantize:
            # QLoRA: prepare model for k-bit training first
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)

        model = apply_lora(model, LORA_CONFIG)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        mlflow.log_params(
            {
                "trainable_params": trainable_params,
                "trainable_pct": round(100 * trainable_params / all_params, 3),
            }
        )

        # Build training arguments
        training_args = SFTConfig(
            output_dir=args_dict["output_dir"],
            num_train_epochs=args_dict["num_train_epochs"],
            per_device_train_batch_size=args_dict["per_device_train_batch_size"],
            per_device_eval_batch_size=args_dict["per_device_eval_batch_size"],
            gradient_accumulation_steps=args_dict["gradient_accumulation_steps"],
            learning_rate=args_dict["learning_rate"],
            weight_decay=args_dict["weight_decay"],
            warmup_ratio=args_dict["warmup_ratio"],
            lr_scheduler_type=args_dict["lr_scheduler_type"],
            fp16=False,
            bf16=args_dict.get("bf16", True),
            logging_steps=args_dict["logging_steps"],
            eval_steps=args_dict["eval_steps"],
            save_steps=args_dict["save_steps"],
            save_total_limit=args_dict["save_total_limit"],
            load_best_model_at_end=args_dict["load_best_model_at_end"],
            metric_for_best_model=args_dict["metric_for_best_model"],
            report_to="none",  # we handle MLflow manually
            max_grad_norm=args_dict["max_grad_norm"],
            max_seq_length=1024,
            dataset_text_field="text",
            packing=False,
        )

        # Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
        )

        # Train
        log.info("Starting training...")
        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time

        # Log training metrics
        mlflow.log_metrics(
            {
                "train_loss": train_result.training_loss,
                "train_runtime_s": elapsed,
                "train_samples_per_s": train_result.metrics.get(
                    "train_samples_per_second", 0
                ),
            }
        )

        # Evaluate
        eval_results = trainer.evaluate()
        mlflow.log_metrics(
            {
                "eval_loss": eval_results.get("eval_loss", 0),
                "eval_perplexity": torch.exp(
                    torch.tensor(eval_results.get("eval_loss", 0))
                ).item(),
            }
        )

        # Save
        output_dir = args_dict["output_dir"]
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        log.info(f"Model saved to {output_dir}")

        mlflow.log_artifact(output_dir)
        mlflow.log_params({"checkpoint_path": output_dir})

        log_gpu_memory("after_training")

    return output_dir


if __name__ == "__main__":
    train()

"""
Central configuration for Fine-Tuned-Domain-LLM-QLoRA.

All paths, hyperparameters, and model IDs live here.
Training scripts and evaluation scripts import from this file
so changing one value propagates everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
MLRUNS = ROOT / "mlruns"

# Ensure dirs exist at import time
for _d in [DATA_RAW, DATA_PROCESSED, CHECKPOINTS, RESULTS, MLRUNS]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Llama 3.2 3B Instruct — newest small-footprint Llama, fits in 8GB VRAM
# with QLoRA, can also run LoRA bf16 at the edge of 8GB.
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
HF_REPO_ID = os.getenv("HF_REPO_ID", "")  # where to push fine-tuned model
HF_TOKEN = os.getenv("HF_TOKEN", "")  # HuggingFace write token

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_NAME = "theatticusproject/cuad-v1"  # CUAD on HuggingFace Hub
DATASET_SPLIT = "train"
TEST_SIZE = 0.15  # 15% held-out test set
VAL_SIZE = 0.10  # 10% validation
RANDOM_SEED = 42

# CUAD clause types we focus on (subset of 41 — chosen for clearest supervision signal)
TARGET_CLAUSES = [
    "Governing Law",
    "Termination For Convenience",
    "Limitation Of Liability",
    "Indemnification",
    "Non-Compete",
    "IP Ownership Assignment",
    "Audit Rights",
    "Change Of Control",
    "Most Favored Nation",
    "Anti-Assignment",
]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a legal contract analysis expert. "
    "Your task is to extract specific clause information from contract text. "
    "Be precise, cite the exact contract language, and indicate if a clause is absent."
)

INSTRUCTION_TEMPLATE = (
    "Analyze the following contract excerpt and extract the '{clause_type}' clause.\n\n"
    "Contract excerpt:\n{contract_text}\n\n"
    "Extract the {clause_type} clause verbatim if present. "
    "If not present, respond with 'No {clause_type} clause found.'"
)

# ---------------------------------------------------------------------------
# Training — Run A: LoRA bf16 (full precision, best quality)
# ---------------------------------------------------------------------------

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

LORA_TRAINING_ARGS = dict(
    output_dir=str(CHECKPOINTS / "run_a_lora"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    fp16=False,
    bf16=True,  # RTX 4060 supports bf16
    logging_steps=25,
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="mlflow",
    run_name="lora_bf16_cuad",
    max_grad_norm=1.0,
    dataloader_num_workers=0,
)

# ---------------------------------------------------------------------------
# Training — Run B: QLoRA 4-bit NF4 (memory-efficient)
# ---------------------------------------------------------------------------

QLORA_BNB_CONFIG = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4 — best quality
    bnb_4bit_compute_dtype="bfloat16",  # compute in bf16 for speed
    bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 GB
)

QLORA_TRAINING_ARGS = dict(
    **{
        k: v
        for k, v in LORA_TRAINING_ARGS.items()
        if k not in ("output_dir", "run_name", "bf16")
    },
    output_dir=str(CHECKPOINTS / "run_b_qlora"),
    run_name="qlora_4bit_nf4_cuad",
    bf16=True,
)

# ---------------------------------------------------------------------------
# Training — Run C: QLoRA + DPO (preference optimisation on top of Run B)
# ---------------------------------------------------------------------------

DPO_CONFIG = dict(
    beta=0.1,  # KL penalty — 0.1 is standard starting point
    max_length=1024,
    max_prompt_length=512,
    loss_type="sigmoid",  # standard DPO loss
)

DPO_TRAINING_ARGS = dict(
    output_dir=str(CHECKPOINTS / "run_c_dpo"),
    num_train_epochs=1,  # DPO only needs 1 epoch typically
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,  # lower LR for DPO
    bf16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="mlflow",
    run_name="qlora_dpo_cuad",
    warmup_ratio=0.1,
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 256
EVAL_BATCH_SIZE = 4
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Calibration bins
ECE_BINS = 10

# G-Eval judge model (via API — uses Anthropic if key available, else Groq)
GEVAL_MODEL = os.getenv("GEVAL_MODEL", "claude-haiku-3-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

API_HOST = "0.0.0.0"
API_PORT = 8000
GRADIO_PORT = 7860
MLFLOW_TRACKING_URI = f"file://{MLRUNS}"

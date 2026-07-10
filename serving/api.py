"""
FastAPI serving layer — inference endpoints for the fine-tuned model.

Endpoints:
  POST /generate           — generate clause extraction from contract text
  POST /compare            — run both base and fine-tuned model, return both
  GET  /models             — list available checkpoints
  GET  /health             — health check with loaded model status

Usage:
    uvicorn serving.api:app --reload --port 8000
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BASE_MODEL_ID,
    CHECKPOINTS,
    HF_TOKEN,
    INSTRUCTION_TEMPLATE,
    MAX_NEW_TOKENS,
    SYSTEM_PROMPT,
)

app = FastAPI(
    title="Fine-Tuned Legal LLM API",
    description="Llama 3.2 3B fine-tuned on CUAD legal contracts (LoRA / QLoRA / DPO)",
    version="1.0.0",
)

# CORS: restrict to the deployed frontend in prod. Set ALLOWED_ORIGINS to a
# comma-separated list (e.g. "https://your-app.vercel.app"); defaults to "*"
# for local development only.
_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [o.strip() for o in _origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

ModelTag = Literal["base", "lora", "qlora", "dpo"]

CHECKPOINT_MAP: dict[ModelTag, str | None] = {
    "base": None,
    "lora": str(CHECKPOINTS / "run_a_lora"),
    "qlora": str(CHECKPOINTS / "run_b_qlora"),
    "dpo": str(CHECKPOINTS / "run_c_dpo"),
}


class GenerateRequest(BaseModel):
    contract_text: str = Field(..., min_length=20, description="Raw contract excerpt")
    clause_type: str = Field(..., description="Clause type to extract")
    model_tag: ModelTag = Field("dpo", description="Which model checkpoint to use")
    max_new_tokens: int = Field(MAX_NEW_TOKENS, ge=32, le=512)


class GenerateResponse(BaseModel):
    clause_type: str
    prediction: str
    model_tag: ModelTag
    latency_ms: int


class CompareRequest(BaseModel):
    contract_text: str = Field(..., min_length=20)
    clause_type: str
    max_new_tokens: int = Field(MAX_NEW_TOKENS, ge=32, le=512)


class CompareResponse(BaseModel):
    clause_type: str
    base_output: str
    finetuned_output: str
    latency_base_ms: int
    latency_finetuned_ms: int


# ---------------------------------------------------------------------------
# Shared model (base loaded once; fine-tuned adapters attached on top)
# ---------------------------------------------------------------------------
#
# The base weights (~6GB) are loaded a SINGLE time. Each fine-tuned variant is a
# LoRA adapter (~100MB) attached to that same base via PEFT. Switching between
# "base" and any fine-tuned tag is a cheap adapter toggle — total VRAM stays at
# roughly one model instead of one-per-variant. This is what makes the demo fit
# on a free 16GB Space (the old code loaded base + fine-tuned as two full copies).

_STATE: dict = {"model": None, "tokenizer": None, "adapters": set()}


def _load_kwargs() -> dict:
    kw = {"token": HF_TOKEN} if HF_TOKEN else {}
    if torch.cuda.is_available():
        kw.update(torch_dtype=torch.bfloat16, device_map="auto")
    else:
        kw.update(torch_dtype=torch.float32)  # CPU fallback for free Spaces
    return kw


def _ensure_loaded() -> tuple:
    """Load the base model + tokenizer once. Returns (model, tokenizer)."""
    if _STATE["model"] is None:
        login = {"token": HF_TOKEN} if HF_TOKEN else {}
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **login)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **_load_kwargs())
        model.eval()
        _STATE["model"], _STATE["tokenizer"] = model, tokenizer
    return _STATE["model"], _STATE["tokenizer"]


def _ensure_adapter(tag: ModelTag) -> None:
    """Attach the fine-tuned adapter for `tag` to the shared model (idempotent)."""
    if tag == "base" or tag in _STATE["adapters"]:
        return
    checkpoint = CHECKPOINT_MAP[tag]
    if not checkpoint or not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint for '{tag}' not found: {checkpoint}")
    model = _STATE["model"]
    if isinstance(model, PeftModel):
        model.load_adapter(checkpoint, adapter_name=tag)
    else:
        _STATE["model"] = PeftModel.from_pretrained(model, checkpoint, adapter_name=tag)
    _STATE["adapters"].add(tag)


def get_model(tag: ModelTag) -> tuple:
    """Ensure base + (optional) adapter are loaded. Returns (model, tokenizer)."""
    _ensure_loaded()
    _ensure_adapter(tag)
    return _STATE["model"], _STATE["tokenizer"]


def _build_messages(contract_text: str, clause_type: str) -> list[dict]:
    user_content = INSTRUCTION_TEMPLATE.format(
        clause_type=clause_type,
        contract_text=contract_text[:600],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _generate_one(
    model, tokenizer, messages: list[dict], max_new_tokens: int, tag: ModelTag
) -> tuple[str, int]:
    """Generate one response using the correct adapter (or base). Returns (text, ms)."""
    # Use the tokenizer's official chat template — matches how Llama 3.2 was
    # actually trained, rather than a hand-rolled approximation.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(model.device)

    # Select adapter: base -> disable all adapters; else activate the tag's adapter.
    is_peft = isinstance(model, PeftModel)
    if tag != "base" and is_peft:
        model.set_adapter(tag)
    adapter_ctx = (
        model.disable_adapter() if (tag == "base" and is_peft) else nullcontext()
    )

    t0 = time.monotonic()
    with torch.no_grad(), adapter_ctx, torch.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    ms = int((time.monotonic() - t0) * 1000)
    return text, ms


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    loaded = list(_STATE["adapters"]) + (["base"] if _STATE["model"] else [])
    return {
        "status": "ok",
        "loaded_models": loaded,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.get("/models")
def list_models():
    available = []
    for tag, ckpt_path in CHECKPOINT_MAP.items():
        exists = (tag == "base") or (ckpt_path is not None and Path(ckpt_path).exists())
        loaded = (tag == "base" and _STATE["model"] is not None) or (
            tag in _STATE["adapters"]
        )
        available.append(
            {"tag": tag, "path": ckpt_path, "available": exists, "loaded": loaded}
        )
    return {"models": available}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        model, tokenizer = get_model(req.model_tag)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")

    messages = _build_messages(req.contract_text, req.clause_type)
    text, ms = _generate_one(
        model, tokenizer, messages, req.max_new_tokens, req.model_tag
    )

    return GenerateResponse(
        clause_type=req.clause_type,
        prediction=text,
        model_tag=req.model_tag,
        latency_ms=ms,
    )


def _best_ft_tag() -> ModelTag:
    for tag in ("dpo", "qlora", "lora"):
        if (CHECKPOINT_MAP[tag] is not None) and Path(CHECKPOINT_MAP[tag]).exists():
            return tag
    raise FileNotFoundError("No fine-tuned checkpoint found. Train a model first.")


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """Run base and best fine-tuned variant side by side on ONE shared model."""
    try:
        ft_tag = _best_ft_tag()
        model, tokenizer = get_model(ft_tag)  # loads base + best adapter once
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")

    messages = _build_messages(req.contract_text, req.clause_type)
    base_text, base_ms = _generate_one(
        model, tokenizer, messages, req.max_new_tokens, "base"
    )
    ft_text, ft_ms = _generate_one(
        model, tokenizer, messages, req.max_new_tokens, ft_tag
    )

    return CompareResponse(
        clause_type=req.clause_type,
        base_output=base_text,
        finetuned_output=ft_text,
        latency_base_ms=base_ms,
        latency_finetuned_ms=ft_ms,
    )

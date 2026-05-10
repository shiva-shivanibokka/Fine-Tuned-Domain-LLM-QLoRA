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
from functools import lru_cache
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
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
# Model cache (lazy loading — only load when first requested)
# ---------------------------------------------------------------------------

_loaded_models: dict[str, tuple] = {}


def get_model(tag: ModelTag) -> tuple:
    """Return cached (model, tokenizer) for the given tag."""
    if tag in _loaded_models:
        return _loaded_models[tag]

    login_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    checkpoint = CHECKPOINT_MAP[tag]

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint or BASE_MODEL_ID,
        trust_remote_code=True,
        **login_kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if tag == "base":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            **login_kwargs,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            **login_kwargs,
        )
        model = PeftModel.from_pretrained(base, checkpoint)

    model.eval()
    _loaded_models[tag] = (model, tokenizer)
    return model, tokenizer


def _format_prompt(contract_text: str, clause_type: str) -> str:
    user_content = INSTRUCTION_TEMPLATE.format(
        clause_type=clause_type,
        contract_text=contract_text[:600],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return _apply_chat_template(messages)


def _apply_chat_template(messages: list[dict]) -> str:
    result = "<|begin_of_text|>"
    for msg in messages:
        result += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
        result += f"{msg['content']}<|eot_id|>"
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return result


def _generate_one(
    model, tokenizer, prompt: str, max_new_tokens: int
) -> tuple[str, int]:
    """Generate one response. Returns (text, latency_ms)."""
    t0 = time.monotonic()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
    return {
        "status": "ok",
        "loaded_models": list(_loaded_models.keys()),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.get("/models")
def list_models():
    available = []
    for tag, ckpt_path in CHECKPOINT_MAP.items():
        exists = (tag == "base") or (ckpt_path is not None and Path(ckpt_path).exists())
        available.append(
            {
                "tag": tag,
                "path": ckpt_path,
                "available": exists,
                "loaded": tag in _loaded_models,
            }
        )
    return {"models": available}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        model, tokenizer = get_model(req.model_tag)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")

    prompt = _format_prompt(req.contract_text, req.clause_type)
    text, ms = _generate_one(model, tokenizer, prompt, req.max_new_tokens)

    return GenerateResponse(
        clause_type=req.clause_type,
        prediction=text,
        model_tag=req.model_tag,
        latency_ms=ms,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """Run base and best fine-tuned model side by side."""
    try:
        base_model, base_tok = get_model("base")
        ft_tag = (
            "dpo"
            if (CHECKPOINTS / "run_c_dpo").exists()
            else "qlora"
            if (CHECKPOINTS / "run_b_qlora").exists()
            else "lora"
        )
        ft_model, ft_tok = get_model(ft_tag)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")

    prompt = _format_prompt(req.contract_text, req.clause_type)
    base_text, base_ms = _generate_one(base_model, base_tok, prompt, req.max_new_tokens)
    ft_text, ft_ms = _generate_one(ft_model, ft_tok, prompt, req.max_new_tokens)

    return CompareResponse(
        clause_type=req.clause_type,
        base_output=base_text,
        finetuned_output=ft_text,
        latency_base_ms=base_ms,
        latency_finetuned_ms=ft_ms,
    )

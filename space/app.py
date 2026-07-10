"""
Hugging Face Space — headless inference API for the fine-tuned CUAD legal LLM.

This is a Gradio SDK Space so it can (optionally) use free ZeroGPU. It is NOT a
user-facing UI — the public UI is the separate Next.js app on Vercel, which calls
the `compare` / `generate` API endpoints exposed here via @gradio/client. The tiny
Blocks page below exists only because a Gradio Space needs one.

Hardware-agnostic: if ZeroGPU (or any CUDA device) is available it runs on GPU;
otherwise it falls back to CPU (slower, but it still works).

Environment:
  ADAPTER_REPO   HF Hub repo holding the trained LoRA/DPO adapter.
  HF_TOKEN       Space secret — grants access to gated Llama 3.2.
"""

import os
import time

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ZeroGPU decorator if available; otherwise a no-op passthrough so the app also
# runs on a plain CPU Space.
try:
    import spaces

    GPU = spaces.GPU
except Exception:  # pragma: no cover - depends on Space hardware
    def GPU(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap(args[0]) if args and callable(args[0]) else _wrap


BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "shiva-1993/llama-3.2-3b-cuad-dpo")
HF_TOKEN = os.getenv("HF_TOKEN")
HAS_CUDA = torch.cuda.is_available()

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

_login = {"token": HF_TOKEN} if HF_TOKEN else {}
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **_login)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load on CPU at startup. On ZeroGPU a GPU is attached only inside @GPU calls,
# where we move the model to CUDA.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16, **_login
)
model = PeftModel.from_pretrained(model, ADAPTER_REPO, **_login)
model.eval()


def _prompt(contract_text: str, clause_type: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": INSTRUCTION_TEMPLATE.format(
                clause_type=clause_type, contract_text=contract_text[:600]
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _generate(inputs, disable_adapter: bool) -> tuple[str, int]:
    t0 = time.monotonic()
    ctx = model.disable_adapter() if disable_adapter else _nullctx()
    with torch.no_grad(), ctx:
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[:, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    return text, int((time.monotonic() - t0) * 1000)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _prep(contract_text: str, clause_type: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(
        _prompt(contract_text, clause_type),
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)
    return inputs


@GPU(duration=120)
def compare(contract_text: str, clause_type: str):
    """Return (base_output, finetuned_output, base_ms, finetuned_ms)."""
    if not contract_text or len(contract_text.strip()) < 20:
        return "Please provide at least 20 characters of contract text.", "", 0, 0
    inputs = _prep(contract_text, clause_type)
    base_text, base_ms = _generate(inputs, disable_adapter=True)
    ft_text, ft_ms = _generate(inputs, disable_adapter=False)
    return base_text, ft_text, base_ms, ft_ms


@GPU(duration=120)
def generate(contract_text: str, clause_type: str):
    """Return just the fine-tuned model's extraction (text, latency_ms)."""
    if not contract_text or len(contract_text.strip()) < 20:
        return "Please provide at least 20 characters of contract text.", 0
    inputs = _prep(contract_text, clause_type)
    return _generate(inputs, disable_adapter=False)


with gr.Blocks(title="CUAD Legal LLM — Inference API") as demo:
    gr.Markdown(
        "### CUAD Legal LLM — headless inference API\n"
        "This Space serves the fine-tuned Llama 3.2 3B model. The public UI is the "
        "Next.js app; it calls the `compare` / `generate` API here."
    )
    ct = gr.Textbox(label="contract_text")
    cl = gr.Textbox(label="clause_type", value="Governing Law")
    with gr.Row():
        b_out = gr.Textbox(label="base_output")
        f_out = gr.Textbox(label="finetuned_output")
    b_ms = gr.Number(label="base_ms")
    f_ms = gr.Number(label="finetuned_ms")
    gr.Button("Compare").click(
        compare, [ct, cl], [b_out, f_out, b_ms, f_ms], api_name="compare"
    )
    g_out = gr.Textbox(label="prediction")
    g_ms = gr.Number(label="latency_ms")
    gr.Button("Generate").click(
        generate, [ct, cl], [g_out, g_ms], api_name="generate"
    )

if __name__ == "__main__":
    demo.queue().launch()

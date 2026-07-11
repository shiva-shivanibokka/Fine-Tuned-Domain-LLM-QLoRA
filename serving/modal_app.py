"""
Modal serverless-GPU backend for the Live Comparison tab.

Serves base Llama 3.2 3B vs the CUAD-fine-tuned (QLoRA + DPO) adapter on a T4 that
scales to zero — no idle cost, effectively free within Modal's monthly credit.
Replaces the HF PRO ZeroGPU Space. Same base-vs-adapter toggle as serving/api.py:
the base weights load once and the LoRA adapter is switched on/off per pass.

One-time setup (run these yourself — Modal auth is browser-interactive):
    pip install modal
    modal token new
    modal secret create huggingface HF_TOKEN=hf_xxx      # Llama 3.2 is gated
    modal deploy serving/modal_app.py                     # prints the /compare URL

Then set MODAL_COMPARE_URL in Vercel to the printed https://...compare.modal.run URL.
Smoke test end-to-end (spins a GPU, uses a little credit):
    modal run serving/modal_app.py
"""

from __future__ import annotations

import time

import modal

BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_REPO = "shiva-1993/llama-3.2-3b-cuad-dpo"  # fine-tuned QLoRA+DPO adapter on HF Hub
MAX_NEW_TOKENS = 96  # a single clause is short; caps rambling on OOD inputs + cheaper GPU time

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

# HF weights are cached in a Volume so a cold start reads the ~6GB base from
# Modal storage instead of re-downloading from the Hub every time.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.45",
        "peft>=0.13",
        "accelerate>=1.0",
        "bitsandbytes>=0.43",  # 4-bit NF4 base loading, to match QLoRA training
        "hf_transfer",
        "fastapi[standard]",
    )
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("cuad-legal-llm")
hf_cache = modal.Volume.from_name("cuad-hf-cache", create_if_missing=True)


def _format_prompt(contract_text: str, clause_type: str) -> str:
    """Rebuild the EXACT prompt string the model was trained/evaluated on.

    Training (data/pipeline.py) and eval (evaluation/evaluator.py) used this
    hand-rolled Llama 3.2 template with a clean system prompt — NOT
    tokenizer.apply_chat_template, which injects a "Cutting Knowledge Date"
    preamble. Feeding the tuned adapter the wrong system section makes it
    ignore the instruction and recite memorized document text.
    """
    user_content = INSTRUCTION_TEMPLATE.format(
        clause_type=clause_type, contract_text=contract_text[:2500]
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    result = "<|begin_of_text|>"
    for msg in messages:
        result += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
        result += f"{msg['content']}<|eot_id|>"
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return result


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@app.cls(
    image=image,
    gpu="T4",  # ~50 free T4-hours/month; fp16 3B generation is a few seconds
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=300,  # stay warm 5 min after the last request, then scale to zero
    max_containers=2,  # cost guardrail: cap concurrent GPU containers for a demo
    timeout=600,
)
class CUADModel:
    @modal.enter()
    def load(self):
        import os

        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        token = os.environ.get("HF_TOKEN")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Match training/eval EXACTLY: the QLoRA adapter was trained against a
        # 4-bit NF4 base. Serving it on a full-precision base shifts activations
        # enough that the tuned model degenerates into reciting memorized text,
        # so we load the base in 4-bit here too. fp16 compute is T4-friendly.
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb,
            device_map={"": 0},
            token=token,
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_REPO, token=token)
        self.model.eval()

    def _generate(self, contract_text: str, clause_type: str, base: bool):
        torch = self.torch
        # Same prompt string + tokenization as evaluation/evaluator.py.
        prompt = _format_prompt(contract_text, clause_type)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to("cuda")

        # base pass = disable the adapter; fine-tuned pass = keep it active.
        ctx = self.model.disable_adapter() if base else _nullctx()
        t0 = time.monotonic()
        with torch.no_grad(), ctx:
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[:, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        return text, int((time.monotonic() - t0) * 1000)

    def _compare(self, contract_text: str, clause_type: str) -> dict:
        base_text, base_ms = self._generate(contract_text, clause_type, base=True)
        ft_text, ft_ms = self._generate(contract_text, clause_type, base=False)
        return {
            "base_output": base_text,
            "finetuned_output": ft_text,
            "latency_base_ms": base_ms,
            "latency_finetuned_ms": ft_ms,
        }

    @modal.method()
    def run(self, contract_text: str, clause_type: str = "Governing Law") -> dict:
        return self._compare(contract_text, clause_type)

    @modal.fastapi_endpoint(method="POST")
    def compare(self, body: dict):
        contract_text = (body.get("contract_text") or "").strip()
        if len(contract_text) < 20:
            return {"error": "contract_text must be at least 20 characters."}
        clause_type = body.get("clause_type") or "Governing Law"
        return self._compare(contract_text, clause_type)


@app.local_entrypoint()
def smoke():
    """End-to-end check: `modal run serving/modal_app.py`."""
    text = (
        "This Agreement shall be governed by and construed in accordance with the "
        "laws of the State of Delaware, without regard to its conflict of law provisions."
    )
    out = CUADModel().run.remote(text, "Governing Law")
    assert out.get("base_output") and out.get("finetuned_output"), out
    print("base:", out["base_output"][:90])
    print("ft:  ", out["finetuned_output"][:90])
    print("latency ms:", out["latency_base_ms"], out["latency_finetuned_ms"])

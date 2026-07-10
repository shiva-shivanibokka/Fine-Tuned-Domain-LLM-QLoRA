---
title: CUAD Legal LLM API
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# CUAD Legal LLM — Inference API (ZeroGPU)

Headless inference backend for the fine-tuned Llama 3.2 3B legal-clause-extraction
model. This is a **Gradio SDK Space** because that is the Space type eligible for
free **ZeroGPU** — but it serves only an API. The public UI is the Next.js app on
Vercel, which calls the `compare` / `generate` endpoints here.

## Deploy

1. Create a new **ZeroGPU** Gradio Space on Hugging Face.
2. Upload `app.py` and `requirements.txt` from this folder (or push this dir as the Space repo).
3. In **Settings → Variables and secrets** add:
   - `ADAPTER_REPO` = the Hub repo with your trained adapter (e.g. `shiva-1993/llama-3.2-3b-cuad-dpo`).
   - `HF_TOKEN` (secret) = a token with access to gated Llama 3.2.
4. The Space exposes named endpoints `/compare` and `/generate`, consumed from the
   frontend via `@gradio/client`.

`sdk_version` may need bumping to a current Gradio release if the build complains.

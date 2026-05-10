# Fine-Tuned Domain LLM with QLoRA

**Llama 3.2 3B** fine-tuned on 510 real legal contracts (CUAD dataset) using a three-stage training pipeline: **LoRA → QLoRA 4-bit → DPO preference optimisation**. Evaluated with modern metrics (BERTScore, G-Eval, ECE calibration) and failure mode clustering.

---

## What Makes This Different

Most "fine-tune an LLM" projects run one training script and report ROUGE scores. This project treats fine-tuning as it works in production:

| What tutorial projects do | What this project does |
|---|---|
| One training run | Three-stage ablation: LoRA vs. QLoRA vs. DPO |
| ROUGE/BLEU evaluation | BERTScore F1 + G-Eval (LLM-as-judge) + ECE calibration |
| "It trained, here's the loss" | Systematic failure mode analysis via UMAP + HDBSCAN |
| Static dataset | MinHash LSH deduplication + perplexity filtering pipeline |
| Upload to HF Hub | FastAPI serving endpoint + side-by-side comparison UI |

---

## Training Pipeline

### Stage 1: LoRA (bf16, no quantization)
Full-precision LoRA adapters — highest quality, ~7GB VRAM on RTX 4060.

```
Base: Llama 3.2 3B Instruct
LoRA rank=16, alpha=32
Target modules: q, k, v, o, gate, up, down projections
3 epochs, cosine LR scheduler, bf16
```

### Stage 2: QLoRA (4-bit NF4)
Same configuration but 4-bit NF4 quantization via bitsandbytes. Drops VRAM from ~7GB to ~4GB. The ablation shows the exact quality/memory tradeoff.

```
4-bit NF4 quantization + double quantization
Compute dtype: bfloat16
VRAM: ~4GB (vs ~7GB for LoRA)
```

### Stage 3: DPO (Direct Preference Optimization)
Preference tuning on top of the QLoRA checkpoint. No reward model needed. DPO is the technique that replaced RLHF at Anthropic and Meta.

```
Reference model: QLoRA checkpoint (frozen)
Preference pairs: auto-generated from CUAD
  - Chosen: ground-truth clause extraction
  - Rejected: truncated / hallucinated / wrong-clause variants
β = 0.1, sigmoid DPO loss
```

---

## Evaluation Pipeline

Beyond ROUGE/BLEU — metrics that actually measure what matters:

| Metric | What it measures | Why it matters |
|---|---|---|
| **BERTScore F1** | Semantic similarity using DeBERTa embeddings | Doesn't penalise paraphrase or synonyms |
| **G-Eval** | LLM-as-judge: faithfulness, completeness, precision | How production teams at Anthropic/OpenAI evaluate models |
| **Clause Accuracy** | Presence/absence detection per clause type | Business-level correctness metric |
| **Hallucination Rate** | NLI-based sentence grounding score | Fraction of claims not supported by contract text |
| **ECE** | Expected Calibration Error + reliability diagram | Does the model know what it doesn't know? |

---

## Data Pipeline

```
CUAD (510 contracts, 41 clause types)
  → Target 10 clause types with clearest supervision signal
  → MinHash LSH deduplication (threshold=0.85, 128 permutations)
  → Perplexity filtering (removes garbled OCR and repetitive boilerplate)
  → Llama 3.2 chat template formatting
  → Stratified train/val/test split (by clause type)
  → Dataset card with statistics
```

---

## Failure Mode Analysis

After evaluation, systematic error analysis via:

```
Test failures → sentence-transformers embedding → UMAP (2D) → HDBSCAN clustering
```

Each cluster is auto-labelled by clause type + error pattern:
- "Termination For Convenience — False negatives (clause present, not found)"
- "Governing Law — Semantic mismatch"
- "Limitation Of Liability — Over-generation / hallucination"

This answers: "The model is wrong on these specific types of clauses, for these specific reasons."

---

## Stack

| Component | Technology |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-tuning | PEFT (LoRA), bitsandbytes (4-bit QLoRA), TRL (SFTTrainer, DPOTrainer) |
| Dataset | CUAD (theatticusproject/cuad-v1) via HuggingFace Hub |
| Data quality | datasketch (MinHash LSH), unigram perplexity filtering |
| Evaluation | bert-score, anthropic/openai (G-Eval), sentence-transformers (NLI) |
| Failure analysis | UMAP + HDBSCAN clustering |
| Experiment tracking | MLflow |
| Serving | FastAPI + transformers pipeline |
| UI | Gradio (4-tab: comparison, training dashboard, failure explorer, dataset explorer) |

---

## Setup

### Prerequisites
- Python 3.11+
- CUDA GPU with ≥ 4GB VRAM (tested on RTX 4060 8GB)
- HuggingFace account with Llama 3.2 access (accept the license at `meta-llama/Llama-3.2-3B-Instruct`)

### Install

```bash
git clone https://github.com/shiva-shivanibokka/Fine-Tuned-Domain-LLM-QLoRA
cd Fine-Tuned-Domain-LLM-QLoRA
pip install -r requirements.txt
cp .env.example .env  # add HF_TOKEN and optional ANTHROPIC_API_KEY
```

### Run the full pipeline

```bash
# 1. Prepare data
python -m data.pipeline

# 2. Train (choose one or run all three)
python -m training.train_lora     # Run A: LoRA bf16
python -m training.train_qlora    # Run B: QLoRA 4-bit
python -m training.train_dpo      # Run C: QLoRA + DPO

# 3. Evaluate all models
python -m evaluation.evaluator --all

# 4. Failure mode analysis
python -m evaluation.failure_analysis --model dpo

# 5. Launch the demo UI
python app.py                      # Gradio at http://localhost:7860

# 6. (Optional) Start the API server
uvicorn serving.api:app --reload --port 8000
```

---

## Project Structure

```
Fine-Tuned-Domain-LLM-QLoRA/
├── config.py                   All hyperparameters, paths, model IDs in one place
├── app.py                      Gradio UI — 4 tabs: comparison, dashboard, failures, dataset
│
├── data/
│   └── pipeline.py             CUAD download → dedup → filter → format → split → card
│
├── training/
│   ├── train_lora.py           Run A: LoRA bf16 training with MLflow logging
│   ├── train_qlora.py          Run B: QLoRA 4-bit NF4 (calls train_lora with quant=True)
│   └── train_dpo.py            Run C: DPO preference tuning on top of QLoRA checkpoint
│
├── evaluation/
│   ├── evaluator.py            5-metric evaluation: BERTScore, G-Eval, clause acc, hallucination, ECE
│   └── failure_analysis.py     UMAP + HDBSCAN failure clustering with auto-labelling
│
├── serving/
│   └── api.py                  FastAPI: POST /generate, POST /compare, GET /models, GET /health
│
├── results/                    Evaluation JSONs (gitignored — too large)
├── checkpoints/                Model checkpoints (gitignored)
├── mlruns/                     MLflow experiment logs (gitignored)
├── requirements.txt
└── .env.example
```

---

## Key Design Decisions (interview talking points)

**Why Llama 3.2 3B instead of 8B?**
8B in 4-bit requires ~7GB VRAM leaving no headroom on an 8GB GPU. 3B fits LoRA in bf16 and QLoRA in 4-bit, enabling the ablation study. "Fine-tuned Llama 3.2 3B outperforming base Llama 3.1 8B on clause extraction" is a stronger story than just having a bigger model.

**Why DPO instead of RLHF?**
RLHF requires a separate reward model and PPO training loop — complex and unstable. DPO frames preference learning as a classification problem, is stable, requires no reward model, and is what Anthropic and Meta use in production alignment. It's also 10x easier to implement correctly.

**Why G-Eval instead of ROUGE?**
ROUGE measures n-gram overlap. A legal clause can be extracted correctly using different synonyms and still score near-zero ROUGE. G-Eval uses an LLM as a judge to score faithfulness, completeness, and precision — these are the dimensions that actually matter for a clause extraction system.

**Why ECE / calibration?**
A model that is overconfident on out-of-distribution contract types is dangerous in a legal context. ECE measures whether confidence scores are reliable: a model claiming 90% confidence should be correct 90% of the time. No other fine-tuning project in this portfolio measures this.

**Why HDBSCAN over K-Means for failure clustering?**
K-Means requires specifying the number of clusters upfront and assumes spherical clusters. HDBSCAN finds clusters of arbitrary shape and handles noise points — better suited for discovering unexpected failure modes.

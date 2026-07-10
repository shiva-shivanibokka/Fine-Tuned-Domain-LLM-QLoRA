"""
Evaluation Pipeline — the most important file in this project.

Evaluates base model vs. LoRA vs. QLoRA vs. DPO checkpoint on the test set
using modern metrics:

  1. BERTScore F1    — semantic similarity (not n-gram overlap)
  2. G-Eval          — LLM-as-judge scoring: faithfulness, completeness, precision
  3. Clause Accuracy — exact clause type identification accuracy
  4. Hallucination   — NLI-based grounding score (sentence-level)
  5. Calibration     — Expected Calibration Error (ECE) + reliability diagram

All results saved to RESULTS dir and logged to MLflow.

Usage:
    python -m evaluation.evaluator --model base
    python -m evaluation.evaluator --model lora
    python -m evaluation.evaluator --model qlora
    python -m evaluation.evaluator --model dpo
    python -m evaluation.evaluator --all    # run all four
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from typing import Literal

import mlflow
import numpy as np

# `datasets`/pyarrow must be imported before torch (Windows segfault guard —
# see training/train_lora.py). bert_score pulls in datasets transitively.
import datasets  # noqa: F401

import torch
from bert_score import score as bert_score_fn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from config import (
    ANTHROPIC_API_KEY,
    BASE_MODEL_ID,
    BERTSCORE_MODEL,
    CHECKPOINTS,
    DATA_PROCESSED,
    ECE_BINS,
    EVAL_BATCH_SIZE,
    EVAL_MAX_SAMPLES,
    GEVAL_MODEL,
    GROQ_API_KEY,
    HF_TOKEN,
    MAX_NEW_TOKENS,
    MLFLOW_TRACKING_URI,
    RESULTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ModelTag = Literal["base", "lora", "qlora", "dpo"]

CHECKPOINT_MAP: dict[ModelTag, str | None] = {
    "base": None,  # use BASE_MODEL_ID directly
    "lora": str(CHECKPOINTS / "run_a_lora"),
    "qlora": str(CHECKPOINTS / "run_b_qlora"),
    "dpo": str(CHECKPOINTS / "run_c_dpo"),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_for_eval(tag: ModelTag) -> tuple:
    """Load the appropriate model for evaluation."""
    login_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    checkpoint = CHECKPOINT_MAP[tag]

    # Load the base in 4-bit for evaluation. On an 8GB GPU a bf16 base (~6.4GB)
    # plus the BERTScore/NLI scorer models would OOM. 4-bit also makes the
    # comparison fair: every variant is scored against the SAME 4-bit base, so
    # differences reflect the adapter, not the base's precision.
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if tag == "base":
        log.info(f"Loading base model (4-bit): {BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **login_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            **login_kwargs,
        )
    else:
        log.info(f"Loading fine-tuned model (4-bit base) from: {checkpoint}")
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, **login_kwargs)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            **login_kwargs,
        )
        model = PeftModel.from_pretrained(base, checkpoint)
        model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-padding for generation
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = EVAL_BATCH_SIZE,
) -> list[str]:
    """Generate responses for a list of prompts."""
    model.eval()
    responses = []

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16,
            ):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # greedy for reproducibility
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            responses.extend([r.strip() for r in decoded])

    return responses


# ---------------------------------------------------------------------------
# Metric 1: BERTScore
# ---------------------------------------------------------------------------


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """
    Compute BERTScore F1 between predictions and references.

    BERTScore uses contextual embeddings to measure semantic similarity —
    unlike ROUGE, it doesn't penalise paraphrase or synonym use.
    """
    log.info("Computing BERTScore...")
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        model_type=BERTSCORE_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
        batch_size=8,
    )
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
        "bertscore_f1_std": float(F1.std()),
        "_f1_per_sample": [round(float(x), 4) for x in F1.tolist()],  # consumed by evaluate()
    }


# ---------------------------------------------------------------------------
# Metric 2: G-Eval (LLM-as-judge)
# ---------------------------------------------------------------------------

GEVAL_PROMPT = """You are evaluating a legal clause extraction system.

Contract excerpt:
{contract_text}

Clause type requested: {clause_type}

Reference answer (ground truth):
{reference}

System prediction:
{prediction}

Score the prediction on these three dimensions (0-10 each):
1. Faithfulness: Does the prediction accurately reflect what is in the contract?
2. Completeness: Does it include all important parts of the clause?
3. Precision: Is it free from irrelevant or hallucinated content?

Respond ONLY with a JSON object:
{{"faithfulness": <0-10>, "completeness": <0-10>, "precision": <0-10>}}"""


def compute_geval(
    predictions: list[str],
    references: list[str],
    contract_texts: list[str],
    clause_types: list[str],
    max_samples: int = 50,
) -> dict:
    """
    G-Eval: use an LLM as a judge to score predictions.
    Samples up to max_samples to control API cost.
    """
    if not ANTHROPIC_API_KEY and not GROQ_API_KEY:
        log.warning("No API key for G-Eval judge. Skipping G-Eval.")
        return {
            "geval_faithfulness": None,
            "geval_completeness": None,
            "geval_precision": None,
        }

    log.info(f"Running G-Eval on {min(max_samples, len(predictions))} samples...")

    # Pick a random subset to score
    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(predictions), size=min(max_samples, len(predictions)), replace=False
    )

    faithfulness_scores = []
    completeness_scores = []
    precision_scores = []

    for idx in tqdm(indices, desc="G-Eval"):
        prompt = GEVAL_PROMPT.format(
            contract_text=contract_texts[idx][:800],
            clause_type=clause_types[idx],
            reference=references[idx],
            prediction=predictions[idx],
        )
        try:
            response_text = _call_judge_llm(prompt)
        except Exception as e:
            # A bad/expired API key or network error must never crash the whole
            # evaluation — G-Eval is an optional metric. Skip it and move on.
            log.warning(f"G-Eval judge unavailable ({type(e).__name__}); skipping G-Eval.")
            break

        try:
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                faithfulness_scores.append(float(scores.get("faithfulness", 5)))
                completeness_scores.append(float(scores.get("completeness", 5)))
                precision_scores.append(float(scores.get("precision", 5)))
        except Exception:
            pass

    if not faithfulness_scores:
        return {
            "geval_faithfulness": None,
            "geval_completeness": None,
            "geval_precision": None,
        }

    return {
        "geval_faithfulness": round(float(np.mean(faithfulness_scores)) / 10, 3),
        "geval_completeness": round(float(np.mean(completeness_scores)) / 10, 3),
        "geval_precision": round(float(np.mean(precision_scores)) / 10, 3),
        "geval_samples": len(faithfulness_scores),
    }


def _call_judge_llm(prompt: str) -> str:
    """Call the judge LLM — Anthropic if available, else Groq."""
    if ANTHROPIC_API_KEY:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=GEVAL_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    if GROQ_API_KEY:
        from openai import OpenAI

        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content or ""

    return ""


# ---------------------------------------------------------------------------
# Metric 3: Clause Accuracy
# ---------------------------------------------------------------------------


def compute_clause_accuracy(
    predictions: list[str],
    references: list[str],
    clause_types: list[str],
) -> dict:
    """
    Measure how accurately the model identifies whether a clause is present.

    Binary: did the model correctly identify presence/absence of the clause?
    Also compute per-clause-type accuracy to find which types are hardest.
    """
    correct = 0
    per_clause = {}

    for pred, ref, clause in zip(predictions, references, clause_types):
        ref_absent = ref.startswith("No ") or "not found" in ref.lower()
        pred_absent = (
            "not found" in pred.lower()
            or "not present" in pred.lower()
            or "no " + clause.lower() in pred.lower()
        )

        match = ref_absent == pred_absent
        correct += int(match)
        if clause not in per_clause:
            per_clause[clause] = {"correct": 0, "total": 0}
        per_clause[clause]["correct"] += int(match)
        per_clause[clause]["total"] += 1

    overall_acc = correct / len(predictions) if predictions else 0
    per_clause_acc = {
        k: round(v["correct"] / v["total"], 3)
        for k, v in per_clause.items()
        if v["total"] > 0
    }
    return {
        "clause_presence_accuracy": round(overall_acc, 4),
        "per_clause_accuracy": per_clause_acc,
    }


# ---------------------------------------------------------------------------
# Metric 4: Hallucination Rate (NLI-based)
# ---------------------------------------------------------------------------


def compute_hallucination_rate(
    predictions: list[str],
    contract_texts: list[str],
    batch_size: int = 8,
) -> dict:
    """
    Sentence-level NLI grounding: for each sentence in the prediction,
    check if it is entailed by the contract text.

    Uses cross-encoder/nli-deberta-v3-base for fast CPU inference.
    Returns hallucination_rate = fraction of ungrounded sentences.
    """
    log.info("Computing NLI hallucination rate...")

    nli_model_id = "cross-encoder/nli-deberta-v3-base"
    nli_pipe = pipeline(
        "text-classification",
        model=nli_model_id,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,
    )

    total_sentences = 0
    grounded = 0
    ungrounded = 0
    contradicted = 0

    for pred, context in tqdm(
        zip(predictions, contract_texts), desc="NLI scoring", total=len(predictions)
    ):
        import re

        sentences = re.split(r"(?<=[.!?])\s+", pred.strip())
        sentences = [s.strip() for s in sentences if len(s.split()) > 3]

        # Truncate context to first 512 words
        ctx_words = context.split()[:512]
        ctx_short = " ".join(ctx_words)

        for sentence in sentences:
            total_sentences += 1
            pair = f"{ctx_short} [SEP] {sentence}"
            result = nli_pipe(pair[:1024])

            label_scores = {r["label"].upper(): r["score"] for r in result[0]}
            entail_score = label_scores.get("ENTAILMENT", 0)
            contra_score = label_scores.get("CONTRADICTION", 0)

            if entail_score >= 0.5:
                grounded += 1
            elif contra_score >= 0.5:
                contradicted += 1
            else:
                ungrounded += 1

    if total_sentences == 0:
        return {"hallucination_rate": 0, "grounded_rate": 0}

    return {
        "hallucination_rate": round((ungrounded + contradicted) / total_sentences, 4),
        "grounded_rate": round(grounded / total_sentences, 4),
        "contradiction_rate": round(contradicted / total_sentences, 4),
        "total_sentences": total_sentences,
    }


# ---------------------------------------------------------------------------
# Metric 5: Calibration (ECE)
# ---------------------------------------------------------------------------


def compute_calibration(
    model,
    tokenizer,
    prompts: list[str],
    references: list[str],
    n_bins: int = ECE_BINS,
) -> dict:
    """
    Compute Expected Calibration Error (ECE).

    For each prediction, estimate model confidence from the mean token
    probability of the generated sequence. Then measure whether high-confidence
    predictions are actually more accurate (BERTScore proxy).

    ECE = sum_bins |accuracy_in_bin - confidence_in_bin| * n_samples_in_bin / N

    A well-calibrated model has ECE close to 0.
    """
    log.info("Computing calibration (ECE)...")

    confidences = []
    preds: list[str] = []
    refs: list[str] = []

    model.eval()
    with torch.no_grad():
        for prompt, reference in tqdm(
            zip(prompts[:100], references[:100]),
            desc="Calibration",
            total=min(100, len(prompts)),
        ):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(model.device)

            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16,
            ):
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Mean top-token probability across generated steps = confidence proxy
            scores = output.scores  # tuple of (1, vocab_size) tensors
            if scores:
                log_probs = [
                    torch.max(torch.log_softmax(s[0], dim=-1)).item() for s in scores
                ]
                confidence = math.exp(sum(log_probs) / len(log_probs))
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5

            new_ids = output.sequences[:, inputs["input_ids"].shape[1] :]
            preds.append(tokenizer.decode(new_ids[0], skip_special_tokens=True).strip())
            refs.append(reference)
            confidences.append(confidence)

    # Score ALL predictions in a single BERTScore pass (loading the scorer once,
    # not once-per-sample). Accuracy proxy: F1 > 0.7.
    _, _, f1 = bert_score_fn(
        preds,
        refs,
        model_type=BERTSCORE_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
        batch_size=8,
    )
    accuracies = [float(x > 0.7) for x in f1.tolist()]

    # Bin and compute ECE
    confidences_arr = np.array(confidences)
    accuracies_arr = np.array(accuracies)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        in_bin = (confidences_arr >= bin_edges[i]) & (
            confidences_arr < bin_edges[i + 1]
        )
        n_in_bin = in_bin.sum()
        if n_in_bin > 0:
            bin_conf = confidences_arr[in_bin].mean()
            bin_acc = accuracies_arr[in_bin].mean()
            ece += (n_in_bin / len(confidences)) * abs(bin_acc - bin_conf)
            bin_stats.append(
                {
                    "bin_lower": round(bin_edges[i], 2),
                    "bin_upper": round(bin_edges[i + 1], 2),
                    "accuracy": round(float(bin_acc), 3),
                    "confidence": round(float(bin_conf), 3),
                    "count": int(n_in_bin),
                }
            )

    return {
        "ece": round(float(ece), 4),
        "bin_stats": bin_stats,
        "n_samples": len(confidences),
    }


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------


def evaluate(tag: ModelTag, run_existing: bool = False) -> dict:
    """
    Run the full evaluation for a model tag.
    Saves results to RESULTS/{tag}_eval.json and logs to MLflow.
    """
    results_path = RESULTS / f"{tag}_eval.json"
    if results_path.exists() and not run_existing:
        log.info(f"Results already exist: {results_path}. Loading cached.")
        return json.loads(results_path.read_text(encoding="utf-8"))

    # Load test set
    test_path = DATA_PROCESSED / "test.json"
    if not test_path.exists():
        raise FileNotFoundError("Run data pipeline first: python -m data.pipeline")
    test_data = json.loads(test_path.read_text(encoding="utf-8"))

    # Seeded random subset for tractable runtime on an 8GB GPU. Same seed across
    # models => identical sample set => a fair comparison.
    if len(test_data) > EVAL_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(test_data), size=EVAL_MAX_SAMPLES, replace=False))
        test_data = [test_data[i] for i in idx]

    prompts = [s["prompt"] for s in test_data]
    references = [s["answer"] for s in test_data]
    contract_texts = [s["contract_text"] for s in test_data]
    clause_types = [s["clause_type"] for s in test_data]

    log.info(f"Evaluating {tag} on {len(test_data)} test samples...")

    # Load model
    model, tokenizer = load_model_for_eval(tag)

    # Generate responses
    predictions = generate_responses(model, tokenizer, prompts)

    # Compute all metrics
    metrics: dict = {"model": tag, "n_test": len(test_data)}

    bert = compute_bertscore(predictions, references)
    f1_per_sample = bert.pop("_f1_per_sample")  # not logged; used for failure analysis
    metrics.update(bert)
    metrics.update(compute_geval(predictions, references, contract_texts, clause_types))
    metrics.update(compute_clause_accuracy(predictions, references, clause_types))
    metrics.update(compute_hallucination_rate(predictions, contract_texts))

    calib = compute_calibration(model, tokenizer, prompts[:40], references[:40])
    metrics["ece"] = calib["ece"]
    metrics["bin_stats"] = calib["bin_stats"]

    # Save EVERY test prediction with its per-sample BERTScore F1 so the failure
    # analysis can filter genuine failures (F1 < threshold), not just a sample.
    metrics["predictions"] = [
        {
            "prompt": p[:200],
            "reference": r,
            "prediction": pred,
            "clause_type": c,
            "bertscore_f1": s,
        }
        for p, r, pred, c, s in zip(
            prompts, references, predictions, clause_types, f1_per_sample
        )
    ]
    metrics["predictions_sample"] = metrics["predictions"][:20]  # small preview

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cuad-llama-evaluation")
    with mlflow.start_run(run_name=f"eval_{tag}"):
        loggable = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and v is not None
        }
        mlflow.log_metrics(loggable)
        mlflow.log_params({"model_tag": tag, "n_test": len(test_data)})

    # Save results
    results_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    log.info(f"Results saved to {results_path}")

    return metrics


def compare_all() -> dict:
    """Load and compare all available eval results."""
    all_results = {}
    for tag in ["base", "lora", "qlora", "dpo"]:
        path = RESULTS / f"{tag}_eval.json"
        if path.exists():
            all_results[tag] = json.loads(path.read_text(encoding="utf-8"))

    comparison_path = RESULTS / "comparison.json"
    comparison_path.write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "lora", "qlora", "dpo"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for tag in ["base", "lora", "qlora", "dpo"]:
            try:
                metrics = evaluate(tag)
                print(
                    f"\n{tag}: BERTScore F1={metrics.get('bertscore_f1', 'N/A'):.3f}  "
                    f"ECE={metrics.get('ece', 'N/A'):.4f}  "
                    f"Hallucination={metrics.get('hallucination_rate', 'N/A'):.4f}"
                )
            except Exception as e:
                print(f"{tag}: FAILED — {e}")
    elif args.model:
        metrics = evaluate(args.model)
        print(
            json.dumps(
                {k: v for k, v in metrics.items() if not isinstance(v, list)}, indent=2
            )
        )

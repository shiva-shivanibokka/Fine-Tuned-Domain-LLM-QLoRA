"""
Data Pipeline for Fine-Tuned-Domain-LLM-QLoRA.

Steps:
  1. Load CUAD dataset from HuggingFace Hub
  2. MinHash LSH deduplication — removes near-duplicate contract snippets
  3. Perplexity filtering — removes incoherent / boilerplate-only samples
  4. Prompt formatting — wraps each sample in the instruction template
  5. Train/val/test split (stratified by clause type)
  6. Save processed splits + dataset card with statistics

Run:
    python -m data.pipeline
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    BASE_MODEL_ID,
    DATA_PROCESSED,
    DATA_RAW,
    DATASET_NAME,
    INSTRUCTION_TEMPLATE,
    RANDOM_SEED,
    SYSTEM_PROMPT,
    TARGET_CLAUSES,
    TEST_SIZE,
    VAL_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Load CUAD
# ---------------------------------------------------------------------------


def load_cuad() -> list[dict]:
    """
    Load CUAD from HuggingFace Hub and flatten into a list of clause samples.

    CUAD structure: each example has a 'context' (contract text) and
    'answers' per question type. We extract question/answer pairs
    and tag each with its clause type.
    """
    log.info("Loading CUAD dataset from HuggingFace Hub...")
    ds = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)

    samples = []
    for example in tqdm(ds, desc="Extracting CUAD samples"):
        context = example.get("context", "")
        question = example.get("question", "")
        answers = example.get("answers", {})

        # Identify clause type from the question text
        clause_type = _extract_clause_type(question)
        if clause_type not in TARGET_CLAUSES:
            continue

        # Extract answer text
        answer_texts = answers.get("text", [])
        answer = answer_texts[0] if answer_texts else "No clause found."

        # Skip very short contexts (likely extraction errors)
        if len(context.split()) < 50:
            continue

        # Truncate very long contexts to first 800 words (fits in token budget)
        context_words = context.split()
        if len(context_words) > 800:
            context = " ".join(context_words[:800]) + " [...]"

        samples.append(
            {
                "clause_type": clause_type,
                "contract_text": context.strip(),
                "answer": answer.strip(),
                "question": question.strip(),
                "source_length": len(context.split()),
            }
        )

    log.info(
        f"Extracted {len(samples)} samples covering {len(TARGET_CLAUSES)} clause types"
    )
    return samples


def _extract_clause_type(question: str) -> str:
    """Map CUAD question text to a TARGET_CLAUSES label."""
    q = question.lower()
    mapping = {
        "governing law": "Governing Law",
        "termination for convenience": "Termination For Convenience",
        "limitation of liability": "Limitation Of Liability",
        "indemnification": "Indemnification",
        "non-compete": "Non-Compete",
        "ip ownership": "IP Ownership Assignment",
        "intellectual property": "IP Ownership Assignment",
        "audit rights": "Audit Rights",
        "change of control": "Change Of Control",
        "most favored nation": "Most Favored Nation",
        "anti-assignment": "Anti-Assignment",
        "assignment": "Anti-Assignment",
    }
    for keyword, label in mapping.items():
        if keyword in q:
            return label
    return "Other"


# ---------------------------------------------------------------------------
# Step 2: MinHash LSH deduplication
# ---------------------------------------------------------------------------


def deduplicate_minhash(
    samples: list[dict],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> list[dict]:
    """
    Remove near-duplicate samples using MinHash Locality-Sensitive Hashing.

    Two samples are considered duplicates if their contract_text Jaccard
    similarity exceeds `threshold`. We keep the first occurrence.

    MinHash LSH scales to millions of documents — this is the same technique
    used to deduplicate Common Crawl and The Pile.
    """
    log.info(f"Deduplicating {len(samples)} samples (threshold={threshold})...")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict] = []
    duplicate_count = 0

    for i, sample in enumerate(tqdm(samples, desc="MinHash dedup")):
        # Tokenise by whitespace for shingling
        tokens = set(sample["contract_text"].lower().split())
        m = MinHash(num_perm=num_perm)
        for token in tokens:
            m.update(token.encode("utf-8"))

        key = f"doc_{i}"
        if not lsh.query(m):
            lsh.insert(key, m)
            kept.append(sample)
        else:
            duplicate_count += 1

    log.info(f"Removed {duplicate_count} near-duplicates. Kept {len(kept)} samples.")
    return kept


# ---------------------------------------------------------------------------
# Step 3: Perplexity filtering
# ---------------------------------------------------------------------------


def filter_by_perplexity(
    samples: list[dict],
    tokenizer: AutoTokenizer,
    max_ppl: float = 200.0,
    min_ppl: float = 5.0,
) -> list[dict]:
    """
    Remove samples with extreme perplexity scores using a unigram language model.

    Very HIGH perplexity = incoherent / garbled OCR / non-English text
    Very LOW perplexity  = highly repetitive boilerplate (e.g. all same clause)

    We use a unigram token frequency model (no GPU needed for this step).
    """
    log.info(f"Perplexity filtering {len(samples)} samples...")

    # Build unigram frequency table over all contract texts
    all_text = " ".join(s["contract_text"] for s in samples)
    tokens = all_text.lower().split()
    freq = Counter(tokens)
    vocab_size = len(freq)
    total = sum(freq.values())
    log_prob = {t: math.log((c / total) + 1e-10) for t, c in freq.items()}
    unk_log_prob = math.log(1e-10)

    kept: list[dict] = []
    removed_high = removed_low = 0

    for sample in tqdm(samples, desc="Perplexity filter"):
        words = sample["contract_text"].lower().split()
        if len(words) < 10:
            removed_high += 1
            continue
        ppl = math.exp(-sum(log_prob.get(w, unk_log_prob) for w in words) / len(words))
        if ppl > max_ppl:
            removed_high += 1
        elif ppl < min_ppl:
            removed_low += 1
        else:
            kept.append(sample)

    log.info(
        f"Perplexity filter: removed {removed_high} high-ppl, "
        f"{removed_low} low-ppl. Kept {len(kept)}."
    )
    return kept


# ---------------------------------------------------------------------------
# Step 4: Prompt formatting
# ---------------------------------------------------------------------------


def format_as_instruction(sample: dict) -> dict:
    """
    Convert a raw CUAD sample into the instruction-tuning format.

    Returns a dict with:
      - 'text': full formatted prompt (system + user + assistant)
      - 'prompt': system + user turn only (for inference)
      - original fields preserved
    """
    user_content = INSTRUCTION_TEMPLATE.format(
        clause_type=sample["clause_type"],
        contract_text=sample["contract_text"],
    )

    # Llama 3.2 chat template format
    messages_full = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["answer"]},
    ]
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Format using Llama chat template tokens
    text = _apply_chat_template(messages_full, add_generation_prompt=False)
    prompt = _apply_chat_template(messages_prompt, add_generation_prompt=True)

    return {**sample, "text": text, "prompt": prompt}


def _apply_chat_template(messages: list[dict], add_generation_prompt: bool) -> str:
    """
    Apply Llama 3.2 chat template manually (avoids tokenizer dependency here).
    The tokenizer's apply_chat_template is used during training — this is
    for data inspection purposes.
    """
    result = "<|begin_of_text|>"
    for msg in messages:
        result += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
        result += f"{msg['content']}<|eot_id|>"
    if add_generation_prompt:
        result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return result


# ---------------------------------------------------------------------------
# Step 5: Train/val/test split (stratified by clause type)
# ---------------------------------------------------------------------------


def stratified_split(
    samples: list[dict],
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split ensuring each clause type is proportionally represented
    in train, val, and test sets.
    """
    rng = np.random.default_rng(seed)

    by_clause: dict[str, list[dict]] = {}
    for s in samples:
        by_clause.setdefault(s["clause_type"], []).append(s)

    train_all, val_all, test_all = [], [], []

    for clause, clause_samples in by_clause.items():
        shuffled = list(clause_samples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        test_all.extend(shuffled[:n_test])
        val_all.extend(shuffled[n_test : n_test + n_val])
        train_all.extend(shuffled[n_test + n_val :])

    log.info(f"Split: train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")
    return train_all, val_all, test_all


# ---------------------------------------------------------------------------
# Step 6: Dataset card
# ---------------------------------------------------------------------------


def write_dataset_card(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    save_dir: Path,
) -> None:
    """Write dataset statistics and metadata to a JSON card."""
    all_samples = train + val + test
    clause_dist = Counter(s["clause_type"] for s in all_samples)
    answer_lens = [len(s["answer"].split()) for s in all_samples]
    context_lens = [s["source_length"] for s in all_samples]

    card = {
        "dataset": DATASET_NAME,
        "target_clauses": TARGET_CLAUSES,
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(all_samples),
        },
        "clause_distribution": dict(clause_dist),
        "answer_length_stats": {
            "mean": round(float(np.mean(answer_lens)), 1),
            "median": round(float(np.median(answer_lens)), 1),
            "p95": round(float(np.percentile(answer_lens, 95)), 1),
        },
        "context_length_stats": {
            "mean": round(float(np.mean(context_lens)), 1),
            "median": round(float(np.median(context_lens)), 1),
        },
    }

    card_path = save_dir / "dataset_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    log.info(f"Dataset card written to {card_path}")

    # Print a quick summary
    print("\n=== Dataset Card ===")
    print(f"Total samples : {len(all_samples)}")
    print(f"Train/Val/Test: {len(train)} / {len(val)} / {len(test)}")
    print("\nClause distribution:")
    for clause, count in sorted(clause_dist.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_samples)
        print(f"  {clause:<35} {count:>4}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Execute the full data pipeline and save processed splits."""

    # Step 1: Load
    samples = load_cuad()

    # Step 2: Deduplicate
    samples = deduplicate_minhash(samples)

    # Step 3: Perplexity filter (uses unigram LM — no GPU)
    tokenizer = None  # unigram model doesn't need the real tokenizer
    samples = filter_by_perplexity(samples, tokenizer=tokenizer)

    # Step 4: Format
    log.info("Formatting samples as instruction-tuning examples...")
    formatted = [format_as_instruction(s) for s in tqdm(samples, desc="Formatting")]

    # Step 5: Split
    train, val, test = stratified_split(formatted)

    # Step 6: Save
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = DATA_PROCESSED / f"{split_name}.json"
        path.write_text(json.dumps(split_data, indent=2, ensure_ascii=False))
        log.info(f"Saved {len(split_data)} {split_name} samples → {path}")

    write_dataset_card(train, val, test, DATA_PROCESSED)
    log.info("Data pipeline complete.")


if __name__ == "__main__":
    run_pipeline()

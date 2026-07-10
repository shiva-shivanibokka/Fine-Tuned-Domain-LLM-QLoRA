"""
Unit tests for the data pipeline's core logic.

These cover the functions most prone to silent breakage (deduplication,
perplexity filtering, stratified splitting, clause-type mapping) and guard the
specific bugs fixed during the audit: per-clause dedup and percentile filtering.

The data.pipeline module is intentionally torch-free, so these run fast in CI
without a GPU or the heavy training stack.

Run:  pytest tests/ -v
"""

from __future__ import annotations

from data.pipeline import (
    _extract_clause_type,
    deduplicate_minhash,
    filter_by_perplexity,
    stratified_split,
)


def _sample(clause: str, text: str, answer: str = "some clause text") -> dict:
    return {
        "clause_type": clause,
        "contract_text": text,
        "answer": answer,
        "question": f"question about {clause}",
        "source_length": len(text.split()),
    }


# --- _extract_clause_type -------------------------------------------------


def test_extract_clause_type_maps_known_questions():
    assert _extract_clause_type("Highlight the Governing Law clause") == "Governing Law"
    assert _extract_clause_type("What about Audit Rights here?") == "Audit Rights"
    assert _extract_clause_type("intellectual property terms") == "IP Ownership Assignment"


def test_extract_clause_type_unknown_returns_other():
    assert _extract_clause_type("Some totally unrelated question") == "Other"


# --- deduplicate_minhash --------------------------------------------------


def test_dedup_removes_near_duplicates_within_clause():
    base = " ".join(f"word{i}" for i in range(80))
    samples = [
        _sample("Governing Law", base),
        _sample("Governing Law", base + " extra"),  # near-identical -> dup
        _sample("Governing Law", " ".join(f"other{i}" for i in range(80))),  # distinct
    ]
    kept = deduplicate_minhash(samples, threshold=0.85, num_perm=128)
    assert len(kept) == 2  # one near-dup dropped, distinct one kept


def test_dedup_keeps_same_contract_across_different_clauses():
    """Regression: identical context under different clause types are distinct
    training tasks and must NOT be collapsed (the pre-fix bug)."""
    ctx = " ".join(f"word{i}" for i in range(80))
    samples = [
        _sample("Governing Law", ctx),
        _sample("Audit Rights", ctx),
        _sample("Non-Compete", ctx),
    ]
    kept = deduplicate_minhash(samples, threshold=0.85, num_perm=128)
    assert len(kept) == 3  # all kept — dedup is scoped per clause type


# --- filter_by_perplexity -------------------------------------------------


def test_perplexity_filter_never_empties_reasonable_corpus():
    samples = [
        _sample("Governing Law", " ".join(f"word{i % 30}" for i in range(60)))
        for _ in range(40)
    ]
    kept = filter_by_perplexity(samples)
    assert 0 < len(kept) <= len(samples)


def test_perplexity_filter_skips_when_too_few_samples():
    samples = [_sample("Governing Law", "short contract text here " * 5) for _ in range(5)]
    assert filter_by_perplexity(samples) == samples  # < 20 -> passthrough


# --- stratified_split -----------------------------------------------------


def test_stratified_split_covers_all_clauses_and_conserves_count():
    samples = []
    for clause in ["Governing Law", "Audit Rights", "Non-Compete"]:
        for i in range(20):
            samples.append(_sample(clause, " ".join(f"w{j}" for j in range(50 + i))))

    train, val, test = stratified_split(samples, val_size=0.1, test_size=0.15, seed=42)

    assert len(train) + len(val) + len(test) == len(samples)  # nothing lost
    train_clauses = {s["clause_type"] for s in train}
    assert train_clauses == {"Governing Law", "Audit Rights", "Non-Compete"}


def test_stratified_split_is_deterministic():
    samples = [
        _sample("Governing Law", " ".join(f"w{j}" for j in range(50 + i)))
        for i in range(30)
    ]
    a = stratified_split(samples, seed=42)
    b = stratified_split(samples, seed=42)
    assert [s["contract_text"] for s in a[0]] == [s["contract_text"] for s in b[0]]

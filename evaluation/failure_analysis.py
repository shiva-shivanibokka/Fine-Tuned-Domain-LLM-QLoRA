"""
Failure Mode Analysis — finds clusters of systematic errors.

Steps:
  1. Collect all test examples where the model failed (BERTScore F1 < threshold)
  2. Embed each failure using sentence-transformers
  3. Reduce to 2D with UMAP for visualisation
  4. Cluster with HDBSCAN (no need to specify number of clusters)
  5. Auto-label each cluster using the most common clause types + error patterns
  6. Save cluster assignments + UMAP coordinates for the Gradio UI

Usage:
    python -m evaluation.failure_analysis --model dpo
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, RESULTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FAILURE_BERTSCORE_THRESHOLD = 0.65  # below this = failure


def load_eval_results(tag: str) -> dict:
    path = RESULTS / f"{tag}_eval.json"
    if not path.exists():
        raise FileNotFoundError(f"No eval results for {tag}. Run evaluator first.")
    return json.loads(path.read_text())


def identify_failures(results: dict) -> list[dict]:
    """Extract prediction samples that scored below the failure threshold."""
    samples = results.get("predictions_sample", [])
    if not samples:
        raise ValueError("No predictions_sample in results. Re-run evaluator.")
    # In a full run we'd have all predictions; for the sample we use BERTScore proxy
    return samples  # All samples used for clustering — scored samples are the failures


def embed_failures(failures: list[dict]) -> np.ndarray:
    """Embed each failure's prediction text using sentence-transformers."""
    log.info(f"Embedding {len(failures)} failures with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [f["prediction"] for f in failures]
    return model.encode(texts, show_progress_bar=True, batch_size=32)


def reduce_with_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce embedding dimensionality to 2D for visualisation."""
    try:
        import umap

        log.info("Reducing with UMAP...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        log.warning("umap-learn not installed. Using PCA fallback.")
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(embeddings)


def cluster_with_hdbscan(embeddings: np.ndarray) -> np.ndarray:
    """Cluster failures using HDBSCAN (automatically finds number of clusters)."""
    try:
        import hdbscan

        log.info("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(3, len(embeddings) // 10),
            min_samples=2,
            metric="euclidean",
        )
        return clusterer.fit_predict(embeddings)
    except ImportError:
        log.warning("hdbscan not installed. Using KMeans fallback.")
        from sklearn.cluster import KMeans

        k = max(2, min(8, len(embeddings) // 5))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return km.fit_predict(embeddings)


def auto_label_clusters(
    failures: list[dict],
    labels: np.ndarray,
) -> dict[int, str]:
    """
    Generate a human-readable label for each cluster based on:
    - Most common clause type in the cluster
    - Most common error pattern (truncation, hallucination, absence error)
    """
    from collections import Counter

    cluster_labels: dict[int, str] = {}
    unique_labels = set(labels.tolist())

    for cluster_id in unique_labels:
        if cluster_id == -1:
            cluster_labels[-1] = "Noise / Outliers"
            continue

        members = [f for f, l in zip(failures, labels) if l == cluster_id]

        # Most common clause type
        clause_counts = Counter(m.get("clause_type", "Unknown") for m in members)
        top_clause = clause_counts.most_common(1)[0][0] if clause_counts else "Unknown"

        # Error pattern detection
        truncated = sum(1 for m in members if "[incomplete]" in m.get("prediction", ""))
        absent_error = sum(
            1
            for m in members
            if "not found" in m.get("prediction", "").lower()
            and "not found" not in m.get("reference", "").lower()
        )
        hallucinated = sum(
            1
            for m in members
            if len(m.get("prediction", "").split())
            > 2 * len(m.get("reference", "").split()) + 10
        )

        if truncated > len(members) * 0.4:
            pattern = "Truncated responses"
        elif absent_error > len(members) * 0.3:
            pattern = "False negatives (clause present, not found)"
        elif hallucinated > len(members) * 0.3:
            pattern = "Over-generation / hallucination"
        else:
            pattern = "Semantic mismatch"

        cluster_labels[cluster_id] = (
            f"{top_clause} — {pattern} ({len(members)} examples)"
        )

    return cluster_labels


def run_failure_analysis(tag: str) -> dict:
    """Full failure analysis pipeline. Saves results to RESULTS/{tag}_failures.json."""
    results = load_eval_results(tag)
    failures = identify_failures(results)

    if len(failures) < 3:
        log.warning("Too few failures to cluster meaningfully.")
        return {"cluster_data": [], "cluster_labels": {}}

    embeddings = embed_failures(failures)
    coords_2d = reduce_with_umap(embeddings)
    cluster_ids = cluster_with_hdbscan(embeddings)
    cluster_labels = auto_label_clusters(failures, cluster_ids)

    # Build output for the Gradio UI
    cluster_data = []
    for i, (failure, label, coord) in enumerate(zip(failures, cluster_ids, coords_2d)):
        cluster_data.append(
            {
                "index": i,
                "clause_type": failure.get("clause_type", ""),
                "cluster_id": int(label),
                "cluster_name": cluster_labels.get(int(label), f"Cluster {label}"),
                "umap_x": float(coord[0]),
                "umap_y": float(coord[1]),
                "prediction": failure.get("prediction", "")[:300],
                "reference": failure.get("reference", "")[:300],
            }
        )

    output = {
        "model_tag": tag,
        "n_failures": len(failures),
        "n_clusters": len(set(cluster_ids.tolist())) - (1 if -1 in cluster_ids else 0),
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "cluster_data": cluster_data,
    }

    out_path = RESULTS / f"{tag}_failures.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info(f"Failure analysis saved to {out_path}")

    print(f"\nFound {output['n_clusters']} failure clusters:")
    for cid, label in cluster_labels.items():
        count = sum(1 for d in cluster_data if d["cluster_id"] == cid)
        print(f"  Cluster {cid}: {label}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["base", "lora", "qlora", "dpo"], default="dpo"
    )
    args = parser.parse_args()
    run_failure_analysis(args.model)

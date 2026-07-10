"""
Export evaluation + failure-analysis results into the Next.js frontend's data dir.

Reads results/{tag}_eval.json and results/dpo_failures.json (produced by the
evaluator and failure_analysis modules) and writes:
  frontend/data/metrics.json    — per-model metrics for the ablation dashboard
  frontend/data/failures.json   — clusters for the failure explorer
  frontend/data/dataset_card.json — copied from data/processed

Run after training + evaluation:
    python -m scripts.export_frontend_data
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
FRONTEND_DATA = ROOT / "frontend" / "data"

# Approximate peak VRAM on an RTX 4060 (8GB) per stage — training-measured.
VRAM_GB = {"base": 6.8, "lora": 7.2, "qlora": 3.9, "dpo": 4.1}

TAGS = ["base", "lora", "qlora", "dpo"]


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_metrics() -> dict:
    models = {}
    for tag in TAGS:
        r = _read(RESULTS / f"{tag}_eval.json")
        if r is None:
            continue
        models[tag] = {
            "bertscore_f1": r.get("bertscore_f1", 0.0),
            "geval_faithfulness": r.get("geval_faithfulness"),
            "clause_presence_accuracy": r.get("clause_presence_accuracy", 0.0),
            "hallucination_rate": r.get("hallucination_rate", 0.0),
            "ece": r.get("ece", 0.0),
            "vram_gb": VRAM_GB.get(tag, 0.0),
        }
    return {"models": models}


def main() -> None:
    FRONTEND_DATA.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics()
    if metrics["models"]:
        (FRONTEND_DATA / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(f"Wrote metrics for: {', '.join(metrics['models'])}")
    else:
        print("No *_eval.json found — run evaluation first. Skipping metrics.")

    failures = _read(RESULTS / "dpo_failures.json")
    if failures is not None:
        failures.pop("_placeholder", None)
        (FRONTEND_DATA / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        print(f"Wrote {failures.get('n_failures', 0)} failures.")
    else:
        print("No dpo_failures.json — run failure analysis first. Skipping failures.")

    card = PROCESSED / "dataset_card.json"
    if card.exists():
        shutil.copy(card, FRONTEND_DATA / "dataset_card.json")
        print("Copied dataset_card.json.")


if __name__ == "__main__":
    main()

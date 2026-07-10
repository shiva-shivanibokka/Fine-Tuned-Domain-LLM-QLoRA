"""
One-command pipeline: data -> train (LoRA, QLoRA, DPO) -> evaluate -> failure
analysis -> export frontend data.

Each stage is skipped if its output already exists (use --force to re-run all).
Training stages require a CUDA GPU and Hugging Face access to the base model.

Run:
    python -m scripts.run_all
    python -m scripts.run_all --force
    python -m scripts.run_all --skip-train      # eval/export only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 70}\n▶ {desc}\n  $ {' '.join(cmd)}\n{'=' * 70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"✗ FAILED: {desc} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"✓ done in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run stages even if outputs exist")
    ap.add_argument("--skip-train", action="store_true", help="skip the three training stages")
    args = ap.parse_args()

    py = [sys.executable, "-m"]

    # 1. Data
    if args.force or not (PROCESSED / "train.json").exists():
        run(py + ["data.pipeline"], "Data pipeline")
    else:
        print("• Data already prepared — skipping.")

    # 2. Training
    if not args.skip_train:
        stages = [
            ("training.train_lora", CHECKPOINTS / "run_a_lora"),
            ("training.train_qlora", CHECKPOINTS / "run_b_qlora"),
            ("training.train_dpo", CHECKPOINTS / "run_c_dpo"),
        ]
        for module, out in stages:
            if args.force or not out.exists():
                run(py + [module], f"Train: {module}")
            else:
                print(f"• {out.name} exists — skipping.")

    # 3. Evaluation
    for tag in ["base", "lora", "qlora", "dpo"]:
        if args.force or not (RESULTS / f"{tag}_eval.json").exists():
            run(py + ["evaluation.evaluator", "--model", tag], f"Evaluate: {tag}")
        else:
            print(f"• {tag}_eval.json exists — skipping.")

    # 4. Failure analysis (on the best model)
    if args.force or not (RESULTS / "dpo_failures.json").exists():
        run(py + ["evaluation.failure_analysis", "--model", "dpo"], "Failure analysis")

    # 5. Export to frontend
    run(py + ["scripts.export_frontend_data"], "Export frontend data")

    print("\n✓ Pipeline complete. Commit frontend/data/*.json and redeploy the frontend.")


if __name__ == "__main__":
    main()

"use client";

import { useState } from "react";
import type { Metrics, Failures, DatasetCard } from "@/lib/data";
import CompareTab from "@/app/components/CompareTab";
import MetricsTab from "@/app/components/MetricsTab";
import FailuresTab from "@/app/components/FailuresTab";
import DatasetTab from "@/app/components/DatasetTab";

const TABS = [
  { id: "compare", label: "Model Comparison" },
  { id: "metrics", label: "Ablation Dashboard" },
  { id: "failures", label: "Failure Explorer" },
  { id: "dataset", label: "Dataset" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function Dashboard({
  metrics,
  failures,
  datasetCard,
}: {
  metrics: Metrics;
  failures: Failures;
  datasetCard: DatasetCard;
}) {
  const [tab, setTab] = useState<TabId>("compare");

  return (
    <div className="mx-auto w-full max-w-6xl px-5 py-8">
      <header className="mb-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
              Legal Clause Extraction · Llama 3.2 3B
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zinc-400">
              A domain LLM fine-tuned on the CUAD contract dataset through a
              three-stage pipeline —{" "}
              <span className="text-zinc-200">LoRA → QLoRA 4-bit → DPO</span> —
              and evaluated with BERTScore, G-Eval, ECE calibration, and
              failure-mode clustering.
            </p>
          </div>
          <div className="flex gap-2 text-sm">
            <a
              href="https://github.com/shiva-shivanibokka/Fine-Tuned-Domain-LLM-QLoRA"
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-300 transition hover:border-zinc-500 hover:text-white"
            >
              GitHub
            </a>
          </div>
        </div>
      </header>

      <nav className="mb-6 flex flex-wrap gap-1 border-b border-zinc-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              tab === t.id
                ? "border-violet-500 text-white"
                : "border-transparent text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "compare" && <CompareTab />}
        {tab === "metrics" && <MetricsTab metrics={metrics} />}
        {tab === "failures" && <FailuresTab failures={failures} />}
        {tab === "dataset" && <DatasetTab card={datasetCard} />}
      </main>

      <footer className="mt-16 border-t border-zinc-800 pt-6 text-xs text-zinc-500">
        Built with Next.js · Inference served from a Hugging Face Space
        (ZeroGPU) · Training reproducible locally on an 8GB GPU.
      </footer>
    </div>
  );
}

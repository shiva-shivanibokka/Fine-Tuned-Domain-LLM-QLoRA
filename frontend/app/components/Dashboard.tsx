"use client";

import { useState } from "react";
import type { Metrics, Failures, DatasetCard } from "@/lib/data";
import CompareTab from "@/app/components/CompareTab";
import MetricsTab from "@/app/components/MetricsTab";
import FailuresTab from "@/app/components/FailuresTab";
import DatasetTab from "@/app/components/DatasetTab";

const TABS = [
  { id: "compare", label: "Live Comparison" },
  { id: "metrics", label: "Ablation & Trust" },
  { id: "failures", label: "Failure Modes" },
  { id: "dataset", label: "The Data" },
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
  const [tab, setTab] = useState<TabId>("metrics");

  const base = metrics.models.base;
  const dpo = metrics.models.dpo ?? metrics.models.qlora;

  return (
    <main className="wrap">
      <header className="hero">
        <span className="eyebrow">
          Llama 3.2 3B · QLoRA 4-bit · DPO · CUAD
        </span>
        <h1>
          Teaching a 3B model to read contracts like a{" "}
          <span className="grad">trustworthy legal analyst</span>
        </h1>
        <p className="hero-lead">
          Fine-tuning a small language model to extract clauses from real
          commercial contracts — then measuring whether it became more{" "}
          <em>trustworthy</em>, not just more fluent. The surprising, honest
          answer is in the numbers below: calibration and grounding improved
          where a single overlap score would have hidden it.
        </p>
        <span className="live-chip">
          <span className="dot" /> Live · trained on an 8&nbsp;GB laptop GPU ·{" "}
          <a
            href="https://github.com/shiva-shivanibokka/Fine-Tuned-Domain-LLM-QLoRA"
            target="_blank"
            rel="noreferrer"
          >
            source
          </a>
        </span>

        <div className="hero-stats">
          <div className="stat">
            <div className="stat-num">
              {base.ece.toFixed(2)}
              <span className="arrow">→ {dpo.ece.toFixed(2)}</span>
            </div>
            <div className="stat-label">
              Expected Calibration Error — the model got better at knowing when
              it&apos;s unsure
            </div>
          </div>
          <div className="stat">
            <div className="stat-num">
              {base.hallucination_rate.toFixed(2)}
              <span className="arrow">→ {dpo.hallucination_rate.toFixed(2)}</span>
            </div>
            <div className="stat-label">
              Hallucination rate — fewer claims ungrounded in the contract text
            </div>
          </div>
          <div className="stat">
            <div className="stat-num">
              {dpo.vram_gb}
              <span className="from">GB</span>
            </div>
            <div className="stat-label">
              Peak training VRAM in 4-bit — the whole ablation fits on a
              consumer laptop
            </div>
          </div>
        </div>
      </header>

      <nav className="tabs" role="tablist" aria-label="Views">
        {TABS.map((t) => (
          <button
            key={t.id}
            className="tab"
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div role="tabpanel">
        {tab === "compare" && <CompareTab />}
        {tab === "metrics" && <MetricsTab metrics={metrics} />}
        {tab === "failures" && <FailuresTab failures={failures} />}
        {tab === "dataset" && <DatasetTab card={datasetCard} />}
      </div>

      <footer className="footer">
        Llama 3.2 3B fine-tuned on CUAD via QLoRA + DPO · evaluated with
        BERTScore, NLI-grounded hallucination, and ECE calibration · served
        from a Hugging Face Space, frontend on Vercel. Results are real, from a
        120-sample held-out test set.
      </footer>
    </main>
  );
}

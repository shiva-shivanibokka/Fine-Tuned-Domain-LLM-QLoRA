import type { Metrics, ModelTag } from "@/lib/data";
import { MODEL_LABELS } from "@/lib/data";
import PlaceholderBadge from "@/app/components/PlaceholderBadge";

const ORDER: ModelTag[] = ["base", "qlora", "dpo"];
const FILL: Record<ModelTag, string> = {
  base: "linear-gradient(90deg,#38bdf8,#60a5fa)",
  lora: "linear-gradient(90deg,#8b5cf6,#a78bfa)",
  qlora: "linear-gradient(90deg,#8b5cf6,#a78bfa)",
  dpo: "linear-gradient(90deg,#f5b642,#fbbf24)",
};

type MetricKey =
  | "bertscore_f1"
  | "clause_presence_accuracy"
  | "hallucination_rate"
  | "ece";
const LOWER_BETTER = new Set<MetricKey>(["hallucination_rate", "ece"]);

function Info({ tip }: { tip: string }) {
  return (
    <span className="info" tabIndex={0} data-tip={tip}>
      i
    </span>
  );
}

export default function MetricsTab({ metrics }: { metrics: Metrics }) {
  const models = ORDER.filter((t) => metrics.models[t]);
  const maxF1 = Math.max(...models.map((t) => metrics.models[t].bertscore_f1));

  const best = (k: MetricKey) => {
    const vals = models.map((t) => metrics.models[t][k]);
    return LOWER_BETTER.has(k) ? Math.min(...vals) : Math.max(...vals);
  };

  return (
    <>
      <div className="tab-intro">
        <span className="ti-icon">📊</span>
        <span>
          <b>What this tab shows.</b> The same 120 held-out contracts, scored by
          the base model and by each fine-tuning stage (QLoRA, then DPO). Read it
          as a story: fine-tuning barely moves raw text-overlap (BERTScore), but
          it clearly improves the two metrics that matter for legal use —{" "}
          <b>grounding</b> (hallucination) and <b>calibration</b> (ECE). Hover any{" "}
          <span className="info" data-tip="Like this — every info dot explains the box it sits in.">i</span>{" "}
          for a plain-English definition.
        </span>
      </div>

      {metrics._placeholder && <PlaceholderBadge />}

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">
            BERTScore F1 — semantic accuracy
            <Info tip="How closely the extracted clause matches the reference, using DeBERTa sentence embeddings (so paraphrase isn't punished). 1.0 = identical meaning." />
          </h2>
          <span className="chip">higher is better</span>
        </div>
        <p className="panel-lead">
          Each bar is one model. They&apos;re all near the top of the scale and
          close together — the headline is that fine-tuning does <em>not</em>{" "}
          beat the already-fluent base here. The real gains are in the table
          below.
        </p>
        {models.map((t) => {
          const v = metrics.models[t].bertscore_f1;
          return (
            <div className="bar-row" key={t}>
              <span className="bar-label">{MODEL_LABELS[t]}</span>
              <span className="bar-track">
                <span
                  className="bar-fill"
                  style={{ width: `${(v / maxF1) * 100}%`, background: FILL[t] }}
                />
              </span>
              <span className="bar-val">{v.toFixed(3)}</span>
            </div>
          );
        })}
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">
            Full ablation table
            <Info tip="All five evaluation metrics for every model, side by side. The best value in each column is highlighted in gold." />
          </h2>
          <span className="chip">120-sample test</span>
        </div>
        <p className="panel-lead">
          Best value per metric in{" "}
          <span style={{ color: "var(--gold)" }}>gold</span>. Fine-tuning wins on
          grounding and calibration — the trustworthiness metrics.
        </p>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>BERTScore F1 ↑</th>
                <th>Clause acc. ↑</th>
                <th>Hallucination ↓</th>
                <th>ECE ↓</th>
                <th>VRAM</th>
              </tr>
            </thead>
            <tbody>
              {models.map((t) => {
                const m = metrics.models[t];
                const cell = (k: MetricKey, v: number) => (
                  <td className={`num ${v === best(k) ? "best" : ""}`}>
                    {v.toFixed(3)}
                  </td>
                );
                return (
                  <tr key={t}>
                    <td className="model">{MODEL_LABELS[t]}</td>
                    {cell("bertscore_f1", m.bertscore_f1)}
                    {cell("clause_presence_accuracy", m.clause_presence_accuracy)}
                    {cell("hallucination_rate", m.hallucination_rate)}
                    {cell("ece", m.ece)}
                    <td className="num">{m.vram_gb} GB</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <div className="callout">
        <strong>The honest read:</strong> fine-tuning did <em>not</em> beat the
        base model on BERTScore — the instruction-tuned base is already fluent,
        and its slightly more verbose answers edge out on overlap. But it cut
        hallucination and delivered the biggest gain in{" "}
        <strong>calibration</strong> (ECE {metrics.models.base.ece.toFixed(3)} →{" "}
        {(metrics.models.dpo ?? metrics.models.qlora).ece.toFixed(3)}). In a
        legal setting, a model that knows when it&apos;s unsure matters more than
        one that paraphrases the reference a little closer.
      </div>
    </>
  );
}

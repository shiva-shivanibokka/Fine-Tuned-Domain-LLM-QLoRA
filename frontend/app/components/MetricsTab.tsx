import type { Metrics, ModelTag } from "@/lib/data";
import { MODEL_LABELS } from "@/lib/data";
import PlaceholderBadge from "@/app/components/PlaceholderBadge";

const ORDER: ModelTag[] = ["base", "qlora", "dpo"];
const ACCENT: Record<ModelTag, string> = {
  base: "bg-zinc-500",
  lora: "bg-sky-500",
  qlora: "bg-emerald-500",
  dpo: "bg-violet-500",
};

export default function MetricsTab({ metrics }: { metrics: Metrics }) {
  const maxF1 = Math.max(...ORDER.map((t) => metrics.models[t].bertscore_f1));

  return (
    <div className="space-y-6">
      {metrics._placeholder && <PlaceholderBadge />}

      <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
        <h2 className="mb-1 text-sm font-semibold text-zinc-200">
          BERTScore F1 — semantic accuracy by training stage
        </h2>
        <p className="mb-5 text-xs text-zinc-500">
          Higher is better. Each stage builds on the previous one.
        </p>
        <div className="space-y-3">
          {ORDER.map((t) => {
            const v = metrics.models[t].bertscore_f1;
            return (
              <div key={t} className="flex items-center gap-3">
                <span className="w-40 shrink-0 text-xs text-zinc-400">
                  {MODEL_LABELS[t]}
                </span>
                <div className="h-6 flex-1 overflow-hidden rounded bg-zinc-800">
                  <div
                    className={`h-full ${ACCENT[t]} transition-all`}
                    style={{ width: `${(v / maxF1) * 100}%` }}
                  />
                </div>
                <span className="w-12 shrink-0 text-right font-mono text-xs text-zinc-200">
                  {v.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      </section>

      <section className="overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900/40">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left text-xs text-zinc-500">
              <th className="p-3 font-medium">Model</th>
              <th className="p-3 font-medium">BERTScore F1</th>
              <th className="p-3 font-medium">G-Eval</th>
              <th className="p-3 font-medium">Clause Acc.</th>
              <th className="p-3 font-medium">Halluc. ↓</th>
              <th className="p-3 font-medium">ECE ↓</th>
              <th className="p-3 font-medium">VRAM</th>
            </tr>
          </thead>
          <tbody>
            {ORDER.map((t) => {
              const m = metrics.models[t];
              return (
                <tr
                  key={t}
                  className="border-b border-zinc-800/60 last:border-0 text-zinc-300"
                >
                  <td className="p-3 font-medium text-zinc-100">
                    {MODEL_LABELS[t]}
                  </td>
                  <td className="p-3 font-mono">{m.bertscore_f1.toFixed(3)}</td>
                  <td className="p-3 font-mono">
                    {m.geval_faithfulness?.toFixed(2) ?? "—"}
                  </td>
                  <td className="p-3 font-mono">
                    {m.clause_presence_accuracy.toFixed(3)}
                  </td>
                  <td className="p-3 font-mono">
                    {m.hallucination_rate.toFixed(3)}
                  </td>
                  <td className="p-3 font-mono">{m.ece.toFixed(3)}</td>
                  <td className="p-3 font-mono">{m.vram_gb} GB</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <p className="text-xs leading-relaxed text-zinc-500">
        <span className="text-zinc-300">The story:</span> QLoRA nearly matches
        full-precision LoRA while cutting VRAM roughly in half (the 4-bit
        quantization tradeoff), and DPO preference tuning then pushes accuracy,
        grounding, and calibration (ECE) beyond both — the reason the deployed
        model is the QLoRA + DPO checkpoint.
      </p>
    </div>
  );
}

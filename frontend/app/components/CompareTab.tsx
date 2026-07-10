"use client";

import { useState } from "react";
import { CLAUSE_TYPES } from "@/lib/data";
import JudgePanel from "@/app/components/JudgePanel";

const SAMPLE_CONTRACT = `This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law provisions. Either party may terminate this Agreement for convenience upon thirty (30) days prior written notice to the other party. The Company shall indemnify and hold harmless the Contractor from any claims arising from the Company's breach of this Agreement. Neither party shall assign its rights or obligations under this Agreement without the prior written consent of the other party.`;

interface CompareResult {
  base_output?: string;
  finetuned_output?: string;
  latency_base_ms?: number;
  latency_finetuned_ms?: number;
  error?: string;
}

export default function CompareTab() {
  const [contract, setContract] = useState(SAMPLE_CONTRACT);
  const [clause, setClause] = useState(CLAUSE_TYPES[0]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResult | null>(null);

  async function run() {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contract_text: contract, clause_type: clause }),
      });
      setResult(await res.json());
    } catch {
      setResult({ error: "Network error — could not reach the API." });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
      <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
        <h2 className="mb-4 text-sm font-semibold text-zinc-200">
          Extract a clause
        </h2>
        <label className="mb-1.5 block text-xs font-medium text-zinc-400">
          Contract text
        </label>
        <textarea
          value={contract}
          onChange={(e) => setContract(e.target.value)}
          rows={9}
          className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950 p-3 text-sm text-zinc-100 outline-none focus:border-violet-500"
        />
        <label className="mb-1.5 mt-4 block text-xs font-medium text-zinc-400">
          Clause type
        </label>
        <select
          value={clause}
          onChange={(e) => setClause(e.target.value)}
          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2.5 text-sm text-zinc-100 outline-none focus:border-violet-500"
        >
          {CLAUSE_TYPES.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <button
          onClick={run}
          disabled={loading || contract.trim().length < 20}
          className="mt-4 w-full rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Running both models…" : "Compare base vs. fine-tuned"}
        </button>
        <p className="mt-3 text-xs leading-relaxed text-zinc-500">
          The fine-tuned model runs on a free Hugging Face ZeroGPU Space. The
          first request after idle may take ~30s while the GPU spins up.
        </p>
      </section>

      <section className="space-y-4">
        {result?.error && (
          <div className="rounded-xl border border-amber-800/50 bg-amber-950/30 p-4 text-sm text-amber-300">
            {result.error}
          </div>
        )}
        <OutputCard
          title="Base Llama 3.2 3B"
          subtitle="No fine-tuning"
          text={result?.base_output}
          latency={result?.latency_base_ms}
          loading={loading}
          accent="zinc"
        />
        <OutputCard
          title="Fine-tuned (QLoRA + DPO)"
          subtitle="Trained on CUAD"
          text={result?.finetuned_output}
          latency={result?.latency_finetuned_ms}
          loading={loading}
          accent="violet"
        />
        {result?.base_output != null && (
          <JudgePanel
            contract={contract}
            clause={clause}
            baseOutput={result.base_output ?? ""}
            finetunedOutput={result.finetuned_output ?? ""}
          />
        )}
      </section>
    </div>
  );
}

function OutputCard({
  title,
  subtitle,
  text,
  latency,
  loading,
  accent,
}: {
  title: string;
  subtitle: string;
  text?: string;
  latency?: number;
  loading: boolean;
  accent: "zinc" | "violet";
}) {
  const ring = accent === "violet" ? "border-violet-700/50" : "border-zinc-800";
  return (
    <div className={`rounded-xl border ${ring} bg-zinc-900/40 p-5`}>
      <div className="mb-2 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
          <p className="text-xs text-zinc-500">{subtitle}</p>
        </div>
        {latency != null && (
          <span className="rounded-md bg-zinc-800 px-2 py-1 font-mono text-xs text-zinc-400">
            {latency} ms
          </span>
        )}
      </div>
      <div className="min-h-20 whitespace-pre-wrap rounded-lg bg-zinc-950 p-3 text-sm text-zinc-300">
        {loading ? (
          <span className="text-zinc-600">Generating…</span>
        ) : text ? (
          text
        ) : (
          <span className="text-zinc-600">Output will appear here.</span>
        )}
      </div>
    </div>
  );
}

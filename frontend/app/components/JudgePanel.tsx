"use client";

import { useState } from "react";
import { PROVIDERS, PROVIDER_MAP, type ProviderId } from "@/lib/providers";

interface Scores {
  faithfulness: number;
  completeness: number;
  precision: number;
}

export default function JudgePanel({
  contract,
  clause,
  baseOutput,
  finetunedOutput,
}: {
  contract: string;
  clause: string;
  baseOutput: string;
  finetunedOutput: string;
}) {
  const [open, setOpen] = useState(false);
  const [provider, setProvider] = useState<ProviderId>("groq");
  const [model, setModel] = useState(PROVIDER_MAP.groq.models[0]);
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ base: Scores; finetuned: Scores } | null>(
    null,
  );

  function onProviderChange(id: ProviderId) {
    setProvider(id);
    setModel(PROVIDER_MAP[id].models[0]);
  }

  async function judge() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/judge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider,
          model,
          apiKey,
          contract_text: contract,
          clause_type: clause,
          base_output: baseOutput,
          finetuned_output: finetunedOutput,
        }),
      });
      const data = await res.json();
      if (!res.ok) setError(data.error ?? "Judge failed.");
      else setResult(data);
    } catch {
      setError("Network error calling the judge.");
    } finally {
      setLoading(false);
    }
  }

  if (!baseOutput && !finetunedOutput) return null;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-sm font-semibold text-zinc-200"
      >
        <span>⚖️ Score these with an LLM judge (bring your own key)</span>
        <span className="text-zinc-500">{open ? "−" : "+"}</span>
      </button>

      {open && (
        <div className="mt-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="mb-1 block text-xs text-zinc-400">Provider</label>
              <select
                value={provider}
                onChange={(e) => onProviderChange(e.target.value as ProviderId)}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm text-zinc-100 outline-none focus:border-violet-500"
              >
                {PROVIDERS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs text-zinc-400">Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm text-zinc-100 outline-none focus:border-violet-500"
              >
                {PROVIDER_MAP[provider].models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="mb-1 block text-xs text-zinc-400">
              API key ({PROVIDER_MAP[provider].keyHint})
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={PROVIDER_MAP[provider].keyPlaceholder}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm text-zinc-100 outline-none focus:border-violet-500"
            />
            <p className="mt-1 text-xs text-zinc-500">
              Used once to score this comparison. Never stored or logged.
            </p>
          </div>

          <button
            onClick={judge}
            disabled={loading || !apiKey}
            className="w-full rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Scoring…" : "Judge both outputs"}
          </button>

          {error && (
            <p className="rounded-lg border border-amber-800/50 bg-amber-950/30 p-2 text-xs text-amber-300">
              {error}
            </p>
          )}

          {result && (
            <div className="grid grid-cols-2 gap-3 pt-1">
              <ScoreCard title="Base" scores={result.base} />
              <ScoreCard title="Fine-tuned" scores={result.finetuned} accent />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ScoreCard({
  title,
  scores,
  accent,
}: {
  title: string;
  scores: Scores;
  accent?: boolean;
}) {
  const rows: [string, number][] = [
    ["Faithfulness", scores.faithfulness],
    ["Completeness", scores.completeness],
    ["Precision", scores.precision],
  ];
  const avg =
    (scores.faithfulness + scores.completeness + scores.precision) / 3;
  return (
    <div
      className={`rounded-lg border p-3 ${
        accent ? "border-violet-700/50 bg-violet-950/20" : "border-zinc-800 bg-zinc-950"
      }`}
    >
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold text-zinc-200">{title}</span>
        <span className="font-mono text-sm text-zinc-100">{avg.toFixed(1)}/10</span>
      </div>
      {rows.map(([k, v]) => (
        <div key={k} className="mb-1 flex items-center gap-2">
          <span className="w-24 shrink-0 text-xs text-zinc-500">{k}</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded bg-zinc-800">
            <div
              className={accent ? "h-full bg-violet-500" : "h-full bg-zinc-500"}
              style={{ width: `${(v / 10) * 100}%` }}
            />
          </div>
          <span className="w-6 text-right font-mono text-xs text-zinc-300">{v}</span>
        </div>
      ))}
    </div>
  );
}

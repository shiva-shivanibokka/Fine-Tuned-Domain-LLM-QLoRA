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
  const [result, setResult] = useState<{ base: Scores; finetuned: Scores } | null>(null);

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
          provider, model, apiKey,
          contract_text: contract, clause_type: clause,
          base_output: baseOutput, finetuned_output: finetunedOutput,
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
    <div className="verdict">
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          all: "unset", cursor: "pointer", display: "flex", width: "100%",
          alignItems: "center", justifyContent: "space-between",
          fontFamily: "var(--font-display)", fontWeight: 600, fontSize: ".9rem", color: "var(--text)",
        }}
      >
        <span>⚖️ Grade both with an LLM judge (your key)</span>
        <span style={{ color: "var(--muted)" }}>{open ? "−" : "+"}</span>
      </button>

      {open && (
        <div style={{ marginTop: "1rem", display: "grid", gap: ".8rem" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: ".7rem" }}>
            <div className="field" style={{ marginBottom: 0 }}>
              <label>Provider</label>
              <select className="input" value={provider} onChange={(e) => onProviderChange(e.target.value as ProviderId)}>
                {PROVIDERS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
            </div>
            <div className="field" style={{ marginBottom: 0 }}>
              <label>Model</label>
              <select className="input" value={model} onChange={(e) => setModel(e.target.value)}>
                {PROVIDER_MAP[provider].models.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
          </div>
          <div className="field" style={{ marginBottom: 0 }}>
            <label>API key · {PROVIDER_MAP[provider].keyHint}</label>
            <input
              className="input" type="password" value={apiKey}
              placeholder={PROVIDER_MAP[provider].keyPlaceholder}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <span className="note">Used once to score this comparison. Never stored or logged.</span>
          </div>
          <button className="btn btn-primary" onClick={judge} disabled={loading || !apiKey}>
            {loading ? "Scoring…" : "Judge both outputs"}
          </button>
          {error && <div className="callout">{error}</div>}
          {result && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: ".7rem" }}>
              <ScoreCard title="Base" scores={result.base} />
              <ScoreCard title="Fine-tuned" scores={result.finetuned} ft />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ScoreCard({ title, scores, ft }: { title: string; scores: Scores; ft?: boolean }) {
  const rows: [string, number][] = [
    ["Faithfulness", scores.faithfulness],
    ["Completeness", scores.completeness],
    ["Precision", scores.precision],
  ];
  const avg = (scores.faithfulness + scores.completeness + scores.precision) / 3;
  return (
    <div className={`verdict ${ft ? "is-ft" : ""}`} style={{ padding: ".9rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: ".6rem" }}>
        <span style={{ fontWeight: 600, fontSize: ".82rem" }}>{title}</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: ".9rem" }}>{avg.toFixed(1)}/10</span>
      </div>
      {rows.map(([k, v]) => (
        <div className="bar-row" key={k} style={{ marginBottom: ".4rem" }}>
          <span className="bar-label" style={{ width: "5.5rem", fontSize: ".72rem" }}>{k}</span>
          <span className="bar-track" style={{ height: ".45rem" }}>
            <span className="bar-fill" style={{ width: `${(v / 10) * 100}%`, background: ft ? "linear-gradient(90deg,#8b5cf6,#f5b642)" : "#6c6f95" }} />
          </span>
          <span className="bar-val" style={{ width: "1.4rem", fontSize: ".72rem" }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

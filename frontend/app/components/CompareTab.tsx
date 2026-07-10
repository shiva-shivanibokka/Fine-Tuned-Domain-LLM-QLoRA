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
    <>
      <div className="tab-intro">
        <span className="ti-icon">⚖️</span>
        <span>
          <b>What this tab does.</b> This is the live demo. Paste a contract,
          pick a clause type, and both models — the original Llama 3.2 3B and the
          fine-tuned one — read it and try to extract that clause. Their answers
          appear side by side as &ldquo;verdicts,&rdquo; and you can optionally
          have another LLM grade them. It needs the GPU inference backend to be
          connected; the other three tabs work without it.
        </span>
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">Extract a clause, live</h2>
          <span className="chip">base vs. fine-tuned</span>
        </div>
        <p className="panel-lead">
          Paste a contract excerpt, choose a clause type, and watch the base
          Llama 3.2 3B and the fine-tuned model answer side by side.
        </p>

        <div style={{ display: "grid", gap: "1.5rem", gridTemplateColumns: "minmax(0,1fr) minmax(0,1.15fr)" }} className="cmp-grid">
          <div>
            <div className="field">
              <label>Contract text</label>
              <textarea
                className="input"
                rows={9}
                value={contract}
                onChange={(e) => setContract(e.target.value)}
              />
            </div>
            <div className="field">
              <label>Clause type to extract</label>
              <select
                className="input"
                value={clause}
                onChange={(e) => setClause(e.target.value)}
              >
                {CLAUSE_TYPES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <button
              className="btn btn-primary"
              style={{ width: "100%" }}
              onClick={run}
              disabled={loading || contract.trim().length < 20}
            >
              {loading ? "Reading the contract…" : "Compare the two analysts"}
            </button>
            <p className="note" style={{ marginTop: ".7rem" }}>
              Inference runs on a serverless GPU that scales to zero; the first
              request after idle can take ~30s while the model wakes up.
            </p>
          </div>

          <div style={{ display: "grid", gap: "1rem", alignContent: "start" }}>
            {result?.error && <div className="callout">{result.error}</div>}
            <Verdict
              who="Base Llama 3.2 3B"
              sub="no fine-tuning"
              text={result?.base_output}
              latency={result?.latency_base_ms}
              loading={loading}
            />
            <Verdict
              who="Fine-tuned"
              sub="QLoRA + DPO on CUAD"
              text={result?.finetuned_output}
              latency={result?.latency_finetuned_ms}
              loading={loading}
              ft
            />
            {result?.base_output != null && (
              <JudgePanel
                contract={contract}
                clause={clause}
                baseOutput={result.base_output ?? ""}
                finetunedOutput={result.finetuned_output ?? ""}
              />
            )}
          </div>
        </div>
      </section>
    </>
  );
}

function Verdict({
  who,
  sub,
  text,
  latency,
  loading,
  ft,
}: {
  who: string;
  sub: string;
  text?: string;
  latency?: number;
  loading: boolean;
  ft?: boolean;
}) {
  return (
    <div className={`verdict ${ft ? "is-ft" : ""}`}>
      <div className="verdict-head">
        <div>
          <span className="who">{who}</span>
          {ft && <span className="tag" style={{ marginLeft: ".5rem" }}>TUNED</span>}
          <div className="sub">{sub}</div>
        </div>
        {latency != null && <span className="latency">{latency} ms</span>}
      </div>
      <div className="verdict-body">
        {loading ? (
          <span className="empty">Generating…</span>
        ) : text ? (
          text
        ) : (
          <span className="empty">The extracted clause will appear here.</span>
        )}
      </div>
    </div>
  );
}

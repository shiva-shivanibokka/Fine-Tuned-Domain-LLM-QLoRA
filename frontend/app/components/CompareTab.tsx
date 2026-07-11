"use client";

import { useState } from "react";
import { CLAUSE_TYPES } from "@/lib/data";
import JudgePanel from "@/app/components/JudgePanel";

// Realistic, CUAD-style excerpts (numbered sections, defined terms, dense
// legalese) — the distribution the model was actually fine-tuned on. Short,
// clean sentences are out-of-distribution and make the tuned model ramble.
// Each example pairs an excerpt with the clause type it contains.
const EXAMPLES: { label: string; clause: string; text: string }[] = [
  {
    label: "Governing Law",
    clause: "Governing Law",
    text: `ARTICLE 14. MISCELLANEOUS.

14.1 Governing Law. This Agreement, and all matters arising out of or relating to this Agreement, whether sounding in contract, tort, or statute, shall be governed by and construed in accordance with the internal laws of the State of New York, without giving effect to any choice or conflict of law provision that would cause the application of the laws of any jurisdiction other than the State of New York.

14.2 Notices. All notices under this Agreement must be in writing and delivered to the addresses set forth on the signature page hereto.`,
  },
  {
    label: "Termination for Convenience",
    clause: "Termination For Convenience",
    text: `SECTION 9. TERM AND TERMINATION.

9.1 Term. This Agreement commences on the Effective Date and continues for three (3) years unless earlier terminated in accordance with this Section 9.

9.2 Termination for Convenience. Either party may terminate this Agreement, in whole or in part, for any reason or no reason upon ninety (90) days' prior written notice to the other party. Upon such termination, Customer shall pay for all Services rendered through the effective date of termination.`,
  },
  {
    label: "Indemnification",
    clause: "Indemnification",
    text: `11. INDEMNIFICATION. The Supplier shall defend, indemnify, and hold harmless the Company and its officers, directors, employees, and agents from and against any and all claims, damages, liabilities, costs, and expenses (including reasonable attorneys' fees) arising out of or resulting from (a) the Supplier's breach of this Agreement, or (b) any negligent act or omission of the Supplier in the performance of the Services hereunder.`,
  },
  {
    label: "Anti-Assignment",
    clause: "Anti-Assignment",
    text: `12.3 Assignment. Neither party may assign, transfer, or delegate any of its rights or obligations under this Agreement, whether by operation of law or otherwise, without the prior written consent of the other party, except that either party may assign this Agreement in its entirety to a successor in connection with a merger, acquisition, or sale of all or substantially all of its assets. Any purported assignment in violation of this Section shall be null and void.`,
  },
  {
    label: "Limitation of Liability",
    clause: "Limitation Of Liability",
    text: `10. LIMITATION OF LIABILITY. IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING LOSS OF PROFITS OR REVENUE, ARISING OUT OF OR RELATED TO THIS AGREEMENT. EACH PARTY'S TOTAL AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE TOTAL FEES PAID BY CUSTOMER IN THE TWELVE (12) MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM.`,
  },
];

const SAMPLE_CONTRACT = EXAMPLES[0].text;

interface CompareResult {
  base_output?: string;
  finetuned_output?: string;
  latency_base_ms?: number;
  latency_finetuned_ms?: number;
  error?: string;
}

export default function CompareTab() {
  const [contract, setContract] = useState(SAMPLE_CONTRACT);
  const [clause, setClause] = useState(EXAMPLES[0].clause);
  const [activeEx, setActiveEx] = useState<string | null>(EXAMPLES[0].label);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResult | null>(null);

  function pickExample(ex: (typeof EXAMPLES)[number]) {
    setContract(ex.text);
    setClause(ex.clause);
    setActiveEx(ex.label);
    setResult(null);
  }

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

      <div className="callout">
        <strong>What to watch for.</strong> The fine-tuned model often
        over-generates — it drifts past the clause or invents contract language.
        That is the whole point of this project: it and the base model score
        almost identically on overlap (BERTScore 0.47 vs 0.49), yet the tuned one
        hallucinates far more. A single similarity number hides that gap — which
        is why the <em>Failure Modes</em> and <em>Ablation &amp; Trust</em> tabs
        measure grounding and calibration instead of just overlap.
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">Extract a clause, live</h2>
          <span className="chip">base vs. fine-tuned</span>
        </div>
        <p className="panel-lead">
          Pick an example (or paste your own), choose the clause type, and watch
          the base Llama 3.2 3B and the fine-tuned model answer side by side.
        </p>

        {/* Contract text */}
        <div className="field">
          <label>Contract text</label>
          <textarea
            className="input"
            rows={8}
            value={contract}
            onChange={(e) => {
              setContract(e.target.value);
              setActiveEx(null);
            }}
          />
        </div>

        {/* Example buttons */}
        <div className="field">
          <label>Or start from an example</label>
          <div className="example-chips">
            {EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                className={`chip-btn ${activeEx === ex.label ? "active" : ""}`}
                onClick={() => pickExample(ex)}
              >
                {ex.label}
              </button>
            ))}
          </div>
        </div>

        {/* Clause + run */}
        <div className="cmp-controls">
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
            className="btn btn-primary cmp-run"
            onClick={run}
            disabled={loading || contract.trim().length < 20}
          >
            {loading ? "Reading the contract…" : "Compare the two analysts"}
          </button>
        </div>
        <p className="note" style={{ marginTop: ".7rem" }}>
          Inference runs on a serverless GPU that scales to zero; the first
          request after idle can take ~30s while the model wakes up.
        </p>

        {result?.error && (
          <div className="callout" style={{ marginTop: "1.2rem" }}>
            {result.error}
          </div>
        )}

        {/* Two model outputs, side by side */}
        <div className="verdict-row">
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
        </div>

        {/* LLM judge, full width */}
        {result?.base_output != null && (
          <div style={{ marginTop: "1rem" }}>
            <JudgePanel
              contract={contract}
              clause={clause}
              baseOutput={result.base_output ?? ""}
              finetunedOutput={result.finetuned_output ?? ""}
            />
          </div>
        )}
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

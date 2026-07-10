"use client";

import { useState } from "react";
import type { Failures } from "@/lib/data";
import PlaceholderBadge from "@/app/components/PlaceholderBadge";

const COLORS = ["#8b5cf6", "#38bdf8", "#f5b642", "#fb7185", "#34d399", "#6366f1"];

export default function FailuresTab({ failures }: { failures: Failures }) {
  const [active, setActive] = useState<number | null>(null);
  const pts = failures.cluster_data;

  const xs = pts.map((p) => p.umap_x);
  const ys = pts.map((p) => p.umap_y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const W = 560, H = 380, M = 26;
  const sx = (x: number) => M + ((x - minX) / (maxX - minX || 1)) * (W - 2 * M);
  const sy = (y: number) => H - M - ((y - minY) / (maxY - minY || 1)) * (H - 2 * M);

  const clusterIds = Object.keys(failures.cluster_labels)
    .map(Number)
    .sort((a, b) => a - b);
  const examples = active != null ? pts.filter((p) => p.cluster_id === active) : [];

  return (
    <>
      {failures._placeholder && <PlaceholderBadge />}

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">Where the model fails, and how</h2>
          <span className="chip">UMAP + HDBSCAN</span>
        </div>
        <p className="panel-lead">
          Every point is a test case the fine-tuned model got wrong (BERTScore F1
          &lt; 0.65), embedded and projected to 2-D, then clustered. Clusters
          reveal <em>systematic</em> error patterns — click one to inspect it.
        </p>

        <div style={{ display: "grid", gap: "1.5rem", gridTemplateColumns: "minmax(0,1.35fr) minmax(0,1fr)" }} className="fail-grid">
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: "rgba(6,7,18,0.4)", borderRadius: 14, border: "1px solid var(--border)" }}>
            {pts.map((p, i) => {
              const on = active == null || active === p.cluster_id;
              return (
                <circle
                  key={i}
                  cx={sx(p.umap_x)}
                  cy={sy(p.umap_y)}
                  r={on ? 6 : 3}
                  fill={COLORS[p.cluster_id % COLORS.length]}
                  opacity={on ? 0.92 : 0.2}
                  style={{ cursor: "pointer", transition: "all .2s ease" }}
                  onClick={() => setActive(p.cluster_id)}
                />
              );
            })}
          </svg>

          <div>
            <p className="eyebrow" style={{ marginBottom: ".7rem" }}>
              Failure clusters
            </p>
            {clusterIds.map((cid) => (
              <button
                key={cid}
                className={`cluster-btn ${active === cid ? "active" : ""}`}
                onClick={() => setActive(active === cid ? null : cid)}
              >
                <span className="cluster-dot" style={{ background: COLORS[cid % COLORS.length] }} />
                <span>{failures.cluster_labels[String(cid)]}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {active != null && examples.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2 className="panel-title">Inside this cluster</h2>
            <span className="chip">{examples.length} cases</span>
          </div>
          {examples.slice(0, 4).map((e, i) => (
            <div className="example" key={i}>
              <div className="ex-clause">{e.clause_type}</div>
              <div className="ex-ref">Reference — {e.reference}</div>
              <div className="ex-pred">Predicted — {e.prediction}</div>
            </div>
          ))}
        </section>
      )}
    </>
  );
}

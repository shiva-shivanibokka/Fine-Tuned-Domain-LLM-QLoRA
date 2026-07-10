"use client";

import { useRef, useState } from "react";
import type { Failures, FailurePoint } from "@/lib/data";
import PlaceholderBadge from "@/app/components/PlaceholderBadge";

// Bright, distinct cluster colors; noise (cluster -1) gets a visible slate.
const COLORS = ["#a78bfa", "#38bdf8", "#fbbf24", "#f472b6", "#34d399", "#818cf8"];
const NOISE = "#94a3b8";
const colorFor = (id: number) => (id < 0 ? NOISE : COLORS[id % COLORS.length]);

function Info({ tip }: { tip: string }) {
  return (
    <span className="info" tabIndex={0} data-tip={tip}>
      i
    </span>
  );
}

export default function FailuresTab({ failures }: { failures: Failures }) {
  const [active, setActive] = useState<number | null>(null);
  const [hover, setHover] = useState<{ p: FailurePoint; x: number; y: number } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const pts = failures.cluster_data;
  const xs = pts.map((p) => p.umap_x);
  const ys = pts.map((p) => p.umap_y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const W = 760, H = 460, M = 34;
  const sx = (x: number) => M + ((x - minX) / (maxX - minX || 1)) * (W - 2 * M);
  const sy = (y: number) => H - M - ((y - minY) / (maxY - minY || 1)) * (H - 2 * M);

  const clusterIds = Object.keys(failures.cluster_labels)
    .map(Number)
    .sort((a, b) => a - b);
  const examples = active != null ? pts.filter((p) => p.cluster_id === active) : [];

  function track(e: React.MouseEvent, p: FailurePoint) {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    setHover({ p, x: e.clientX - rect.left, y: e.clientY - rect.top });
  }

  return (
    <>
      <div className="tab-intro">
        <span className="ti-icon">🔬</span>
        <span>
          <b>What this tab shows.</b> Every dot is one test contract the
          fine-tuned model got <b>wrong</b>. Each wrong answer is turned into a
          vector (an embedding), squeezed down to 2-D with UMAP, and grouped by
          HDBSCAN so similar mistakes sit together. Dots of the same color are
          the same <b>type</b> of error. Hover a dot to see the case; click a
          cluster to read examples. Grey dots are one-off outliers (&ldquo;noise&rdquo;).
        </span>
      </div>

      {failures._placeholder && <PlaceholderBadge />}

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">
            Failure map
            <Info tip="A 2-D projection (UMAP) of the model's wrong answers. Position has no units — what matters is which dots cluster together, revealing systematic error patterns." />
          </h2>
          <span className="chip">UMAP + HDBSCAN · {pts.length} failures</span>
        </div>
        <p className="panel-lead">
          Hover any dot for its clause type and what the model predicted. Click a
          cluster in the legend to filter and read real examples below.
        </p>

        <div className="umap-wrap" ref={wrapRef}>
          <svg className="umap-svg" viewBox={`0 0 ${W} ${H}`} aria-label="Failure clusters">
            {pts.map((p, i) => {
              const on = active == null || active === p.cluster_id;
              return (
                <circle
                  key={i}
                  cx={sx(p.umap_x)}
                  cy={sy(p.umap_y)}
                  r={on ? 8 : 4}
                  fill={colorFor(p.cluster_id)}
                  stroke="rgba(255,255,255,0.25)"
                  strokeWidth={0.75}
                  opacity={on ? 0.95 : 0.18}
                  onMouseEnter={(e) => track(e, p)}
                  onMouseMove={(e) => track(e, p)}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => setActive(p.cluster_id)}
                >
                  <title>{`${p.clause_type} — predicted: ${p.prediction.slice(0, 80)}`}</title>
                </circle>
              );
            })}
          </svg>
          {hover && (
            <div className="umap-tip" style={{ left: hover.x, top: hover.y }}>
              <div className="t-clause">{hover.p.clause_type}</div>
              <div className="t-pred">
                Predicted: {hover.p.prediction.slice(0, 110)}
                {hover.p.prediction.length > 110 ? "…" : ""}
              </div>
            </div>
          )}
        </div>

        <div className="umap-legend">
          {clusterIds.map((cid) => (
            <button
              key={cid}
              className={`lg ${active === cid ? "" : ""}`}
              onClick={() => setActive(active === cid ? null : cid)}
              style={{
                all: "unset",
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: ".45rem",
                opacity: active == null || active === cid ? 1 : 0.4,
              }}
            >
              <span className="sw" style={{ background: colorFor(cid) }} />
              {failures.cluster_labels[String(cid)]}
            </button>
          ))}
        </div>

        {active != null && (
          <div className="sel-bar">
            <span className="sel-dot" style={{ background: colorFor(active) }} />
            <span className="sel-txt">
              Showing <b>{failures.cluster_labels[String(active)]}</b> —{" "}
              {examples.length} of {pts.length} failures highlighted
            </span>
            <button className="sel-clear" onClick={() => setActive(null)}>
              ✕ Show full map
            </button>
          </div>
        )}
      </section>

      {active != null && examples.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2 className="panel-title">
              Inside this cluster
              <Info tip="Every test case in the selected cluster: the reference (correct) clause vs. what the model actually produced. Scroll to read them all." />
            </h2>
            <span className="chip">{examples.length} cases · scroll to see all</span>
          </div>
          <div className="example-scroll">
            {examples.map((e, i) => (
              <div className="example" key={i}>
                <div className="ex-clause">{e.clause_type}</div>
                <div className="ex-ref">Reference — {e.reference}</div>
                <div className="ex-pred">Predicted — {e.prediction}</div>
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  );
}

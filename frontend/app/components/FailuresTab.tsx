"use client";

import { useState } from "react";
import type { Failures } from "@/lib/data";
import PlaceholderBadge from "@/app/components/PlaceholderBadge";

const COLORS = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ec4899", "#22c55e", "#ef4444"];

export default function FailuresTab({ failures }: { failures: Failures }) {
  const [active, setActive] = useState<number | null>(null);
  const pts = failures.cluster_data;

  const xs = pts.map((p) => p.umap_x);
  const ys = pts.map((p) => p.umap_y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 0.12;
  const W = 560, H = 380, M = 24;
  const sx = (x: number) =>
    M + ((x - minX) / (maxX - minX || 1)) * (W - 2 * M) * (1 - pad) + pad * 20;
  const sy = (y: number) =>
    H - M - ((y - minY) / (maxY - minY || 1)) * (H - 2 * M) * (1 - pad);

  const clusterIds = Object.keys(failures.cluster_labels)
    .map(Number)
    .sort((a, b) => a - b);

  const examples = active != null ? pts.filter((p) => p.cluster_id === active) : [];

  return (
    <div className="space-y-6">
      {failures._placeholder && <PlaceholderBadge />}

      <p className="text-sm text-zinc-400">
        Every point is a test case where the fine-tuned model failed (BERTScore
        F1 &lt; 0.65), embedded and projected with UMAP, then clustered with
        HDBSCAN. Clusters reveal <em className="text-zinc-200">systematic</em>{" "}
        error patterns — not just one-off mistakes.
      </p>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
            {pts.map((p, i) => (
              <circle
                key={i}
                cx={sx(p.umap_x)}
                cy={sy(p.umap_y)}
                r={active === p.cluster_id || active == null ? 6 : 3}
                fill={COLORS[p.cluster_id % COLORS.length]}
                opacity={active == null || active === p.cluster_id ? 0.9 : 0.25}
                className="cursor-pointer transition-all"
                onClick={() => setActive(p.cluster_id)}
              />
            ))}
          </svg>
        </section>

        <section className="space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            Failure clusters
          </h3>
          {clusterIds.map((cid) => (
            <button
              key={cid}
              onClick={() => setActive(active === cid ? null : cid)}
              className={`flex w-full items-start gap-2.5 rounded-lg border p-3 text-left text-sm transition ${
                active === cid
                  ? "border-zinc-600 bg-zinc-800/60"
                  : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
              }`}
            >
              <span
                className="mt-1 inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: COLORS[cid % COLORS.length] }}
              />
              <span className="text-zinc-300">
                {failures.cluster_labels[String(cid)]}
              </span>
            </button>
          ))}
        </section>
      </div>

      {active != null && examples.length > 0 && (
        <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
          <h3 className="mb-3 text-sm font-semibold text-zinc-200">
            Examples from this cluster
          </h3>
          <div className="space-y-3">
            {examples.slice(0, 4).map((e, i) => (
              <div key={i} className="rounded-lg bg-zinc-950 p-3 text-xs">
                <p className="mb-1 text-zinc-500">
                  Clause: <span className="text-zinc-300">{e.clause_type}</span>
                </p>
                <p className="text-emerald-400/80">
                  Reference: <span className="text-zinc-300">{e.reference}</span>
                </p>
                <p className="mt-1 text-rose-400/80">
                  Prediction:{" "}
                  <span className="text-zinc-300">{e.prediction}</span>
                </p>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

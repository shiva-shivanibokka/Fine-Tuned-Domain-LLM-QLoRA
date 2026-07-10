import type { DatasetCard } from "@/lib/data";

export default function DatasetTab({ card }: { card: DatasetCard }) {
  const dist = Object.entries(card.clause_distribution).sort((a, b) => b[1] - a[1]);
  const maxCount = Math.max(...dist.map(([, c]) => c));

  const stats = [
    { label: "Total samples", value: card.splits.total.toLocaleString() },
    { label: "Train", value: card.splits.train.toLocaleString() },
    { label: "Validation", value: card.splits.val.toLocaleString() },
    { label: "Test", value: card.splits.test.toLocaleString() },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {stats.map((s) => (
          <div
            key={s.label}
            className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4"
          >
            <div className="text-2xl font-semibold text-white">{s.value}</div>
            <div className="mt-1 text-xs text-zinc-500">{s.label}</div>
          </div>
        ))}
      </div>

      <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
        <h2 className="mb-1 text-sm font-semibold text-zinc-200">
          Clause type distribution
        </h2>
        <p className="mb-5 text-xs text-zinc-500">
          Source: {card.dataset} · stratified so every clause type appears in
          train / val / test.
        </p>
        <div className="space-y-2.5">
          {dist.map(([name, count]) => (
            <div key={name} className="flex items-center gap-3">
              <span className="w-48 shrink-0 truncate text-xs text-zinc-400">
                {name}
              </span>
              <div className="h-5 flex-1 overflow-hidden rounded bg-zinc-800">
                <div
                  className="h-full bg-violet-500/80"
                  style={{ width: `${(count / maxCount) * 100}%` }}
                />
              </div>
              <span className="w-10 shrink-0 text-right font-mono text-xs text-zinc-300">
                {count}
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
          <h3 className="mb-2 text-sm font-semibold text-zinc-200">
            Answer length (words)
          </h3>
          <dl className="space-y-1 text-sm text-zinc-400">
            <Row k="Mean" v={card.answer_length_stats.mean} />
            <Row k="Median" v={card.answer_length_stats.median} />
            <Row k="95th pct" v={card.answer_length_stats.p95} />
          </dl>
        </div>
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
          <h3 className="mb-2 text-sm font-semibold text-zinc-200">
            Context length (words)
          </h3>
          <dl className="space-y-1 text-sm text-zinc-400">
            <Row k="Mean" v={card.context_length_stats.mean} />
            <Row k="Median" v={card.context_length_stats.median} />
          </dl>
        </div>
      </section>
    </div>
  );
}

function Row({ k, v }: { k: string; v: number }) {
  return (
    <div className="flex justify-between">
      <dt>{k}</dt>
      <dd className="font-mono text-zinc-200">{v}</dd>
    </div>
  );
}

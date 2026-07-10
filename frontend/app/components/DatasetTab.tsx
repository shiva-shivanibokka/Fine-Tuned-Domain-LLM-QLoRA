import type { DatasetCard } from "@/lib/data";

export default function DatasetTab({ card }: { card: DatasetCard }) {
  const dist = Object.entries(card.clause_distribution).sort((a, b) => b[1] - a[1]);
  const maxCount = Math.max(...dist.map(([, c]) => c));

  const tiles = [
    { big: card.splits.total.toLocaleString(), lbl: "clause examples" },
    { big: card.splits.train.toLocaleString(), lbl: "train" },
    { big: card.splits.val.toLocaleString(), lbl: "validation" },
    { big: card.splits.test.toLocaleString(), lbl: "test" },
  ];

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">CUAD, cleaned</h2>
          <span className="chip">{card.dataset}</span>
        </div>
        <p className="panel-lead">
          Real contracts annotated by attorneys, filtered through per-clause
          deduplication and perplexity filtering, then split stratified so every
          clause type appears in each split.
        </p>
        <div className="tile-grid">
          {tiles.map((t) => (
            <div className="tile" key={t.lbl}>
              <div className="big">{t.big}</div>
              <div className="lbl">{t.lbl}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">Clause type distribution</h2>
          <span className="chip">imbalanced by design</span>
        </div>
        <p className="panel-lead">
          Governing Law and Anti-Assignment dominate; Most Favored Nation is
          rare. Stratified splitting keeps that shape consistent across train,
          val, and test.
        </p>
        {dist.map(([name, count]) => (
          <div className="bar-row" key={name}>
            <span className="bar-label" title={name}>
              {name}
            </span>
            <span className="bar-track">
              <span
                className="bar-fill"
                style={{
                  width: `${(count / maxCount) * 100}%`,
                  background: "linear-gradient(90deg,#6366f1,#8b5cf6)",
                }}
              />
            </span>
            <span className="bar-val">{count}</span>
          </div>
        ))}
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">Length profile</h2>
        </div>
        <div className="tile-grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
          <div className="tile">
            <div className="big">{card.answer_length_stats.mean}</div>
            <div className="lbl">mean answer length (words)</div>
          </div>
          <div className="tile">
            <div className="big">{card.answer_length_stats.p95}</div>
            <div className="lbl">95th-pct answer length</div>
          </div>
          <div className="tile">
            <div className="big">{card.context_length_stats.mean}</div>
            <div className="lbl">mean context length (words)</div>
          </div>
        </div>
      </section>
    </>
  );
}

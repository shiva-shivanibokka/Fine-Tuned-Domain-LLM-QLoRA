import type { DatasetCard } from "@/lib/data";

function Info({ tip }: { tip: string }) {
  return (
    <span className="info" tabIndex={0} data-tip={tip}>
      i
    </span>
  );
}

const CLAUSE_FILL = "linear-gradient(90deg,#8b5cf6,#38bdf8)";

export default function DatasetTab({ card }: { card: DatasetCard }) {
  const dist = Object.entries(card.clause_distribution).sort((a, b) => b[1] - a[1]);
  const maxCount = Math.max(...dist.map(([, c]) => c));

  const tiles = [
    { big: card.splits.total.toLocaleString(), lbl: "clause examples", tip: "Total labeled clause examples after cleaning — every row the model could learn from or be tested on." },
    { big: card.splits.train.toLocaleString(), lbl: "train", tip: "Examples the model actually learns from during fine-tuning." },
    { big: card.splits.val.toLocaleString(), lbl: "validation", tip: "Held out during training to tune settings and watch for overfitting — the model never trains on these." },
    { big: card.splits.test.toLocaleString(), lbl: "test", tip: "Fully unseen until the final scoring. Every number in the Ablation tab comes from this split." },
  ];

  return (
    <>
      <div className="tab-intro">
        <span className="ti-icon">🗂️</span>
        <span>
          <b>What this tab shows.</b> The training data behind the model — CUAD,
          a set of real contracts annotated by lawyers. It shows how many
          examples there are, how they split into train/validation/test, which
          clause types appear (and how often), and how long the text runs. This
          is the raw material; the model only knows what&apos;s here.
        </span>
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">
            CUAD, cleaned
            <Info tip="Contracts annotated by attorneys, then run through per-clause deduplication and a perplexity filter to drop garbled or duplicate text before training." />
          </h2>
          <span className="chip">{card.dataset}</span>
        </div>
        <p className="panel-lead">
          Each tile is a count. The dataset was split stratified, so every clause
          type appears in all three splits in the same proportion.
        </p>
        <div className="tile-grid">
          {tiles.map((t) => (
            <div className="tile" key={t.lbl}>
              <div className="big">{t.big}</div>
              <div className="tile-head">
                <span className="lbl">{t.lbl}</span>
                <Info tip={t.tip} />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">
            Clause type distribution
            <Info tip="How many training examples exist for each clause type. Longer bar = more examples. The model sees Governing Law far more often than Most Favored Nation." />
          </h2>
          <span className="chip">imbalanced by design</span>
        </div>
        <p className="panel-lead">
          Each bar&apos;s length is the number of examples for that clause type.
          Governing Law and Anti-Assignment dominate; Most Favored Nation is
          rare — a real imbalance the model has to cope with.
        </p>
        {dist.map(([name, count]) => (
          <div className="bar-row" key={name}>
            <span className="bar-label" title={name}>
              {name}
            </span>
            <span className="bar-track">
              <span
                className="bar-fill"
                style={{ width: `${(count / maxCount) * 100}%`, background: CLAUSE_FILL }}
              />
            </span>
            <span className="bar-val">{count}</span>
          </div>
        ))}
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2 className="panel-title">
            Length profile
            <Info tip="How long the answers and contract excerpts are, in words. Answers are short spans; contexts are long, which is why the model needs a long input window." />
          </h2>
        </div>
        <div
          className="tile-grid"
          style={{ gridTemplateColumns: "repeat(3,1fr)" }}
        >
          <div className="tile">
            <div className="big">{card.answer_length_stats.mean}</div>
            <div className="tile-head">
              <span className="lbl">mean answer length (words)</span>
              <Info tip="Average length of the correct clause the model must output. Short spans — extraction, not essay writing." />
            </div>
          </div>
          <div className="tile">
            <div className="big">{card.answer_length_stats.p95}</div>
            <div className="tile-head">
              <span className="lbl">95th-pct answer length</span>
              <Info tip="95% of answers are shorter than this. Sets the generation budget (MAX_NEW_TOKENS) so the model rarely gets cut off." />
            </div>
          </div>
          <div className="tile">
            <div className="big">{card.context_length_stats.mean}</div>
            <div className="tile-head">
              <span className="lbl">mean context length (words)</span>
              <Info tip="Average length of the contract excerpt fed in as context. Long inputs are why the model needs a wide context window." />
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

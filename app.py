"""
Fine-Tuned-Domain-LLM-QLoRA — Gradio Demo UI

Four tabs:
  1. Model Comparison   — paste contract, compare base vs fine-tuned side by side
  2. Training Dashboard — loss curves, metrics table, memory vs quality scatter
  3. Failure Explorer   — UMAP cluster plot, click cluster to see examples
  4. Dataset Explorer   — clause distribution, sample training examples

Run: python app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd

from config import (
    ANTHROPIC_API_KEY,
    CHECKPOINTS,
    DATA_PROCESSED,
    RESULTS,
    TARGET_CLAUSES,
)

# ---------------------------------------------------------------------------
# Helper: load results if they exist
# ---------------------------------------------------------------------------


def _load_results() -> dict:
    """Load all available eval results."""
    tags = ["base", "lora", "qlora", "dpo"]
    data = {}
    for tag in tags:
        path = RESULTS / f"{tag}_eval.json"
        if path.exists():
            data[tag] = json.loads(path.read_text())
    return data


def _load_failure_data(tag: str = "dpo") -> dict:
    path = RESULTS / f"{tag}_failures.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_dataset_card() -> dict:
    path = DATA_PROCESSED / "dataset_card.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# ---------------------------------------------------------------------------
# Tab 1: Model Comparison
# ---------------------------------------------------------------------------


def compare_models(
    contract_text: str,
    clause_type: str,
    api_key: str,
) -> tuple[str, str, str, str]:
    """Call the FastAPI /compare endpoint or run inference directly."""
    if not contract_text.strip():
        return "Please enter contract text.", "", "", ""

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    try:
        import requests

        resp = requests.post(
            "http://localhost:8000/compare",
            json={"contract_text": contract_text, "clause_type": clause_type},
            timeout=120,
        )
        if resp.status_code == 200:
            d = resp.json()
            return (
                d["base_output"],
                d["finetuned_output"],
                f"Base: {d['latency_base_ms']}ms",
                f"Fine-tuned: {d['latency_finetuned_ms']}ms",
            )
        return f"API error: {resp.status_code}", "", "", ""
    except Exception as e:
        return (
            f"API not running. Start with: uvicorn serving.api:app\nError: {e}",
            "",
            "",
            "",
        )


# ---------------------------------------------------------------------------
# Tab 2: Training Dashboard
# ---------------------------------------------------------------------------


def build_metrics_table() -> pd.DataFrame:
    """Build a comparison table of all models."""
    results = _load_results()
    if not results:
        return pd.DataFrame(
            {"Status": ["No results yet. Run: python -m evaluation.evaluator --all"]}
        )

    rows = []
    for tag, r in results.items():
        rows.append(
            {
                "Model": tag.upper(),
                "BERTScore F1": f"{r.get('bertscore_f1', 0):.3f}",
                "G-Eval Faithfulness": f"{r.get('geval_faithfulness', 'N/A')}",
                "Hallucination Rate": f"{r.get('hallucination_rate', 0):.3f}",
                "Clause Accuracy": f"{r.get('clause_presence_accuracy', 0):.3f}",
                "ECE (↓ better)": f"{r.get('ece', 0):.4f}",
            }
        )
    return pd.DataFrame(rows)


def build_memory_quality_scatter() -> str:
    """Return plotly JSON for memory vs quality scatter."""
    try:
        import plotly.graph_objects as go

        # VRAM usage estimates for Llama 3.2 3B on RTX 4060
        memory_usage = {"BASE": 6.8, "LORA": 7.2, "QLORA": 3.9, "DPO": 4.1}
        results = _load_results()

        x, y, labels = [], [], []
        for tag, r in results.items():
            score = r.get("bertscore_f1", 0)
            mem = memory_usage.get(tag.upper(), 5.0)
            x.append(mem)
            y.append(score)
            labels.append(tag.upper())

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=14, color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
                ),
            )
        )
        fig.update_layout(
            title="VRAM Usage vs. BERTScore F1 — Quality/Efficiency Tradeoff",
            xaxis_title="VRAM (GB)",
            yaxis_title="BERTScore F1",
            template="plotly_white",
        )
        return fig.to_json()
    except ImportError:
        return "{}"


# ---------------------------------------------------------------------------
# Tab 3: Failure Mode Explorer
# ---------------------------------------------------------------------------


def build_umap_plot(model_tag: str = "dpo") -> str:
    """Return plotly JSON for the UMAP cluster plot."""
    try:
        import plotly.graph_objects as go

        data = _load_failure_data(model_tag)
        if not data or not data.get("cluster_data"):
            return "{}"

        cluster_data = data["cluster_data"]
        labels = data.get("cluster_labels", {})

        # Group by cluster
        clusters: dict[int, list] = {}
        for point in cluster_data:
            cid = point["cluster_id"]
            clusters.setdefault(cid, []).append(point)

        fig = go.Figure()
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

        for i, (cid, points) in enumerate(sorted(clusters.items())):
            name = labels.get(str(cid), f"Cluster {cid}")
            fig.add_trace(
                go.Scatter(
                    x=[p["umap_x"] for p in points],
                    y=[p["umap_y"] for p in points],
                    mode="markers",
                    name=name[:60],
                    marker=dict(size=8, color=colors[i % len(colors)]),
                    text=[
                        f"Clause: {p['clause_type']}<br>Pred: {p['prediction'][:100]}..."
                        for p in points
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Failure Mode Clusters — {model_tag.upper()} Model",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            template="plotly_white",
            legend=dict(x=1, y=1),
        )
        return fig.to_json()
    except ImportError:
        return "{}"


def get_cluster_examples(model_tag: str, cluster_id: int) -> str:
    """Return formatted examples from a specific cluster."""
    data = _load_failure_data(model_tag)
    if not data:
        return "No failure data loaded."

    examples = [
        p for p in data.get("cluster_data", []) if p["cluster_id"] == cluster_id
    ][:5]

    if not examples:
        return f"No examples found for cluster {cluster_id}."

    lines = [
        f"**Cluster {cluster_id} — {data['cluster_labels'].get(str(cluster_id), '')}**\n"
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"**Example {i}** ({ex['clause_type']})")
        lines.append(f"*Reference:* {ex['reference'][:200]}...")
        lines.append(f"*Prediction:* {ex['prediction'][:200]}...")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 4: Dataset Explorer
# ---------------------------------------------------------------------------


def build_clause_distribution_chart() -> str:
    """Return plotly JSON for clause distribution bar chart."""
    try:
        import plotly.graph_objects as go

        card = _load_dataset_card()
        if not card:
            return "{}"

        dist = card.get("clause_distribution", {})
        sorted_items = sorted(dist.items(), key=lambda x: -x[1])

        fig = go.Figure(
            go.Bar(
                x=[v for _, v in sorted_items],
                y=[k for k, _ in sorted_items],
                orientation="h",
                marker_color="#636EFA",
            )
        )
        fig.update_layout(
            title="Clause Type Distribution (Training Data)",
            xaxis_title="Count",
            template="plotly_white",
            height=500,
        )
        return fig.to_json()
    except ImportError:
        return "{}"


def get_sample_examples(clause_type: str) -> str:
    """Return a few sample training examples for the chosen clause type."""
    train_path = DATA_PROCESSED / "train.json"
    if not train_path.exists():
        return "Run the data pipeline first: python -m data.pipeline"

    train_data = json.loads(train_path.read_text())
    samples = [s for s in train_data if s.get("clause_type") == clause_type][:3]

    if not samples:
        return f"No samples found for clause type: {clause_type}"

    lines = [f"**{len(samples)} sample(s) for: {clause_type}**\n"]
    for i, s in enumerate(samples, 1):
        lines.append(f"**Sample {i}**")
        lines.append(f"*Contract excerpt:* {s['contract_text'][:300]}...")
        lines.append(f"*Ground truth:* {s['answer'][:200]}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# Fine-Tuned Legal LLM — Llama 3.2 3B on CUAD

Llama 3.2 3B fine-tuned on 510 real legal contracts (CUAD dataset) using
**LoRA**, **QLoRA 4-bit**, and **DPO** preference optimisation.

Key differentiators: BERTScore/G-Eval evaluation, ECE calibration, failure mode clustering.
"""

SAMPLE_CONTRACT = """This Agreement shall be governed by and construed in accordance with the laws
of the State of Delaware, without regard to its conflict of law provisions.
Either party may terminate this Agreement for convenience upon thirty (30) days
prior written notice to the other party. The Company shall indemnify and hold
harmless the Contractor from any claims arising from the Company's breach of
this Agreement. Neither party shall assign its rights or obligations under
this Agreement without the prior written consent of the other party."""

with gr.Blocks(title="Fine-Tuned Legal LLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # ── Tab 1: Model Comparison ──────────────────────────────────────
        with gr.TabItem("Model Comparison"):
            gr.Markdown("### Compare base Llama 3.2 3B vs. fine-tuned model")
            with gr.Row():
                with gr.Column(scale=2):
                    contract_input = gr.Textbox(
                        label="Contract Text",
                        value=SAMPLE_CONTRACT,
                        lines=8,
                        placeholder="Paste contract excerpt here...",
                    )
                    clause_selector = gr.Dropdown(
                        choices=TARGET_CLAUSES,
                        value=TARGET_CLAUSES[0],
                        label="Clause Type to Extract",
                    )
                    api_key_input = gr.Textbox(
                        label="Anthropic API Key (for G-Eval scoring, optional)",
                        type="password",
                        value=ANTHROPIC_API_KEY or "",
                    )
                    compare_btn = gr.Button("Compare Models", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        base_output = gr.Textbox(label="Base Model Output", lines=6)
                        ft_output = gr.Textbox(label="Fine-tuned Model Output", lines=6)
                    with gr.Row():
                        base_latency = gr.Textbox(
                            label="Base Latency", interactive=False
                        )
                        ft_latency = gr.Textbox(
                            label="Fine-tuned Latency", interactive=False
                        )

            compare_btn.click(
                fn=compare_models,
                inputs=[contract_input, clause_selector, api_key_input],
                outputs=[base_output, ft_output, base_latency, ft_latency],
            )

        # ── Tab 2: Training Dashboard ────────────────────────────────────
        with gr.TabItem("Training Dashboard"):
            gr.Markdown("### Ablation study: LoRA vs. QLoRA vs. DPO")
            metrics_table = gr.DataFrame(
                value=build_metrics_table,
                label="Evaluation Metrics Comparison",
                every=30,
            )
            gr.Markdown(
                "**Note:** Lower ECE = better calibrated. Lower hallucination rate = more grounded."
            )

            scatter_plot = gr.Plot(label="VRAM vs. Quality Tradeoff")

            def render_scatter():
                import json as _json

                try:
                    import plotly.io as pio

                    fig_json = build_memory_quality_scatter()
                    if fig_json != "{}":
                        return pio.from_json(fig_json)
                except Exception:
                    pass
                return None

            refresh_btn = gr.Button("Refresh Charts")
            refresh_btn.click(fn=render_scatter, outputs=scatter_plot)

        # ── Tab 3: Failure Mode Explorer ─────────────────────────────────
        with gr.TabItem("Failure Mode Explorer"):
            gr.Markdown("### Systematic error analysis via UMAP + HDBSCAN clustering")
            gr.Markdown(
                "Each point is a test case where the model failed. "
                "Clusters reveal systematic error patterns (e.g., 'Termination clauses — false negatives')."
            )
            with gr.Row():
                failure_model = gr.Dropdown(
                    choices=["base", "lora", "qlora", "dpo"],
                    value="dpo",
                    label="Model to analyse",
                )
                cluster_id_input = gr.Number(
                    value=0, label="Cluster ID to inspect", precision=0
                )

            umap_plot = gr.Plot(label="UMAP Failure Clusters")
            cluster_examples = gr.Markdown(label="Cluster Examples")

            def render_umap(tag: str):
                import json as _json

                try:
                    import plotly.io as pio

                    fig_json = build_umap_plot(tag)
                    if fig_json != "{}":
                        return pio.from_json(fig_json)
                except Exception:
                    pass
                return None

            failure_model.change(
                fn=render_umap, inputs=failure_model, outputs=umap_plot
            )
            cluster_id_input.change(
                fn=get_cluster_examples,
                inputs=[failure_model, cluster_id_input],
                outputs=cluster_examples,
            )

        # ── Tab 4: Dataset Explorer ──────────────────────────────────────
        with gr.TabItem("Dataset Explorer"):
            gr.Markdown("### CUAD dataset — processed training data")
            dataset_card = _load_dataset_card()
            if dataset_card:
                gr.Markdown(f"""
**Total samples:** {dataset_card.get("splits", {}).get("total", "N/A")}  
**Train / Val / Test:** {dataset_card.get("splits", {}).get("train", "N/A")} / {dataset_card.get("splits", {}).get("val", "N/A")} / {dataset_card.get("splits", {}).get("test", "N/A")}  
**Mean answer length:** {dataset_card.get("answer_length_stats", {}).get("mean", "N/A")} words  
**Mean context length:** {dataset_card.get("context_length_stats", {}).get("mean", "N/A")} words
""")
            else:
                gr.Markdown(
                    "*Run `python -m data.pipeline` to generate the dataset card.*"
                )

            dist_plot = gr.Plot(label="Clause Distribution")

            def render_dist():
                import json as _json

                try:
                    import plotly.io as pio

                    fig_json = build_clause_distribution_chart()
                    if fig_json != "{}":
                        return pio.from_json(fig_json)
                except Exception:
                    pass
                return None

            clause_example_selector = gr.Dropdown(
                choices=TARGET_CLAUSES,
                value=TARGET_CLAUSES[0],
                label="View examples for clause type",
            )
            example_output = gr.Markdown()

            clause_example_selector.change(
                fn=get_sample_examples,
                inputs=clause_example_selector,
                outputs=example_output,
            )

            demo.load(fn=render_dist, outputs=dist_plot)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

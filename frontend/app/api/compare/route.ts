// Server-side proxy to the Modal serverless-GPU backend (serving/modal_app.py).
//
// Modal serves base Llama 3.2 3B vs the fine-tuned QLoRA+DPO adapter on a T4 that
// scales to zero. Configure via env:
//   MODAL_COMPARE_URL   the deployed https://...compare.modal.run endpoint

export const maxDuration = 300; // allow for Modal cold start on Fluid Compute

export async function POST(request: Request) {
  const url = process.env.MODAL_COMPARE_URL;
  if (!url) {
    return Response.json(
      {
        error:
          "Live model comparison requires the GPU inference backend, which isn't connected in this demo. The Ablation Dashboard, Failure Explorer, and Dataset tabs use real evaluation results and are fully interactive.",
      },
      { status: 503 },
    );
  }

  let body: { contract_text?: string; clause_type?: string };
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  if (!body.contract_text || body.contract_text.trim().length < 20) {
    return Response.json(
      { error: "contract_text must be at least 20 characters." },
      { status: 400 },
    );
  }

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contract_text: body.contract_text,
        clause_type: body.clause_type ?? "Governing Law",
      }),
      signal: AbortSignal.timeout(290_000),
    });
    if (!res.ok) throw new Error(`backend ${res.status}`);
    // Modal returns exactly { base_output, finetuned_output, latency_base_ms, latency_finetuned_ms }.
    return Response.json(await res.json());
  } catch {
    return Response.json(
      {
        error:
          "Could not reach the inference backend. The GPU may be waking up (cold start) — try again in a moment.",
      },
      { status: 504 },
    );
  }
}

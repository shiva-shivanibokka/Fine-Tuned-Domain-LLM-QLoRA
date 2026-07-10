// Server-side proxy to the Hugging Face inference Space (Gradio SDK + ZeroGPU).
//
// Uses @gradio/client to call the Space's `/compare` API endpoint. Keeps the
// Space id / token server-side and sidesteps CORS. Configure via env:
//   HF_SPACE_ID     e.g. "shiva-1993/cuad-legal-llm" (or a full Space URL)
//   HF_SPACE_TOKEN  optional, only for a private Space

import { Client } from "@gradio/client";

export async function POST(request: Request) {
  const spaceId = process.env.HF_SPACE_ID;
  if (!spaceId) {
    return Response.json(
      { error: "Inference backend not configured (set HF_SPACE_ID)." },
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
    const token = process.env.HF_SPACE_TOKEN as `hf_${string}` | undefined;
    const client = await Client.connect(spaceId, token ? { token } : {});
    const result = await client.predict("/compare", {
      contract_text: body.contract_text,
      clause_type: body.clause_type ?? "Governing Law",
    });

    const [base, finetuned, baseMs, ftMs] = result.data as [
      string,
      string,
      number,
      number,
    ];
    return Response.json({
      base_output: base,
      finetuned_output: finetuned,
      latency_base_ms: baseMs,
      latency_finetuned_ms: ftMs,
    });
  } catch {
    return Response.json(
      {
        error:
          "Could not reach the inference backend. The ZeroGPU Space may be waking up (cold start) — try again in a moment.",
      },
      { status: 504 },
    );
  }
}

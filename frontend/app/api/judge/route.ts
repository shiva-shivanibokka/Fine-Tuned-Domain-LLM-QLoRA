// BYOK LLM-as-judge. Scores the base and fine-tuned clause extractions on
// faithfulness / completeness / precision using the caller's own API key.
//
// The key is read from the request body, used for exactly one upstream call,
// and never stored, logged, or persisted. Supports Anthropic, OpenAI, Groq,
// and Google Gemini.

type ProviderId = "anthropic" | "openai" | "google" | "groq";

interface JudgeBody {
  provider: ProviderId;
  model: string;
  apiKey: string;
  contract_text: string;
  clause_type: string;
  base_output: string;
  finetuned_output: string;
}

function buildPrompt(b: JudgeBody): string {
  return `You are an expert legal-AI evaluator scoring a contract clause-extraction system.

Contract excerpt:
${b.contract_text.slice(0, 1500)}

Clause type requested: ${b.clause_type}

Output A:
${b.base_output || "(empty)"}

Output B:
${b.finetuned_output || "(empty)"}

Score EACH output 0-10 on:
- faithfulness: does it accurately reflect the contract text?
- completeness: does it capture the full relevant clause?
- precision: is it free of irrelevant or hallucinated content?

Respond with ONLY this JSON, no prose:
{"A":{"faithfulness":n,"completeness":n,"precision":n},"B":{"faithfulness":n,"completeness":n,"precision":n}}`;
}

async function callProvider(b: JudgeBody, prompt: string): Promise<string> {
  if (b.provider === "anthropic") {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": b.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: b.model,
        max_tokens: 300,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    if (!res.ok) throw new Error(`Anthropic ${res.status}: ${await res.text()}`);
    const data = await res.json();
    return data.content?.[0]?.text ?? "";
  }

  if (b.provider === "openai" || b.provider === "groq") {
    const baseUrl =
      b.provider === "openai"
        ? "https://api.openai.com/v1"
        : "https://api.groq.com/openai/v1";
    const res = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${b.apiKey}`,
      },
      body: JSON.stringify({
        model: b.model,
        max_tokens: 300,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    if (!res.ok) throw new Error(`${b.provider} ${res.status}: ${await res.text()}`);
    const data = await res.json();
    return data.choices?.[0]?.message?.content ?? "";
  }

  if (b.provider === "google") {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${b.model}:generateContent?key=${b.apiKey}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
    });
    if (!res.ok) throw new Error(`Google ${res.status}: ${await res.text()}`);
    const data = await res.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  }

  throw new Error("Unknown provider");
}

function parseScores(text: string) {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("Judge did not return JSON");
  const raw = JSON.parse(match[0]);
  const norm = (o: Record<string, unknown>) => ({
    faithfulness: Number(o?.faithfulness ?? 0),
    completeness: Number(o?.completeness ?? 0),
    precision: Number(o?.precision ?? 0),
  });
  return { base: norm(raw.A ?? {}), finetuned: norm(raw.B ?? {}) };
}

export async function POST(request: Request) {
  let b: JudgeBody;
  try {
    b = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }
  if (!b.apiKey || !b.provider || !b.model) {
    return Response.json(
      { error: "provider, model, and apiKey are required." },
      { status: 400 },
    );
  }

  try {
    const text = await callProvider(b, buildPrompt(b));
    return Response.json(parseScores(text));
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Judge request failed.";
    // Surface a short, safe message (may include upstream status but not the key).
    return Response.json({ error: msg.slice(0, 300) }, { status: 502 });
  }
}

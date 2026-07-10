// LLM-judge providers for the BYOK (bring-your-own-key) scoring feature.
// The user's key is sent to our /api/judge route, used for a single request,
// and never stored or logged.

export type ProviderId = "anthropic" | "openai" | "google" | "groq";

export interface Provider {
  id: ProviderId;
  label: string;
  models: string[];
  keyPlaceholder: string;
  keyHint: string;
}

export const PROVIDERS: Provider[] = [
  {
    id: "anthropic",
    label: "Anthropic (Claude)",
    models: ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"],
    keyPlaceholder: "sk-ant-...",
    keyHint: "console.anthropic.com",
  },
  {
    id: "openai",
    label: "OpenAI",
    models: ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    keyPlaceholder: "sk-...",
    keyHint: "platform.openai.com",
  },
  {
    id: "google",
    label: "Google (Gemini)",
    models: ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    keyPlaceholder: "AIza...",
    keyHint: "aistudio.google.com",
  },
  {
    id: "groq",
    label: "Groq",
    models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    keyPlaceholder: "gsk_...",
    keyHint: "console.groq.com (free tier)",
  },
];

export const PROVIDER_MAP: Record<ProviderId, Provider> = Object.fromEntries(
  PROVIDERS.map((p) => [p.id, p]),
) as Record<ProviderId, Provider>;

import metricsJson from "@/data/metrics.json";
import failuresJson from "@/data/failures.json";
import datasetJson from "@/data/dataset_card.json";

export type ModelTag = "base" | "lora" | "qlora" | "dpo";

export interface ModelMetrics {
  bertscore_f1: number;
  geval_faithfulness: number | null;
  clause_presence_accuracy: number;
  hallucination_rate: number;
  ece: number;
  vram_gb: number;
}

export interface Metrics {
  _placeholder?: boolean;
  models: Record<ModelTag, ModelMetrics>;
}

export interface FailurePoint {
  cluster_id: number;
  clause_type: string;
  umap_x: number;
  umap_y: number;
  prediction: string;
  reference: string;
}

export interface Failures {
  _placeholder?: boolean;
  model_tag: string;
  n_failures: number;
  n_clusters: number;
  cluster_labels: Record<string, string>;
  cluster_data: FailurePoint[];
}

export interface DatasetCard {
  dataset: string;
  splits: { train: number; val: number; test: number; total: number };
  clause_distribution: Record<string, number>;
  answer_length_stats: { mean: number; median: number; p95: number };
  context_length_stats: { mean: number; median: number };
}

export const metrics = metricsJson as Metrics;
export const failures = failuresJson as Failures;
export const datasetCard = datasetJson as DatasetCard;

export const MODEL_LABELS: Record<ModelTag, string> = {
  base: "Base (Llama 3.2 3B)",
  lora: "LoRA (bf16)",
  qlora: "QLoRA (4-bit)",
  dpo: "QLoRA + DPO",
};

export const CLAUSE_TYPES = [
  "Governing Law",
  "Termination For Convenience",
  "Limitation Of Liability",
  "Indemnification",
  "Non-Compete",
  "IP Ownership Assignment",
  "Audit Rights",
  "Change Of Control",
  "Most Favored Nation",
  "Anti-Assignment",
];

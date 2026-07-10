import Dashboard from "@/app/components/Dashboard";
import { metrics, failures, datasetCard } from "@/lib/data";

export default function Home() {
  return (
    <Dashboard metrics={metrics} failures={failures} datasetCard={datasetCard} />
  );
}

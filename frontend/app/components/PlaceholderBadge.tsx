export default function PlaceholderBadge() {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-amber-800/40 bg-amber-950/20 px-3 py-2 text-xs text-amber-300/90">
      <span className="inline-block h-2 w-2 rounded-full bg-amber-400" />
      Preview data — these numbers are placeholders until the local training +
      evaluation run populates <code className="text-amber-200">results/</code>.
    </div>
  );
}

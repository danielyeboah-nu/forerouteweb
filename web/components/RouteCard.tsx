import type { Route } from "@/lib/types";
import { RiskBadge } from "./RiskBadge";
import { ClockIcon, RulerIcon } from "./icons";
import { VERDICT_LABEL, VERDICT_TONE } from "@/lib/mlVerdict";

export interface RouteCardProps {
  route: Route;
  selected?: boolean;
  onSelect?: () => void;
}

function formatDistance(m: number): string {
  return `${(m / 1000).toFixed(1)} km`;
}
function formatDuration(s: number): string {
  const mins = Math.round(s / 60);
  if (mins < 60) return `${mins} min`;
  const h = Math.floor(mins / 60);
  const r = mins % 60;
  return `${h}h ${r}m`;
}

const TONE_CLASS: Record<string, string> = {
  good: "bg-emerald-500/15 text-emerald-300 ring-emerald-700",
  neutral: "bg-slate-500/15 text-slate-300 ring-slate-600",
  warn: "bg-amber-500/15 text-amber-300 ring-amber-700",
  bad: "bg-red-500/15 text-red-300 ring-red-700",
  muted: "bg-slate-500/10 text-slate-400 ring-slate-700",
};

export function RouteCard({ route, selected, onSelect }: RouteCardProps) {
  const verdict = route.risk.mlVerdict;
  const tone = verdict ? VERDICT_TONE[verdict] : null;
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left rounded-lg p-4 ring-1 transition ${
        selected
          ? "bg-slate-900 ring-sky-500"
          : "bg-slate-900/60 ring-slate-800 hover:ring-slate-600"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-semibold">{route.name}</h3>
        <RiskBadge level={route.risk.level} value={route.risk.value} />
      </div>
      <div className="mt-2 flex items-center gap-4 text-sm text-slate-400">
        <span className="inline-flex items-center gap-1">
          <RulerIcon width={12} height={12} />
          {formatDistance(route.totalDistance)}
        </span>
        <span className="inline-flex items-center gap-1">
          <ClockIcon width={12} height={12} />
          {formatDuration(route.estimatedDuration)}
        </span>
        <span>{route.segments.length} segments</span>
      </div>
      {verdict && tone ? (
        <div className="mt-2 flex items-center gap-2 text-xs">
          <span className="text-slate-500">Crash history:</span>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 ring-1 ${TONE_CLASS[tone]}`}
          >
            {VERDICT_LABEL[verdict]}
          </span>
        </div>
      ) : null}
    </button>
  );
}

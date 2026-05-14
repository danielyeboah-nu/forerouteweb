import type {
  MlVerdict,
  RiskFactor,
  RiskFactorType,
  RiskLevel,
  RiskScore,
  Route,
  RouteSegment,
  WeatherCondition,
} from "@/lib/types";
import {
  PRECIPITATION_LABEL,
  RISK_FACTOR_LABEL,
  RISK_LEVEL_LABEL,
  ROAD_TYPE_LABEL,
} from "@/lib/labels";
import {
  VERDICT_LABEL,
  VERDICT_DESCRIPTION,
  VERDICT_TONE,
} from "@/lib/mlVerdict";
import { RiskBadge } from "./RiskBadge";
import {
  CheckCircleIcon,
  ClockIcon,
  DropIcon,
  EyeIcon,
  PrecipitationIcon,
  RiskFactorIcon,
  RiskLevelIcon,
  RulerIcon,
  ThermometerIcon,
  WindIcon,
} from "./icons";

const LEVEL_TEXT: Record<RiskLevel, string> = {
  safe: "text-risk-safe",
  caution: "text-risk-caution",
  risky: "text-risk-risky",
  hazardous: "text-risk-hazardous",
};

const LEVEL_BG: Record<RiskLevel, string> = {
  safe: "bg-risk-safe",
  caution: "bg-risk-caution",
  risky: "bg-risk-risky",
  hazardous: "bg-risk-hazardous",
};

const VERDICT_TONE_CLASS: Record<string, string> = {
  good: "bg-emerald-500/15 text-emerald-300 ring-emerald-700",
  neutral: "bg-slate-500/15 text-slate-300 ring-slate-600",
  warn: "bg-amber-500/15 text-amber-300 ring-amber-700",
  bad: "bg-red-500/15 text-red-300 ring-red-700",
  muted: "bg-slate-500/10 text-slate-400 ring-slate-700",
};

function severityColor(severity: number): string {
  if (severity > 0.7) return "bg-risk-hazardous";
  if (severity > 0.4) return "bg-risk-risky";
  return "bg-risk-caution";
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

function VerdictPill({ verdict }: { verdict: MlVerdict }) {
  const tone = VERDICT_TONE[verdict];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ${VERDICT_TONE_CLASS[tone]}`}
    >
      {VERDICT_LABEL[verdict]}
    </span>
  );
}

function ScoreCard({ risk }: { risk: RiskScore }) {
  const pct = Math.max(0, Math.min(100, risk.value));
  return (
    <div className="rounded-lg ring-1 ring-slate-800 bg-slate-900/60 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">
            Weather right now
          </div>
          <div className={`text-2xl font-bold ${LEVEL_TEXT[risk.level]}`}>
            {RISK_LEVEL_LABEL[risk.level]}
          </div>
        </div>
        <RiskLevelIcon
          level={risk.level}
          width={36}
          height={36}
          className={LEVEL_TEXT[risk.level]}
        />
      </div>
      <div className="mt-3 h-2 w-full rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full ${LEVEL_BG[risk.level]}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="mt-1 text-xs text-slate-400 font-mono">
        {Math.round(risk.value)}/100
      </div>
    </div>
  );
}

function PatternRiskCard({ risk }: { risk: RiskScore }) {
  if (!risk.mlVerdict) return null;
  const ratio = risk.mlRatio;
  return (
    <div className="rounded-lg ring-1 ring-slate-800 bg-slate-900/60 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">
            Crash history on this route
          </div>
          <div className="mt-1">
            <VerdictPill verdict={risk.mlVerdict} />
          </div>
        </div>
        {typeof ratio === "number" && risk.mlVerdict !== "outOfRegion" ? (
          <div className="text-right">
            <div className="font-mono text-lg text-slate-200">{ratio.toFixed(1)}×</div>
            <div className="text-xs text-slate-500">compared to other Boston roads</div>
          </div>
        ) : null}
      </div>
      <p className="mt-2 text-xs text-slate-400 leading-relaxed">
        {VERDICT_DESCRIPTION[risk.mlVerdict]}
      </p>
    </div>
  );
}

function ScoresExplainer() {
  return (
    <details className="rounded-md bg-slate-900/40 ring-1 ring-slate-800 p-3 text-sm text-slate-400 group">
      <summary className="cursor-pointer text-slate-300 select-none">
        What do these two scores mean?
      </summary>
      <div className="mt-3 space-y-3 leading-relaxed">
        <p>
          <span className="font-semibold text-slate-200">Weather right now</span>{" "}
          tells you how dangerous the conditions are <em>at this moment</em> —
          snow, ice, rain, wind, fog, and tricky roads like bridges. It tells
          you <span className="italic">how</span> to drive: slow down, leave
          more space, brake earlier.
        </p>
        <p>
          <span className="font-semibold text-slate-200">Crash history</span>{" "}
          tells you how often crashes have actually happened on this stretch of
          road, using real Boston records from the last eight years. It ignores
          today's weather. It tells you{" "}
          <span className="italic">which</span> route to pick.
        </p>
        <p>
          The two can disagree, and that's useful: a snowy bridge at 3 AM
          Saturday is dangerous by weather but quiet by history (nobody's out
          to crash there). A clear Wednesday rush hour through downtown is the
          opposite — calm weather, but a crash-prone spot. We show both so you
          get the full picture.
        </p>
      </div>
    </details>
  );
}

function DisagreementBanner({ risk }: { risk: RiskScore }) {
  const v = risk.mlVerdict;
  if (!v || v === "outOfRegion") return null;
  // Conditions look safe but pattern says danger
  if (risk.level === "safe" && (v === "above" || v === "muchHigher")) {
    return (
      <div className="rounded-md bg-amber-500/10 ring-1 ring-amber-700/50 p-3 text-sm text-amber-200">
        <span className="font-semibold">Heads up:</span> the weather looks fine,
        but{" "}
        {v === "muchHigher"
          ? "this is one of the crashier stretches in Boston"
          : "more crashes happen here than on most Boston roads"}
        . Drive with extra care.
      </div>
    );
  }
  // Conditions look dangerous but pattern says quiet
  if ((risk.level === "risky" || risk.level === "hazardous") && v === "lower") {
    return (
      <div className="rounded-md bg-sky-500/10 ring-1 ring-sky-700/50 p-3 text-sm text-sky-200">
        <span className="font-semibold">Note:</span> the weather is rough, but
        crashes are uncommon along this stretch. Drive carefully — the weather
        is the bigger concern today.
      </div>
    );
  }
  return null;
}

function WhySafeBlock({ route }: { route: Route }) {
  if (route.segments.length === 0) return null;
  const c = route.segments[0].conditions;
  const roadTypes = Array.from(new Set(route.segments.map((s) => s.roadType)));
  const precipDisplay =
    c.precipitationType === "none" || c.precipitationIntensity === 0
      ? "Clear (no precipitation)"
      : `${PRECIPITATION_LABEL[c.precipitationType]} (${c.precipitationIntensity.toFixed(1)} mm/h)`;
  const lines: Array<{ icon: React.ReactNode; text: string }> = [
    {
      icon: <PrecipitationIcon type={c.precipitationType} width={14} height={14} />,
      text: precipDisplay,
    },
    {
      icon: <ThermometerIcon width={14} height={14} />,
      text: `${c.temperature.toFixed(1)} °C`,
    },
    {
      icon: <EyeIcon width={14} height={14} />,
      text: `Visibility ${c.visibility.toFixed(1)} km`,
    },
    {
      icon: <WindIcon width={14} height={14} />,
      text: `Wind ${c.windSpeed.toFixed(0)} km/h`,
    },
    {
      icon: <RulerIcon width={14} height={14} />,
      text: `Road types: ${roadTypes.map((r) => ROAD_TYPE_LABEL[r]).join(", ")}`,
    },
  ];
  return (
    <div className="rounded-md bg-emerald-500/5 ring-1 ring-emerald-900/40 p-3">
      <div className="flex items-center gap-2 text-emerald-300 font-medium text-sm mb-2">
        <CheckCircleIcon width={16} height={16} />
        No weather hazards detected
      </div>
      <ul className="space-y-1.5 text-sm text-slate-300">
        {lines.map((l, i) => (
          <li key={i} className="flex items-center gap-2">
            <span className="text-slate-500">{l.icon}</span>
            <span>{l.text}</span>
          </li>
        ))}
      </ul>
      <details className="mt-3 text-xs text-slate-400 group">
        <summary className="cursor-pointer text-slate-300 select-none">
          When does this score get worse?
        </summary>
        <ul className="mt-2 space-y-1 pl-4 list-disc">
          <li>Snow, sleet, or freezing rain starts falling</li>
          <li>Heavy rain — enough to risk hydroplaning</li>
          <li>Temperatures near freezing while roads are wet — ice can form</li>
          <li>Fog or heavy weather drops visibility below a kilometre</li>
          <li>Strong winds, the kind that push your car around</li>
          <li>Your route crosses a bridge, tunnel, or mountain road</li>
        </ul>
      </details>
    </div>
  );
}

function FactorRow({ factor }: { factor: RiskFactor }) {
  const pct = Math.round(factor.severity * 100);
  return (
    <li className="flex items-start gap-3 rounded-md bg-slate-900/60 ring-1 ring-slate-800 p-3">
      <RiskFactorIcon
        type={factor.type as RiskFactorType}
        width={20}
        height={20}
        className="mt-0.5 text-sky-300 shrink-0"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">
            {RISK_FACTOR_LABEL[factor.type as RiskFactorType]}
          </span>
          <span className="font-mono text-slate-400">{pct}%</span>
        </div>
        <div className="mt-1 h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
          <div
            className={`h-full ${severityColor(factor.severity)}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="mt-1 text-xs text-slate-500">{factor.description}</p>
      </div>
    </li>
  );
}

function WeatherCard({ c }: { c: WeatherCondition }) {
  return (
    <div className="rounded-md bg-slate-900/60 ring-1 ring-slate-800 p-3">
      <div className="flex items-center gap-3">
        <PrecipitationIcon
          type={c.precipitationType}
          width={28}
          height={28}
          className="text-sky-300"
        />
        <div>
          <div className="font-medium">
            {PRECIPITATION_LABEL[c.precipitationType]}
          </div>
          <div className="text-xs text-slate-400 flex items-center gap-1">
            <ThermometerIcon width={12} height={12} />
            {c.temperature.toFixed(1)} °C
          </div>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <Metric icon={<WindIcon width={12} height={12} />} label="Wind">
          {c.windSpeed.toFixed(0)} km/h
        </Metric>
        <Metric icon={<EyeIcon width={12} height={12} />} label="Visibility">
          {c.visibility.toFixed(1)} km
        </Metric>
        <Metric icon={<DropIcon width={12} height={12} />} label="Precip">
          {c.precipitationIntensity.toFixed(1)} mm/h
        </Metric>
      </div>
    </div>
  );
}

function Metric({
  icon,
  label,
  children,
}: {
  icon: React.ReactNode;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-1 text-slate-500">
        {icon}
        <span>{label}</span>
      </div>
      <div className="font-semibold mt-0.5">{children}</div>
    </div>
  );
}

function SegmentRow({ seg, idx }: { seg: RouteSegment; idx: number }) {
  return (
    <li className="rounded-md bg-slate-900/60 ring-1 ring-slate-800 p-3 text-sm">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <span className="font-medium">
          Segment {idx + 1} · {ROAD_TYPE_LABEL[seg.roadType]}
        </span>
        <div className="flex items-center gap-2">
          {seg.risk.mlVerdict ? <VerdictPill verdict={seg.risk.mlVerdict} /> : null}
          <RiskBadge level={seg.risk.level} value={seg.risk.value} />
        </div>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400">
        <span className="inline-flex items-center gap-1">
          <ThermometerIcon width={12} height={12} />
          {seg.conditions.temperature.toFixed(1)} °C
        </span>
        <span className="inline-flex items-center gap-1">
          <PrecipitationIcon type={seg.conditions.precipitationType} width={12} height={12} />
          {PRECIPITATION_LABEL[seg.conditions.precipitationType]}
          {seg.conditions.precipitationIntensity > 0
            ? ` (${seg.conditions.precipitationIntensity.toFixed(1)} mm/h)`
            : ""}
        </span>
        <span className="inline-flex items-center gap-1">
          <WindIcon width={12} height={12} />
          {seg.conditions.windSpeed.toFixed(0)} km/h
        </span>
        <span className="inline-flex items-center gap-1">
          <EyeIcon width={12} height={12} />
          {seg.conditions.visibility.toFixed(1)} km
        </span>
        <span>Humidity {seg.conditions.humidity.toFixed(0)}%</span>
        <span className="inline-flex items-center gap-1">
          <RulerIcon width={12} height={12} />
          {(seg.distance / 1000).toFixed(1)} km
        </span>
      </div>
    </li>
  );
}

export function RouteDetail({ route }: { route: Route }) {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold">{route.name}</h2>
        <p className="text-sm text-slate-400">
          {route.start.name ?? "Origin"} → {route.end.name ?? "Destination"}
        </p>
        <div className="mt-2 flex items-center gap-4 text-sm text-slate-300">
          <span className="inline-flex items-center gap-1">
            <RulerIcon width={14} height={14} />
            {formatDistance(route.totalDistance)}
          </span>
          <span className="inline-flex items-center gap-1">
            <ClockIcon width={14} height={14} />
            {formatDuration(route.estimatedDuration)}
          </span>
        </div>
      </div>

      <DisagreementBanner risk={route.risk} />

      <div className="grid gap-3 sm:grid-cols-2">
        <ScoreCard risk={route.risk} />
        <PatternRiskCard risk={route.risk} />
      </div>

      <ScoresExplainer />

      {route.risk.factors.length > 0 ? (
        <section>
          <h3 className="text-sm uppercase tracking-wide text-slate-400 mb-2">
            Risk factors
          </h3>
          <ul className="space-y-2">
            {route.risk.factors.map((f) => (
              <FactorRow key={f.type} factor={f} />
            ))}
          </ul>
        </section>
      ) : (
        <WhySafeBlock route={route} />
      )}

      {route.segments.length > 0 ? (
        <section>
          <h3 className="text-sm uppercase tracking-wide text-slate-400 mb-2">
            Weather conditions
          </h3>
          <div className="grid gap-2 sm:grid-cols-2">
            {route.segments.map((s) => (
              <WeatherCard key={`w-${s.id}`} c={s.conditions} />
            ))}
          </div>
        </section>
      ) : null}

      <section>
        <h3 className="text-sm uppercase tracking-wide text-slate-400 mb-2">
          Route breakdown
        </h3>
        <ul className="space-y-2">
          {route.segments.map((s, i) => (
            <SegmentRow key={s.id} seg={s} idx={i} />
          ))}
        </ul>
      </section>
    </div>
  );
}

import type { RiskLevel } from "@/lib/types";
import { RISK_LEVEL_LABEL } from "@/lib/labels";
import { RiskLevelIcon } from "./icons";

const COLOR: Record<RiskLevel, string> = {
  safe: "bg-risk-safe/15 text-risk-safe ring-risk-safe/40",
  caution: "bg-risk-caution/15 text-risk-caution ring-risk-caution/40",
  risky: "bg-risk-risky/15 text-risk-risky ring-risk-risky/40",
  hazardous: "bg-risk-hazardous/15 text-risk-hazardous ring-risk-hazardous/40",
};

export function RiskBadge({ level, value }: { level: RiskLevel; value: number }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${COLOR[level]}`}
    >
      <RiskLevelIcon level={level} width={12} height={12} />
      {RISK_LEVEL_LABEL[level]}
      <span className="opacity-60 font-mono">{value.toFixed(0)}</span>
    </span>
  );
}

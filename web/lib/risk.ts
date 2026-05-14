import type {
  RiskFactor,
  RiskLevel,
  RiskScore,
  RoadType,
  RouteSegment,
  WeatherCondition,
} from "./types";

const ROAD_RISK_MULTIPLIER: Record<RoadType, number> = {
  highway: 1.0,
  arterial: 0.9,
  residential: 0.8,
  bridge: 1.5,
  tunnel: 1.2,
  mountain: 1.4,
};

export function roadMultiplier(road: RoadType): number {
  return ROAD_RISK_MULTIPLIER[road];
}

export function levelFor(value: number): RiskLevel {
  if (value < 30) return "safe";
  if (value < 60) return "caution";
  if (value < 80) return "risky";
  return "hazardous";
}

function hasIceRisk(c: WeatherCondition): boolean {
  return (
    c.temperature <= 2.0 &&
    (c.precipitationType === "rain" || c.precipitationType === "snow" || c.humidity > 80)
  );
}

function hasSnowRisk(c: WeatherCondition): boolean {
  return c.precipitationType === "snow" && c.precipitationIntensity > 0;
}

function hasRainRisk(c: WeatherCondition): boolean {
  return c.precipitationType === "rain" && c.precipitationIntensity > 2.5;
}

function hasLowVisibility(c: WeatherCondition): boolean {
  return c.visibility < 1.0;
}

export function scoreSegment(
  conditions: WeatherCondition,
  roadType: RoadType
): RiskScore {
  let value = 0;
  const factors: RiskFactor[] = [];

  if (hasSnowRisk(conditions)) {
    const severity = Math.min(conditions.precipitationIntensity / 5.0, 1.0);
    value += 30 * severity;
    factors.push({
      type: "snow",
      severity,
      description: `${conditions.precipitationIntensity.toFixed(1)} mm/h snow expected`,
    });
  }

  if (hasIceRisk(conditions)) {
    const severity =
      conditions.temperature <= 0 ? 1.0 : (2.0 - conditions.temperature) / 2.0;
    value += 25 * severity;
    factors.push({
      type: "ice",
      severity,
      description: `Temperature near freezing (${conditions.temperature.toFixed(1)}°C) — ice risk`,
    });
  }

  if (hasRainRisk(conditions)) {
    const severity = Math.min(conditions.precipitationIntensity / 10.0, 1.0);
    value += 20 * severity;
    factors.push({
      type: "rain",
      severity,
      description: `Heavy rain (${conditions.precipitationIntensity.toFixed(1)} mm/h) — hydroplaning risk`,
    });
  }

  if (hasLowVisibility(conditions)) {
    const severity = 1.0 - conditions.visibility / 1.0;
    value += 15 * severity;
    factors.push({
      type: "lowVisibility",
      severity,
      description: `Low visibility (${conditions.visibility.toFixed(1)} km)`,
    });
  }

  if (conditions.windSpeed > 40) {
    const severity = Math.min((conditions.windSpeed - 40) / 40.0, 1.0);
    value += 10 * severity;
    factors.push({
      type: "wind",
      severity,
      description: `High winds (${conditions.windSpeed.toFixed(0)} km/h)`,
    });
  }

  const mult = roadMultiplier(roadType);
  value *= mult;
  if (mult > 1.0) {
    factors.push({
      type: "roadType",
      severity: (mult - 1.0) / 0.5,
      description: `${roadType} — higher risk in bad weather`,
    });
  }

  value = Math.min(value, 100);
  return { value, level: levelFor(value), factors };
}

export function scoreRoute(segments: RouteSegment[]): RiskScore {
  if (segments.length === 0) {
    return { value: 0, level: "safe", factors: [] };
  }
  const total = segments.reduce((acc, s) => acc + s.distance, 0) || 1;
  let value = 0;
  const byType = new Map<string, RiskFactor>();
  for (const s of segments) {
    const weight = s.distance / total;
    value += s.risk.value * weight;
    for (const f of s.risk.factors) {
      const existing = byType.get(f.type);
      if (!existing || f.severity > existing.severity) byType.set(f.type, f);
    }
  }
  const factors = Array.from(byType.values()).sort((a, b) => b.severity - a.severity);
  return { value, level: levelFor(value), factors };
}

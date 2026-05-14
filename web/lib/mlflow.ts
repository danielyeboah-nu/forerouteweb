import type { RoadType, WeatherCondition } from "./types";

const FEATURE_COLS = [
  "temperature",
  "precipitation_type",
  "precipitation_intensity",
  "wind_speed",
  "visibility",
  "humidity",
  "dew_point",
  "road_type",
  "segment_distance_m",
  "lat",
  "lon",
  "hour_of_day",
  "day_of_week",
  "month",
  "prior_year_crash_count",
] as const;

export interface MlInput {
  conditions: WeatherCondition;
  roadType: RoadType;
  distance: number;
  lat: number;
  lon: number;
  hourOfDay: number;
  dayOfWeek: number; // pandas convention: 0=Mon..6=Sun
  month: number; // 1..12
  priorYearCrashCount: number;
}

export interface MlResult {
  prediction: 0 | 1;
  probability?: number;
}

function toRow(input: MlInput): unknown[] {
  return [
    input.conditions.temperature,
    input.conditions.precipitationType,
    input.conditions.precipitationIntensity,
    input.conditions.windSpeed,
    input.conditions.visibility,
    input.conditions.humidity,
    input.conditions.dewPoint,
    input.roadType,
    input.distance,
    input.lat,
    input.lon,
    input.hourOfDay,
    input.dayOfWeek,
    input.month,
    input.priorYearCrashCount,
  ];
}

export async function scoreBatch(
  serveUrl: string,
  inputs: MlInput[]
): Promise<MlResult[]> {
  if (inputs.length === 0) return [];
  const body = {
    dataframe_split: {
      columns: FEATURE_COLS,
      data: inputs.map(toRow),
    },
  };
  const res = await fetch(serveUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`MLflow ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as
    | { predictions: number[] }
    | number[];
  const preds = Array.isArray(data) ? data : data.predictions;
  return preds.map((p) => ({ prediction: p >= 0.5 ? 1 : 0, probability: typeof p === "number" ? p : undefined }));
}

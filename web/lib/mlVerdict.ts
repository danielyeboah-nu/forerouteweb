// Client-safe constants and pure helpers for the ML verdict layer.
// Server-only annotation + threshold-loading lives in mlVerdictServer.ts.

import type { MlVerdict } from "./types";

// Training bbox (must match BBOX_LAT / BBOX_LON in ml/build_boston_dataset.py)
export const TRAINING_BBOX = {
  latMin: 42.2,
  latMax: 42.45,
  lngMin: -71.2,
  lngMax: -70.95,
};

export function inTrainingRegion(lat: number, lng: number): boolean {
  return (
    lat >= TRAINING_BBOX.latMin &&
    lat <= TRAINING_BBOX.latMax &&
    lng >= TRAINING_BBOX.lngMin &&
    lng <= TRAINING_BBOX.lngMax
  );
}

export interface VerdictThresholds {
  /** below this probability → "lower" (bottom 25% of negatives) */
  lower_max: number;
  /** below this probability → "typical" (middle 50%) */
  typical_max: number;
  /** below this probability → "above" (75–90 percentile) */
  above_max: number;
  /** model's predicted probability on a typical Boston negative — used for the ratio caption */
  median: number;
}

/** Fallback thresholds used when verdict_thresholds.json isn't shipped. */
export const DEFAULT_THRESHOLDS: VerdictThresholds = {
  lower_max: 0.284,
  typical_max: 0.463,
  above_max: 0.514,
  median: 0.402,
};

/** Map a raw probability to a verdict bucket using calibrated thresholds. */
export function verdictForProbability(
  probability: number,
  inRegion: boolean,
  thresholds: VerdictThresholds = DEFAULT_THRESHOLDS
): MlVerdict {
  if (!inRegion) return "outOfRegion";
  if (probability < thresholds.lower_max) return "lower";
  if (probability < thresholds.typical_max) return "typical";
  if (probability < thresholds.above_max) return "above";
  return "muchHigher";
}

export const VERDICT_LABEL: Record<MlVerdict, string> = {
  outOfRegion: "Not enough data",
  lower: "Fewer crashes than usual",
  typical: "About average",
  above: "More crashes than usual",
  muchHigher: "Crashy stretch",
};

export const VERDICT_DESCRIPTION: Record<MlVerdict, string> = {
  outOfRegion:
    "We don't have crash data for this stretch of road, so we can't tell you how it compares. Use the weather score above to decide.",
  lower:
    "Fewer crashes have happened here than on most Boston roads at this time of day.",
  typical:
    "About the same number of crashes happen here as on most Boston roads — nothing unusual.",
  above:
    "More crashes have happened on this stretch of road than on most Boston roads. Drive with extra care.",
  muchHigher:
    "This is one of the crashier stretches in Boston. Slow down and stay focused, even if the weather looks fine.",
};

export const VERDICT_TONE: Record<
  MlVerdict,
  "neutral" | "good" | "warn" | "bad" | "muted"
> = {
  outOfRegion: "muted",
  lower: "good",
  typical: "neutral",
  above: "warn",
  muchHigher: "bad",
};

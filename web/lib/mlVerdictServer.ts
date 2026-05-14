// Server-only helpers: loads calibrated verdict thresholds from disk and
// annotates routes with absolute-probability verdicts + median-relative ratios.

import fs from "fs";
import path from "path";

import type { Route } from "./types";
import {
  DEFAULT_THRESHOLDS,
  inTrainingRegion,
  verdictForProbability,
  type VerdictThresholds,
} from "./mlVerdict";

interface CalibrationPayload {
  source?: string;
  model_alias?: string;
  quantiles?: Record<string, number>;
  thresholds: VerdictThresholds;
}

// Cached per file-mtime so an updated calibration takes effect without a
// dev-server restart, but we don't re-read every request unnecessarily.
let cached: { path: string; mtimeMs: number; data: CalibrationPayload } | null = null;
let warnedMissing = false;

function load(): CalibrationPayload {
  const candidates = [
    path.resolve(process.cwd(), "../ml/data/verdict_thresholds.json"),
    path.resolve(process.cwd(), "ml/data/verdict_thresholds.json"),
    path.resolve(process.cwd(), "lib/data/verdict_thresholds.json"),
  ];
  for (const p of candidates) {
    try {
      const stat = fs.statSync(p);
      if (cached && cached.path === p && cached.mtimeMs === stat.mtimeMs) {
        return cached.data;
      }
      const data = JSON.parse(fs.readFileSync(p, "utf-8")) as CalibrationPayload;
      cached = { path: p, mtimeMs: stat.mtimeMs, data };
      const t = data.thresholds;
      console.log(
        `[mlVerdict] loaded thresholds (lower<${t.lower_max.toFixed(3)} ` +
          `typical<${t.typical_max.toFixed(3)} above<${t.above_max.toFixed(3)} ` +
          `median=${t.median.toFixed(3)}) from ${p}`
      );
      return data;
    } catch {
      // file doesn't exist or parse failed — try next candidate
    }
  }
  if (!warnedMissing) {
    console.warn(
      "[mlVerdict] no verdict_thresholds.json found — using defaults. " +
        "Run `cd ml && python3 calibrate_thresholds.py` to refresh."
    );
    warnedMissing = true;
  }
  return { thresholds: DEFAULT_THRESHOLDS };
}

/**
 * For each segment and each route-level prediction, assign:
 *   - mlVerdict via absolute probability thresholds (calibrated quantiles)
 *   - mlRatio = probability / median (the "1.4× a typical Boston road" caption)
 *
 * Out-of-region segments are tagged regardless of probability.
 */
export function annotateMlVerdicts(routes: Route[]): void {
  const { thresholds } = load();
  for (const r of routes) {
    let anyInRegion = false;
    for (const s of r.segments) {
      if (typeof s.risk.mlProbability !== "number") continue;
      const midLat = (s.start.lat + s.end.lat) / 2;
      const midLng = (s.start.lng + s.end.lng) / 2;
      const inRegion = inTrainingRegion(midLat, midLng);
      if (inRegion) anyInRegion = true;
      s.risk.mlVerdict = verdictForProbability(
        s.risk.mlProbability,
        inRegion,
        thresholds
      );
      s.risk.mlRatio = s.risk.mlProbability / thresholds.median;
    }
    if (typeof r.risk.mlProbability === "number") {
      r.risk.mlVerdict = verdictForProbability(
        r.risk.mlProbability,
        anyInRegion,
        thresholds
      );
      r.risk.mlRatio = r.risk.mlProbability / thresholds.median;
    }
  }
}

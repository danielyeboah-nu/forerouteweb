import type {
  PrecipitationType,
  RiskFactorType,
  RiskLevel,
  RoadType,
} from "./types";

export const ROAD_TYPE_LABEL: Record<RoadType, string> = {
  highway: "Highway",
  arterial: "Arterial",
  residential: "Residential",
  bridge: "Bridge",
  tunnel: "Tunnel",
  mountain: "Mountain",
};

export const RISK_FACTOR_LABEL: Record<RiskFactorType, string> = {
  snow: "Snow",
  rain: "Rain",
  ice: "Ice Risk",
  lowVisibility: "Low Visibility",
  wind: "High Winds",
  temperature: "Temperature",
  roadType: "Road Type",
};

export const PRECIPITATION_LABEL: Record<PrecipitationType, string> = {
  none: "Clear",
  rain: "Rain",
  snow: "Snow",
  sleet: "Sleet",
  freezingRain: "Freezing Rain",
};

export const RISK_LEVEL_LABEL: Record<RiskLevel, string> = {
  safe: "Safe",
  caution: "Caution",
  risky: "Risky",
  hazardous: "Hazardous",
};

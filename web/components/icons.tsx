import type { SVGProps } from "react";
import type {
  PrecipitationType,
  RiskFactorType,
  RiskLevel,
} from "@/lib/types";

type IconProps = SVGProps<SVGSVGElement>;

const base = (props: IconProps) => ({
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  ...props,
});

export function SnowflakeIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <line x1="12" y1="2" x2="12" y2="22" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
      <line x1="19.07" y1="4.93" x2="4.93" y2="19.07" />
    </svg>
  );
}

export function RainIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M17 14a5 5 0 1 0-9.58-2 4 4 0 0 0-.42 7.94" />
      <line x1="8" y1="19" x2="8" y2="22" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="16" y1="19" x2="16" y2="22" />
    </svg>
  );
}

export function SunIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="4" />
      <line x1="12" y1="2" x2="12" y2="5" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="2" y1="12" x2="5" y2="12" />
      <line x1="19" y1="12" x2="22" y2="12" />
      <line x1="4.93" y1="4.93" x2="7" y2="7" />
      <line x1="17" y1="17" x2="19.07" y2="19.07" />
      <line x1="4.93" y1="19.07" x2="7" y2="17" />
      <line x1="17" y1="7" x2="19.07" y2="4.93" />
    </svg>
  );
}

export function SleetIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M17 12a5 5 0 1 0-9.58-2 4 4 0 0 0-.42 7.94" />
      <line x1="8" y1="18" x2="8" y2="20" />
      <line x1="16" y1="18" x2="16" y2="20" />
      <circle cx="12" cy="20" r="0.8" />
    </svg>
  );
}

export function FreezingRainIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M17 12a5 5 0 1 0-9.58-2 4 4 0 0 0-.42 7.94" />
      <path d="M12 16v6" />
      <path d="M9 19l3 3 3-3" />
    </svg>
  );
}

export function IceIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="9" />
      <line x1="12" y1="6" x2="12" y2="18" />
      <line x1="6" y1="12" x2="18" y2="12" />
      <line x1="8" y1="8" x2="16" y2="16" />
      <line x1="16" y1="8" x2="8" y2="16" />
    </svg>
  );
}

export function WindIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M3 8h11a3 3 0 1 0-3-3" />
      <path d="M3 16h15a3 3 0 1 1-3 3" />
      <path d="M3 12h9" />
    </svg>
  );
}

export function EyeSlashIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M2 12s3.5-7 10-7c2 0 3.7.6 5.1 1.5" />
      <path d="M22 12s-3.5 7-10 7c-2 0-3.7-.6-5.1-1.5" />
      <line x1="3" y1="3" x2="21" y2="21" />
    </svg>
  );
}

export function EyeIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export function ThermometerIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M14 14V5a2 2 0 1 0-4 0v9a4 4 0 1 0 4 0z" />
    </svg>
  );
}

export function RoadIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M5 21l3-18" />
      <path d="M19 21l-3-18" />
      <line x1="12" y1="4" x2="12" y2="6" />
      <line x1="12" y1="10" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12" y2="20" />
    </svg>
  );
}

export function CheckCircleIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="10" />
      <path d="M8 12l3 3 5-6" />
    </svg>
  );
}

export function WarnTriangleIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M10.3 3.9 2.5 17a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <circle cx="12" cy="17" r="0.6" fill="currentColor" />
    </svg>
  );
}

export function OctagonIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <polygon points="8 2 16 2 22 8 22 16 16 22 8 22 2 16 2 8" />
      <line x1="12" y1="8" x2="12" y2="13" />
      <circle cx="12" cy="16" r="0.6" fill="currentColor" />
    </svg>
  );
}

export function XCircleIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="10" />
      <line x1="9" y1="9" x2="15" y2="15" />
      <line x1="15" y1="9" x2="9" y2="15" />
    </svg>
  );
}

export function DropIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M12 2.5s6 7 6 11.5a6 6 0 1 1-12 0c0-4.5 6-11.5 6-11.5z" />
    </svg>
  );
}

export function RulerIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M3 8l13 13 5-5L8 3 3 8z" />
      <line x1="7" y1="8" x2="9" y2="10" />
      <line x1="10" y1="11" x2="12" y2="13" />
      <line x1="13" y1="14" x2="15" y2="16" />
    </svg>
  );
}

export function ClockIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="9" />
      <polyline points="12 7 12 12 15 14" />
    </svg>
  );
}

export function PrecipitationIcon({
  type,
  ...props
}: { type: PrecipitationType } & IconProps) {
  switch (type) {
    case "snow":
      return <SnowflakeIcon {...props} />;
    case "rain":
      return <RainIcon {...props} />;
    case "sleet":
      return <SleetIcon {...props} />;
    case "freezingRain":
      return <FreezingRainIcon {...props} />;
    case "none":
    default:
      return <SunIcon {...props} />;
  }
}

export function RiskFactorIcon({
  type,
  ...props
}: { type: RiskFactorType } & IconProps) {
  switch (type) {
    case "snow":
      return <SnowflakeIcon {...props} />;
    case "rain":
      return <RainIcon {...props} />;
    case "ice":
      return <IceIcon {...props} />;
    case "lowVisibility":
      return <EyeSlashIcon {...props} />;
    case "wind":
      return <WindIcon {...props} />;
    case "temperature":
      return <ThermometerIcon {...props} />;
    case "roadType":
      return <RoadIcon {...props} />;
  }
}

export function RiskLevelIcon({
  level,
  ...props
}: { level: RiskLevel } & IconProps) {
  switch (level) {
    case "safe":
      return <CheckCircleIcon {...props} />;
    case "caution":
      return <WarnTriangleIcon {...props} />;
    case "risky":
      return <OctagonIcon {...props} />;
    case "hazardous":
      return <XCircleIcon {...props} />;
  }
}

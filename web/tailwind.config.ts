import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        risk: {
          safe: "#22c55e",
          caution: "#eab308",
          risky: "#f97316",
          hazardous: "#ef4444",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;

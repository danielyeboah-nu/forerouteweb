import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ForeRoute — weather-aware safe routes",
  description: "Pick the safest driving route, not just the fastest.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}

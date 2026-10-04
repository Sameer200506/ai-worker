import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "TaskPilot",
  description: "Autonomous AI task worker",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KiCad AI Pipeline",
  description: "AI-assisted PCB design: from requirements to manufacturing files",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}

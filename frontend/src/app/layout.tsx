import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import { SiteNav } from "@/components/SiteNav";
import { QueryProvider } from "@/lib/QueryProvider";
import "./globals.css";

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  style: ["normal", "italic"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: { default: "MAiSTRO", template: "%s" },
  description: "An LSTM that composes classical piano, and the tools to steer and judge it.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${playfair.variable} ${inter.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <QueryProvider>
          <SiteNav />
          <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-12">{children}</main>
          <footer className="border-t border-border">
            <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-6 py-8 text-xs text-muted-foreground">
              <span>MAiSTRO · Western Cyber Society</span>
              <span>Trained on Mozart, Beethoven and Chopin. MIT licensed.</span>
            </div>
          </footer>
        </QueryProvider>
      </body>
    </html>
  );
}

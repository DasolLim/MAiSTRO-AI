import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import { ProgramNav } from "@/components/ProgramNav";
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
  title: "MAiSTRO",
  description: "AI classical music composition studio",
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
          <ProgramNav />
          <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-12">{children}</main>
        </QueryProvider>
      </body>
    </html>
  );
}

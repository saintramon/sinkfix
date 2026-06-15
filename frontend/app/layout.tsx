import Link from "next/link";
import type { ReactNode } from "react";
import { Analytics } from "@vercel/analytics/next";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/600.css";
import "@fontsource/jetbrains-mono/700.css";
import "./globals.css";

export default function RootLayout({ children }: { children: ReactNode }) {
    return (
        <html lang="en">
            <body>
                <header className="border-b border-[#2a2a2a] bg-[#0a0a0a]/90 px-4 py-4 text-[#e3e2e2] backdrop-blur sm:px-6 lg:px-8">
                    <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
                        <Link href="/" className="group">
                            <p className="font-mono text-lg font-semibold tracking-tight text-[#ffb4a8] transition group-hover:text-white">
                                SinkFix
                            </p>
                            <p className="font-mono text-xs uppercase tracking-[0.18em] text-[#cc0000]">
                                Prototype 
                            </p>
                        </Link>

                        <nav className="flex items-center gap-2 font-mono text-sm">
                            <Link
                                href="/"
                                className="rounded-md border border-[#2a2a2a] px-3 py-2 text-[#c8c6c5] transition hover:border-[#cc0000] hover:text-[#ffb4a8]"
                            >
                                Input
                            </Link>
                            <Link
                                href="/results"
                                className="rounded-md border border-[#2a2a2a] px-3 py-2 text-[#c8c6c5] transition hover:border-[#cc0000] hover:text-[#ffb4a8]"
                            >
                                Results
                            </Link>
                        </nav>
                    </div>
                </header>

                {children}
                <Analytics />
            </body>
        </html>
    );
}

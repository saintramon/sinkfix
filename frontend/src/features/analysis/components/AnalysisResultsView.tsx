"use client";

import Link from "next/link";
import { useMemo, useSyncExternalStore } from "react";
import { AnalysisResult } from "./AnalysisResult";
import type { AnalyzeResponse } from "../types/analysis";

const LAST_ANALYSIS_RESULT_KEY = "sinkfix:lastAnalysisResult";

function subscribeToStorageUpdates(callback: () => void) {
    window.addEventListener("storage", callback);

    return () => {
        window.removeEventListener("storage", callback);
    };
}

function getStoredResult() {
    if (typeof window === "undefined") {
        return null;
    }

    return window.sessionStorage.getItem(LAST_ANALYSIS_RESULT_KEY);
}

function parseStoredResult(storedResult: string | null): AnalyzeResponse | null {
    if (!storedResult) {
        return null;
    }

    try {
        return JSON.parse(storedResult) as AnalyzeResponse;
    } catch {
        return null;
    }
}

export function AnalysisResultsView() {
    const storedResult = useSyncExternalStore(
        subscribeToStorageUpdates,
        getStoredResult,
        () => null,
    );
    const result = useMemo(() => parseStoredResult(storedResult), [storedResult]);

    if (!result) {
        return (
            <div className="rounded-md border border-dashed border-[#343535] bg-[#0d0e0f] p-6">
                <p className="font-mono text-sm font-medium text-[#e3e2e2]">No diagnostic report found</p>
                <p className="mt-2 text-sm leading-6 text-[#c8c6c5]">
                    Run an analysis from the input page to generate a token-level attention report.
                </p>
                <Link
                    href="/"
                    className="mt-5 inline-flex rounded-md border border-[#cc0000] bg-[#cc0000] px-4 py-2 font-mono text-sm font-semibold uppercase tracking-[0.16em] text-white transition hover:bg-[#a80000]"
                >
                    Go to input
                </Link>
            </div>
        );
    }

    return <AnalysisResult result={result} />;
}

import { AnalysisResultsView } from "@/src/features/analysis/components/AnalysisResultsView";

export default function ResultsPage() {
    return (
        <main className="min-h-[calc(100vh-81px)] px-4 py-10 text-[#e3e2e2] sm:px-6 lg:px-8">
            <div className="mx-auto max-w-6xl space-y-6">
                <section className="flex flex-col gap-3 border-b border-[#2a2a2a] pb-6 sm:flex-row sm:items-end sm:justify-between">
                    <div>
                        <p className="font-mono text-xs font-semibold uppercase tracking-[0.24em] text-[#cc0000]">
                            Diagnostic output
                        </p>
                        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-[#ffb4a8] sm:text-4xl">
                            Results
                        </h1>
                    </div>
                    <p className="max-w-xl text-sm leading-6 text-[#c8c6c5]">
                        Review the latest token-level attention report generated from the input page.
                    </p>
                </section>

                <AnalysisResultsView />
            </div>
        </main>
    );
}

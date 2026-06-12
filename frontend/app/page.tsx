import { AnalysisWorkspace } from "@/src/features/analysis/components/AnalysisWorkspace";

export default function HomePage() {
    return (
        <main className="flex min-h-[calc(100vh-81px)] items-center px-4 py-10 text-[#e3e2e2] sm:px-6 lg:px-8">
            <div className="mx-auto grid w-full max-w-5xl gap-8 lg:grid-cols-[minmax(0,0.9fr)_minmax(360px,520px)] lg:items-center">
                <section className="space-y-5">
                    <p className="font-mono text-xs font-semibold uppercase tracking-[0.24em] text-[#cc0000]">
                        Attention diagnostics
                    </p>
                    <div className="space-y-3">
                        <h1 className="text-4xl font-semibold tracking-tight text-[#ffb4a8] sm:text-5xl">
                            Inspect transformer attention sinks.
                        </h1>
                        <p className="max-w-xl text-sm leading-7 text-[#c8c6c5] sm:text-base">
                            SinkFix analyzes BERT attention patterns, identifies high-attention
                            tokens, and turns the result into a focused explainability report.
                        </p>
                    </div>
                    <div className="border-l-2 border-[#cc0000] pl-4 text-sm leading-6 text-[#c8c6c5]">
                        This prototype explains model behavior. It does not fine-tune,
                        retrain, or claim to repair BERT.
                    </div>
                </section>

                <section>
                    <AnalysisWorkspace />
                </section>
            </div>
        </main>
    );
}

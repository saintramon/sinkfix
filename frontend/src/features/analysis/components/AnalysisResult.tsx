import { AnalyzeResponse, SinkClassification, TokenAnalysisRow } from "../types/analysis";
import { AttentionHeatmap } from "./AttentionHeatmap";


type AnalyzeResponseProps = {
    result: AnalyzeResponse;
};

const classificationStyles: Record<SinkClassification, string> = {
    beneficial: "bg-green-950/50 text-green-300 ring-green-500/30",
    neutral: "bg-[#1f2020] text-[#c8c6c5] ring-[#343535]",
    detrimental: "bg-[#cc00001a] text-[#ffb4a8] ring-[#cc000033]",
};

export function AnalysisResult({ result }: AnalyzeResponseProps) { 
    const rows: TokenAnalysisRow[] = result.token_list.map((token, index) => ({
        index,
        token,
        classification: result.classifications[index],
        attentionReceived: result.att_received_scores[index],
        valueNorm: result.value_norms[index],
    }));

    const strongestSink = rows.length
        ? rows.reduce((strongest, row) => {
            if (row.attentionReceived > strongest.attentionReceived) { 
                return row;
            }

            return strongest;
        }, rows[0])
        : null;

    const topSinks = [...rows]
        .sort((a, b) => b.attentionReceived - a.attentionReceived)
        .slice(0, 5);

    const summary = rows.reduce(
        (counts, row) => ({
            ...counts,
            [row.classification]: counts[row.classification] + 1,
        }),
        {
            beneficial: 0,
            neutral: 0,
            detrimental: 0,
        } satisfies Record<SinkClassification, number>,
    );

    return (
        <section className="space-y-4 rounded-md border border-[#2a2a2a] bg-[#0d0e0f] p-5">
            <div className="flex flex-col gap-1 border-b border-[#2a2a2a] pb-4">
                <p className="font-mono text-xs font-semibold uppercase tracking-[0.22em] text-[#cc0000]">
                    Diagnostic report
                </p>
                <h2 className="text-xl font-semibold text-[#e3e2e2]">
                    Analysis Result
                </h2>
            </div>

            <div className="grid gap-3 sm:grid-cols-4">
                <div className="rounded-md border border-[#2a2a2a] bg-[#121414] px-3 py-3">
                    <p className="font-mono text-xs font-medium text-[#c8c6c5]">Total Tokens</p>
                    <p className="text-lg font-semibold text-[#e3e2e2]">{rows.length}</p>
                </div>
                <div className="rounded-md border border-[#2a2a2a] bg-[#121414] px-3 py-3">
                    <p className="font-mono text-xs font-medium text-[#c8c6c5]">Beneficial</p>
                    <p className="text-lg font-semibold text-green-300">{summary.beneficial}</p>
                </div>
                <div className="rounded-md border border-[#2a2a2a] bg-[#121414] px-3 py-3">
                    <p className="font-mono text-xs font-medium text-[#c8c6c5]">Neutral</p>
                    <p className="text-lg font-semibold text-[#e3e2e2]">{summary.neutral}</p>
                </div>
                <div className="rounded-md border border-[#cc000033] bg-[#cc00001a] px-3 py-3">
                    <p className="font-mono text-xs font-medium text-[#e8bdb6]">Detrimental</p>
                    <p className="text-lg font-semibold text-[#ffb4a8]">{summary.detrimental}</p>
                </div>
            </div>

            <p className="text-sm text-[#c8c6c5]">
                Special tokens like [CLS] and [SEP] are included in this prototype analysis.
            </p>

            {strongestSink ? (
                <div className="rounded-md border border-[#cc000033] bg-[#cc00001a] p-4">
                    <p className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#cc0000]">
                        Main Finding
                    </p>

                    <p className="mt-2 text-sm leading-6 text-[#e8bdb6]">
                        <span className="font-mono font-semibold text-[#ffb4a8]">
                            {strongestSink.token}
                        </span>{" "}
                        received{" "}
                        <span className="font-semibold text-[#ffb4a8]">
                            {(strongestSink.attentionReceived * 100).toFixed(2)}%
                        </span>{" "}
                        of normalized attention and was classified as{" "}
                        <span className="font-semibold text-[#ffb4a8]">
                            {strongestSink.classification}
                        </span>
                        .
                    </p>
                </div>  
            ) : null}

            {topSinks.length ? (
                <div className="rounded-md border border-[#2a2a2a] bg-[#121414] p-4">
                    <div className="mb-3">
                        <h3 className="font-mono text-sm font-semibold text-[#e3e2e2]">
                            Top Attention Sinks
                        </h3>
                        <p className="text-sm text-[#c8c6c5]">
                            Tokens receiving the highest share of normalized attention.
                        </p>
                    </div>

                    <div className="space-y-2">
                        {topSinks.map((sink) => (
                            <div
                                key={`top-${sink.index}-${sink.token}`}
                                className="flex items-center justify-between gap-3 rounded-md border border-[#1f2020] bg-[#0d0e0f] px-3 py-2"
                            >
                                <div className="min-w-0">
                                    <p className="truncate font-mono text-sm font-semibold text-[#e3e2e2]">
                                        {sink.token}
                                    </p>
                                    <p className="text-xs text-[#c8c6c5]">
                                        Index {sink.index}
                                    </p>
                                </div>

                                <div className="flex shrink-0 items-center gap-2">
                                    <span className="text-sm font-semibold text-[#ffb4a8]">
                                        {(sink.attentionReceived * 100).toFixed(2)}%
                                    </span>

                                    <span
                                        className={`inline-flex rounded-md px-1 text-xs font-medium capitalize ring-1 ring-inset
                                        ${classificationStyles[sink.classification]}`}
                                    >
                                        {sink.classification}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ) : null}

            <AttentionHeatmap tokens={result.token_list} matrix={result.att_matrix} />

            <div className="overflow-x-auto rounded-md border border-[#2a2a2a]">
                <table className="w-full text-left text-sm">
                    <thead className="bg-[#121414] font-mono text-[#c8c6c5]">
                        <tr>
                            <th className="px-3 py-2">Index</th> 
                            <th className="px-3 py-2">Token</th> 
                            <th className="px-3 py-2">Classification</th> 
                            <th className="px-3 py-2">Attention Received</th> 
                            <th className="px-3 py-2">Value Norm</th> 
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                           <tr
                            key={`${row.index}-${row.token}`}
                            className="border-t border-[#2a2a2a] text-[#e3e2e2]"
                           >
                            <td className="px-3 py-2 text-[#c8c6c5]">
                                {row.index}
                            </td>
                            <td className="px-3 py-2 font-mono">
                                {row.token}
                            </td>
                            <td className="px-3 py-2">
                                <span
                                    className={`inline-flex rounded-md px-2 py-1 text-xs font-medium capitalize ring-1 ring-inset ${classificationStyles[row.classification]}`}
                                >
                                    {row.classification}
                                </span>
                            </td>
                            <td className="px-3 py-2">
                                {row.attentionReceived.toFixed(4)}
                            </td>
                            <td className="px-3 py-2">
                                {row.valueNorm.toFixed(4)}
                            </td>
                           </tr> 
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}

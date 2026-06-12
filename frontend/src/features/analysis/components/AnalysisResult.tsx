import { AnalyzeResponse, SinkClassification, TokenAnalysisRow } from "../types/analysis";


type AnalyzeResponseProps = {
    result: AnalyzeResponse;
}

const classificationStyles: Record<SinkClassification, string> = {
    beneficial: "bg-green-50 text-green-700 ring-green-600/20",
    neutral: "bg-zinc-50 text-zinc-700 ring-zinc-600/20",
    detrimental: "bg-red-50 text-red-700 ring-red-600/20",
};

export function AnalysisResult({ result }: AnalyzeResponseProps) { 
    const rows: TokenAnalysisRow[] = result.token_list.map((token, index) => ({
        index,
        token,
        classification: result.classifications[index],
        attentionReceived: result.att_received_scores[index],
        valueNorm: result.value_norms[index],
    }));

    const strongestSink = rows.length ? rows.reduce((strongest, row) => {
        if (row.attentionReceived > strongest.attentionReceived) { 
            return row;
        }

        return strongest;
    }, rows[0]): null; 

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
        <section className="space-y-3">
            <h2 className="text-lg font-semibold text-zinc-900">
                Analysis Result
            </h2>

            <div className="grid gap-3 sm:grid-cols-4">
                <div className="rounded-md border border-zinc-200 px-3 py-2">
                    <p className="text-xs font-medium text-zinc-500">Total Tokens</p>
                    <p className="text-lg font-semibold text-zinc-900">{rows.length}</p>
                </div>
                <div className="rounded-md border border-zinc-200 px-3 py-2">
                    <p className="text-xs font-medium text-zinc-500">Beneficial</p>
                    <p className="text-lg font-semibold text-green-700">{summary.beneficial}</p>
                </div>
                <div className="rounded-md border border-zinc-200 px-3 py-2">
                    <p className="text-xs font-medium text-zinc-500">Neutral</p>
                    <p className="text-lg font-semibold text-zinc-900">{summary.neutral}</p>
                </div>
                <div className="rounded-md border border-zinc-200 px-3 py-2">
                    <p className="text-xs font-medium text-zinc-500">Detrimental</p>
                    <p className="text-lg font-semibold text-red-700">{summary.detrimental}</p>
                </div>
            </div>

            <p className="text-sm text-zinc-600">
                Special tokens like [CLS] and [SEP] are included in this prototype analysis.
            </p>
            {strongestSink ? (
                <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
                <p className="text-xs font-medium uppercase text-zinc-500">
                    Main Finding
                </p>

                <p className="mt-2 text-sm text-zinc-700">
                    <span className="font-mono font-semibold text-zinc-900">
                        {strongestSink.token}
                    </span>{" "}
                    received{" "}
                    <span className="font-semibold text-zinc-900">
                        {(strongestSink.attentionReceived * 100).toFixed(2)}%
                    </span>{" "}
                    of normalized attention and was classified as{" "}
                    <span className="font-semibold text-zinc-900">
                        {strongestSink.classification}
                    </span>
                    .
                </p>
                </div>  
            ) : null}
            <div className="overflow-x-auto rounded-md border border-zinc-200">
                <table className="w-full text-left text-sm">
                    <thead className="bg-zinc-50 text-zinc-700">
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
                            className="border-t border-zinc-200"
                           >
                            <td className="px-3 py-2 text-zinc-500">
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

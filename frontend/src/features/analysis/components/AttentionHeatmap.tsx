type AttentionHeatmapProps = {
    tokens: string[];
    matrix?: number[][];
};

function getHeatmapColor(value: number, maxValue: number) {
    const intensity = maxValue ? value / maxValue : 0;
    const lightness = 12 + intensity * 40;
    const saturation = 35 + intensity * 55;

    return `hsl(0 ${saturation}% ${lightness}%)`;
}

export function AttentionHeatmap({ tokens, matrix }: AttentionHeatmapProps) {
    const hasValidMatrix =
        Boolean(matrix?.length) &&
        matrix?.length === tokens.length &&
        matrix.every((row) => row.length === tokens.length);

    if (!hasValidMatrix) {
        return null;
    }

    const tokenCount = tokens.length;
    const cellCount = tokenCount * tokenCount;
    const isCompact = tokenCount > 32;
    const cellClassName = isCompact ? "h-4 w-4" : "h-6 w-6";
    const rowLabelClassName = isCompact ? "w-16" : "w-20";
    const headerHeightClassName = isCompact ? "h-16" : "h-18";
    const maxValue = Math.max(...matrix.flat());

    return (
        <div className="rounded-md border border-[#2a2a2a] bg-[#121414] p-4">
            <div className="mb-3">
                <h3 className="font-mono text-sm font-semibold text-[#e3e2e2]">
                    Attention Heatmap
                </h3>
                <p className="text-sm text-[#c8c6c5]">
                    Rows are source tokens. Columns are target tokens
                </p>
                <p className="mt-1 font-mono text-xs text-[#8f8b89]">
                    {tokenCount} tokens, {cellCount.toLocaleString()} attention cells
                    {isCompact ? " · compact view enabled" : ""}
                </p>
            </div>

            <div className="max-h-[640px] overflow-auto">
                <div className="min-w-max space-y-1">
                    <div className="sticky top-0 z-20 flex items-end gap-1 bg-[#121414] pb-1">
                        <div className={`sticky left-0 z-30 shrink-0 bg-[#121414] ${rowLabelClassName}`} />

                        {tokens.map((token, index) => (
                            <div
                                key={`column-${index}-${token}`}
                                className={`relative shrink-0 ${cellClassName} ${headerHeightClassName}`}
                                title={`Target token: ${token}`}
                            >
                                <span className="absolute bottom-1 left-1 origin-bottom-left -rotate-45 whitespace-nowrap font-mono text-xs text-[#c8c6c5]">
                                    {token}
                                </span>
                            </div>
                        ))}
                    </div>

                    {matrix.map((row, rowIndex) => (
                        <div key={rowIndex} className="flex items-center gap-1">
                            <div
                                className={`sticky left-0 z-10 shrink-0 truncate bg-[#121414] pr-2 text-right font-mono text-xs text-[#c8c6c5] ${rowLabelClassName}`}
                                title={`Source token: ${tokens[rowIndex]}`}
                            >
                                {tokens[rowIndex]}
                            </div>

                            {row.map((value, columnIndex) => (
                                <div
                                    key={`${rowIndex}-${columnIndex}`}
                                    title={`${tokens[rowIndex]} -> ${tokens[columnIndex]}: ${value.toFixed(4)}`}
                                    className={`shrink-0 rounded-sm border border-[#2a2a2a] ${cellClassName}`}
                                    style={{
                                        backgroundColor: getHeatmapColor(value, maxValue),
                                    }}
                                />
                            ))}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

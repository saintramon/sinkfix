import { useState } from "react";
import { AnalysisFormValues } from "../types/analysis";

type AnalysisFormProps = {
    isSubmitting: boolean;
    onSubmit: (values: AnalysisFormValues) => void;
};

const EXAMPLE_INPUTS = [
    {
        title: "Normal Sentence",
        description: "A simple baseline sentence for a quick diagnostic run.",
        text: "To get something you never had, you need to do something you never did.",
    },
    {
        title: "Repetition",
        description: "Shows how repeated words can affect attention patterns.",
        text: "Very very very small changes can create very different attention patterns.",
    },
    {
        title: "Classification-style",
        description: "Useful for observing structural-token behavior.",
        text: "Classification tasks often rely on the first token representation.",
    },
    {
        title: "Mixed Punctuation",
        description: "Includes punctuation and varied token boundaries.",
        text: "Sinkfix inspects attention patterns: tokens, punctuation, and repeated phrases can behave differently."
    }
];

export function AnalysisForm({ isSubmitting, onSubmit }: AnalysisFormProps) {
    const [modelName, setModelName] = useState("google-bert/bert-base-uncased");
    const [text, setText] = useState("");
    const [error, setError] = useState<string | null>(null);

    function handleSubmit(event: React.SubmitEvent<HTMLFormElement>) {
        event.preventDefault();

        const trimmedModelname = modelName.trim();
        const trimmedText = text.trim();

        if (!trimmedModelname) {
            setError("Model name is required");
            return;
        }

        if (!trimmedText) {
            setError("Text is required");
            return;
        }

        setError(null);

        onSubmit({
            modelName: trimmedModelname,
            text: trimmedText,
        });
    }
    
    return (
        <form
            onSubmit={handleSubmit}
            className="space-y-5 rounded-md border border-[#2a2a2a] bg-[#0d0e0f] p-5 shadow-[0_0_40px_rgba(204,0,0,0.08)]"
        >
            <div className="space-y-1">
                <h2 className="font-mono text-lg font-semibold text-[#e3e2e2]">Run diagnostics</h2>
                <p className="text-sm text-[#c8c6c5]">
                    Submit text to inspect token-level attention behavior.
                </p>
            </div>

            <div className="space-y-2">
                <label
                    htmlFor="model-name"
                    className="block font-mono text-sm font-medium text-[#e8bdb6]"
                >
                    Model Name
                </label>
                <input 
                    id="model-name"
                    type="text" 
                    name="modelName"
                    value={modelName}
                    disabled={isSubmitting}
                    onChange={(event) => setModelName(event.target.value)}
                    className="w-full rounded-md border border-[#343535] bg-[#121414] px-3 py-2 font-mono text-sm text-[#e3e2e2] outline-none transition focus:border-[#cc0000] focus:ring-2 focus:ring-[#cc000033] disabled:cursor-not-allowed disabled:opacity-60"
                />
            </div>
            <div className="space-y-3">
                <div className="space-y-1">
                    <h3 className="font-mono text-sm font-semibold text-[#e8bdb6]">
                        Try an example
                    </h3>
                    <p className="text-xs text-[#c8c6c5]">
                        Load a guided input, then edit it before running analysis.
                    </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                    {EXAMPLE_INPUTS.map((example) => (
                        <button
                            key={example.title}
                            type="button"
                            disabled={isSubmitting}
                            onClick={() => {
                                setText(example.text);
                                setError(null);
                            }}
                            className="rounded-md border border-[#343535] bg-[#121414] p-3 text-left transition hover:border-[#cc0000] hover:bg-[#1a1111] disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            <span className="block font-mono text-sm font-semibold text-[#e3e2e2]">
                                {example.title}
                            </span>
                            <span className="mt-1 block text-xs text-[#c8c6c5]">
                                {example.description}
                            </span>
                        </button>
                    ))}
                </div>
            </div>
            <div className="space-y-2">
                <label 
                    htmlFor="analysis-text"
                    className="block font-mono text-sm font-medium text-[#e8bdb6]"
                >
                    Text to analyze
                </label>
                <textarea 
                    id="analysis-text"
                    name="text" 
                    value={text}
                    disabled={isSubmitting}
                    onChange={(event) => setText(event.target.value)}
                    rows={8}
                    placeholder="Enter the text for attention sink analysis..."
                    className="w-full resize-y rounded-md border border-[#343535] bg-[#121414] px-3 py-2 text-sm text-[#e3e2e2] outline-none transition placeholder:text-[#5e3f3a] focus:border-[#cc0000] focus:ring-2 focus:ring-[#cc000033] disabled:cursor-not-allowed disabled:opacity-60"
                />
            </div>

            {error ? <p className="text-sm text-[#ffb4a8]">{error}</p> : null}

            <button
                type="submit"
                disabled={isSubmitting}
                className="rounded-md border border-[#cc0000] bg-[#cc0000] px-4 py-2 font-mono text-sm font-semibold uppercase tracking-[0.16em] text-white transition hover:bg-[#a80000] disabled:cursor-not-allowed disabled:border-[#5e3f3a] disabled:bg-[#5e3f3a]"
            >
                {isSubmitting ? "Analyzing..." : "Analyze"}
            </button>
        </form>
    );
}

import { useState } from "react";
import { AnalysisFormValues } from "../types/analysis";

type AnalysisFormProps = {
    isSubmitting: boolean;
    onSubmit: (values: AnalysisFormValues) => void;
}

export function AnalysisForm({ isSubmitting, onSubmit }: AnalysisFormProps) {
    const[modelName, setModelName] = useState("google-bert/bert-base-uncased");
    const[text, setText] = useState("");
    const[error, setError] = useState<string | null>(null);

    function handleSubmit(event: React.SubmitEvent<HTMLFormElement>) {
        event.preventDefault();

        const trimmedModelname = modelName.trim();
        const trimmedText = text.trim();

        if(!trimmedModelname) {
            setError("Model name is required");
            return;
        }

        if(!trimmedText) {
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
        <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-2">
                <label
                    htmlFor="model-name"
                    className="block text-sm font-medium text-zinc-900"
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
                    className="w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 outline-none focus:border-zinc-900"
                />
            </div>
            <div className="space-y-2">
                <label 
                    htmlFor="analysis-text"
                    className="block text-sm font-medium text-zinc-900"
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
                    className="w-full resize-y rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 outline-none focus:border-zinc-900"
                />
            </div>

            {error ? <p className="text-sm text-red-600">{error}</p> : null}

            <button
                type="submit"
                disabled={isSubmitting}
                className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
            >
                {isSubmitting ? "Analyzing..." : "Analyze"}
            </button>
        </form>
    );
}
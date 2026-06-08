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

        let trimmedModelname = modelName.trim();
        let trimmedText = text.trim();

        if(!trimmedModelname) {
            setError("Model name is required");
            return;
        }

        if(!trimmedText) {
            setError("Text is required")
        }

        setError(null);

        onSubmit({
            modelName: trimmedModelname,
            text: trimmedText,
        });
    }
    
}
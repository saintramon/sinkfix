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
}
"use client";

import { useState } from "react";
import { AnalysisForm } from "./AnalysisForm";
import { AnalysisResult } from "./AnalysisResult";
import { sendAnalyzeRequest } from "../api/analyzeRequest";
import type { AnalysisFormValues, AnalyzeResponse } from "../types/analysis";


export function AnalysisWorkspace() {
    const [ isSubmitting, setIsSubmitting ] = useState(false);
    const [ error, setError] = useState<string | null>(null);
    const [ result, setResult ] = useState<AnalyzeResponse | null>(null);

    async function handleSubmit(values: AnalysisFormValues) {
        setIsSubmitting(true);
        setError(null);

        try {
            const response = await sendAnalyzeRequest({
                model_name: values.modelName,
                text: values.text,
            });

            setResult(response);
        } catch {
            setError("Could not analyze the text. Check backend is running");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <section className="space-y-6">
            <AnalysisForm
                isSubmitting={isSubmitting}
                onSubmit={handleSubmit}
            />
 
            {error ? (
                <p className="text-sm text-red-600">{error}</p>
            ) : null}
 
            {result ? (
                <AnalysisResult result={result} /> 
            ) : null}
        </section>
    );
}
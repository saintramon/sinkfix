"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { AnalysisForm } from "./AnalysisForm";
import { sendAnalyzeRequest } from "../api/analyzeRequest";
import type { AnalysisFormValues } from "../types/analysis";

const LAST_ANALYSIS_RESULT_KEY = "sinkfix:lastAnalysisResult";

export function AnalysisWorkspace() {
    const router = useRouter();
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    async function handleSubmit(values: AnalysisFormValues) {
        setIsSubmitting(true);
        setError(null);

        try {
            const response = await sendAnalyzeRequest({
                model_name: values.modelName,
                text: values.text,
            });

            const resultForDisplay = {
                token_list: response.token_list,
                classifications: response.classifications,
                att_received_scores: response.att_received_scores,
                value_norms: response.value_norms,
                att_matrix: response.att_matrix,
            }

            sessionStorage.setItem(LAST_ANALYSIS_RESULT_KEY, JSON.stringify(resultForDisplay));
            router.push("/results");
        } catch (error){
            console.error("Analysis failed: ", error)
            setError("Analysis failed. Make sure the FastAPI backend is running and the model name is valid.");
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <section className="space-y-4">
            <AnalysisForm
                isSubmitting={isSubmitting}
                onSubmit={handleSubmit}
            />
 
            {error ? (
                <div className="rounded-md border border-[#cc0000] bg-[#cc00001a] p-4 text-sm text-[#ffb4a8]">
                    {error}
                </div>
            ) : null}

            {isSubmitting ? (
                <div className="rounded-md border border-[#2a2a2a] bg-[#0d0e0f] p-4 text-sm text-[#c8c6c5]">
                    Analyzing attention patterns...
                </div>
            ) : null}
        </section>
    );
}

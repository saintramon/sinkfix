import { AnalyzeRequest, AnalyzeResponse } from "../types/analysis";

export async function sendAnalyzeRequest(req: AnalyzeRequest): Promise<AnalyzeResponse> {
    const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type" : "application/json" },
        body: JSON.stringify(req)
    });

    const data: AnalyzeResponse = await response.json();

    return data;
};
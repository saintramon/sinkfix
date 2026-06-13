import { AnalyzeRequest, AnalyzeResponse } from "../types/analysis";

const API_BASE_URL = (
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "")
).replace(/\/$/, "");

export async function sendAnalyzeRequest(req: AnalyzeRequest): Promise<AnalyzeResponse> {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type" : "application/json" },
        body: JSON.stringify(req)
    });

    const data: AnalyzeResponse = await response.json();

    return data;
};
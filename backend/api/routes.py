from fastapi import APIRouter
from backend.api.schemas import ModelResponse, ModelTextInput
from backend.ml.utils import load_model, extract_attention
from backend.ml.sink_detector import detect_sinks
from backend.ml.classifier import classify_sink
from backend.ml.optimizer import redistribute_attention

router = APIRouter()

@router.post("/analyze")
async def analyze(input: ModelTextInput):
    model, tokenizer = load_model(input.model_name)
    att_weights, token_list = extract_attention(model, tokenizer, input.text)

    sink_mask, normalized = detect_sinks(att_weights, threshold=0.05)

    classifications = []

    for i in range(len(token_list)):
        if sink_mask[i]:
            label = classify_sink(i, normalized, att_weights, 0)
        else:
            label = "neutral"

        classifications.append(label)
        
    corrected_att_weights = redistribute_attention(att_weights, sink_mask, classifications)

    return ModelResponse(
        token_list=token_list, 
        corrected_att_scores=corrected_att_weights.tolist(), 
        classifications=classifications,
    )
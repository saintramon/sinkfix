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
    att_weights, token_list, value_norms = extract_attention(model, tokenizer, input.text)

    sink_mask, att_received_scores = detect_sinks(att_weights, threshold=0.02)

    classifications = []

    for i in range(len(token_list)):
        if sink_mask[i]:
            label = classify_sink(i, value_norms[i], att_received_scores[i], 0)
        else:
            label = "neutral"

        classifications.append(label)
        
    corrected_att_weights = redistribute_attention(att_weights, sink_mask, classifications)

    return ModelResponse(
        token_list=token_list, 
        corrected_att_scores=corrected_att_weights.tolist(), 
        classifications=classifications,
        att_received_scores = att_received_scores.tolist(),
        value_norms=value_norms.tolist()
    )
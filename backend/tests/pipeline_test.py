from backend.ml.utils import load_model, extract_attention
from backend.ml.sink_detector import detect_sinks
from backend.ml.classifier import classify_sink
from backend.ml.optimizer import redistribute_attention

model, tokenizer = load_model("google-bert/bert-base-uncased")

att_weights, token_list = extract_attention(model, tokenizer, "I want to be a Data Scientist soon! But where do I start?")

sink_mask, normalized = detect_sinks(att_weights)

classifications = []

for i in range(len(token_list)):
    if sink_mask[i]:
        label = classify_sink(i, normalized, att_weights, 0)
    else:
        label = "neutral"
    
    classifications.append(label)

corrected_att_tensor = redistribute_attention(att_weights, sink_mask, classifications)

print(corrected_att_tensor.shape)

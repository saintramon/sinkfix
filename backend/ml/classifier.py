def classify_sink(token_idx, value_norms, attention_received, layer_depth):
    if token_idx == 0 and layer_depth <= 2:
        return "beneficial"
    
    elif value_norms[token_idx] < 0.01 and attention_received[token_idx] > 0.2:
        return "detrimental"
    
    return "neutral"

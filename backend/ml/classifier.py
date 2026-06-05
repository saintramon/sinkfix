def classify_sink(token_idx, value_norm, attention_received_score, layer_depth):
    if token_idx == 0 and layer_depth <= 2:
        return "beneficial"
    
    elif attention_received_score > 0.02 and value_norm < 0.95:
        return "detrimental"
    
    return "neutral"

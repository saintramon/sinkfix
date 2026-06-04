from transformers import AutoModel, AutoTokenizer
import torch

def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(
        model_name, 
        output_attentions=True
    )

    model.eval()

    return model, tokenizer

def extract_attention(model, tokenizer, text):

    tokens = tokenizer(text, return_tensors="pt")

    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    with torch.no_grad():
        output = model(**tokens, output_attentions=True, output_hidden_states=True)
        attention_weights = output.attentions
        hidden_states = output.hidden_states

        layer_idx = 0
        layer_hidden = hidden_states[layer_idx]

        value_vectors = model.encode.layer[layer_idx].attention.self.value(layer_hidden)

        value_norms = value_norms / value_norms.mean().clamp(min=1e-9)

    return attention_weights, token_list, value_norms
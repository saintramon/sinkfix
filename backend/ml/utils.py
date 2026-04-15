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
        output = model(**tokens)
        attention_weights = output.attentions

    return attention_weights, token_list
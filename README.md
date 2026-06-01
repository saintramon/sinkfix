# SinkFix

SinkFix is a personal research project about understanding and mitigating **attention sinks** in transformer models.

The goal is not to train a new model from scratch. The goal is to study model behavior, detect when attention becomes overly concentrated on unhelpful tokens, and experiment with inference-time corrections without retraining.

## What the project is trying to do

SinkFix explores a simple idea:

- inspect attention maps from a transformer
- identify tokens that behave like sinks
- decide whether those sinks are useful or harmful
- redistribute attention away from harmful sinks at inference time

The project is about analysis first, intervention second.

## Current scope

The repository currently contains the core ML modules for:

- loading a model and tokenizer
- extracting attention weights
- detecting candidate sinks
- classifying sinks
- redistributing attention weights

The broader product direction is still in progress: a clean API, a usable interface, and a stronger evaluation story.

## What this is not

- not a full model-training pipeline
- not a replacement for proper fine-tuning
- not a claim that attention editing always improves results

## Learning goals

I am using this project to learn:

- how attention actually behaves in transformer models
- how to design a small ML research workflow
- how to build a backend around model analysis
- how to present technical work clearly for other people to inspect

## Tech stack

- PyTorch
- Hugging Face Transformers
- FastAPI for the backend direction
- Next.js for the frontend direction

## Run locally

The current code is centered on the backend ML modules.

```python
from backend.ml.utils import load_model, extract_attention

model, tokenizer = load_model("bert-base-uncased")
attention_weights, token_list = extract_attention(model, tokenizer, "The cat sat on the mat")
```

## Documentation

- [Future directions](FUTURE_DIRECTIONS.md)

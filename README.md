# sinkfix

A research tool for detecting, classifying, and correcting **attention sinks** in transformer-based language models.

## What is an attention sink?

Transformer models use attention mechanisms to decide which tokens to focus on when processing text. In practice, certain tokens absorb a disproportionately large share of attention — these are called **attention sinks**. Not all sinks are harmful: some (like `[CLS]` tokens in early layers) serve a legitimate aggregation role. Others drain attention without contributing meaningful information, which can degrade model output quality.

`sinkfix` identifies which tokens are acting as sinks, classifies whether each sink is beneficial or detrimental, and redistributes attention away from the detrimental ones — all without retraining the model.

## How it works

The pipeline runs in four stages:

```
Input text
    │
    ▼
[utils.py]         Load model + tokenizer, extract raw attention weights
    │
    ▼
[sink_detector.py] Average attention across layers and heads,
                   flag tokens above a normalized threshold
    │
    ▼
[classifier.py]    Classify each flagged token as beneficial,
                   detrimental, or neutral
    │
    ▼
[optimizer.py]     Zero out detrimental sinks on the key axis,
                   renormalize remaining weights proportionally
```

## Project structure

```
backend/
└── ml/
    ├── utils.py          # Model loading and attention extraction
    ├── sink_detector.py  # Sink detection via attention aggregation
    ├── classifier.py     # Per-token sink classification
    └── optimizer.py      # Attention redistribution
```

## Tech stack

- **PyTorch** — tensor operations and attention weight manipulation
- **Hugging Face Transformers** — model and tokenizer loading

## Status

The four core ML modules are complete. An end-to-end integration pipeline and REST API are in progress.

## Setup

```bash
pip install torch transformers
```

Run a model (BERT used as default):

```python
from backend.ml.utils import load_model, extract_attention

model, tokenizer = load_model("bert-base-uncased")
attention_weights, token_list = extract_attention(model, tokenizer, "The cat sat on the mat")
```

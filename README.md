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

The repository currently contains a working backend prototype for:

- loading a model and tokenizer
- extracting attention weights
- extracting per-token value vector norms
- detecting candidate sinks
- classifying sinks as beneficial, detrimental, or neutral
- redistributing attention weights
- exposing the pipeline through a FastAPI endpoint

The broader product direction is still in progress: a usable interface, better visualization, and a stronger evaluation story.

## Current method

SinkFix currently uses `google-bert/bert-base-uncased` as the main prototype model.

The backend pipeline is:

1. tokenize input text
2. run the model with attention and hidden-state outputs enabled
3. average attention across layers and heads
4. compute normalized attention received by each token
5. compute value vector norms for each token from the selected BERT layer
6. classify sink candidates using attention received and value norm
7. redistribute attention away from detrimental sink tokens

The current classification rule is intentionally simple:

- token index `0` is treated as a beneficial sink candidate
- high attention received with low value norm is treated as detrimental
- everything else is neutral

This is a research prototype, not a finished attention-editing method.

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

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn backend.api.main:app --reload
```

Call the analysis endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_name":"google-bert/bert-base-uncased","text":"[MASK] [MASK] [MASK] [MASK] The actual sentence contains no useful semantic content."}'
```

Run a syntax check:

```bash
python -m compileall backend
```

## API response

`POST /api/analyze` returns:

- `token_list`: model tokens
- `classifications`: one label per token
- `att_received_scores`: normalized attention received by each token
- `value_norms`: normalized value vector norm per token
- `corrected_att_scores`: redistributed attention tensor

The full corrected attention tensor is dense and large. It is kept for inspection and future visualization work.

## Current limitations

- The value-norm extraction is currently BERT-specific.
- Thresholds are experimental and need evaluation.
- The project does not yet measure output quality or task-level improvement.
- The current endpoint loads the model per request, which is acceptable for experimentation but not efficient for production.
- Autoregressive language models are not supported yet.

## Documentation

- [Future directions](FUTURE_DIRECTIONS.md)

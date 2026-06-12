# SinkFix

SinkFix is an attention-sink explainability tool for inspecting and diagnosing transformer attention behavior.

The project helps answer a practical question:

> Where is the model placing attention, and does that attention pattern look useful, structural, or suspicious?

SinkFix currently focuses on BERT-style encoder models. It analyzes attention weights, identifies tokens that receive unusually concentrated attention, classifies those tokens, and displays the result in a small full-stack application.

## What the app does

SinkFix takes a model name and input text, then returns a token-level diagnostic view:

- model tokens produced by the tokenizer
- normalized attention received by each token
- normalized value-vector norm for each token
- a classification for each token: `beneficial`, `neutral`, or `detrimental`
- summary counts for each classification
- a table that makes high-attention tokens easy to inspect

The current frontend is designed to make the backend analysis understandable, not to hide the model behavior behind a single score.

## Current method

SinkFix currently uses `google-bert/bert-base-uncased` as the main prototype model.

The backend pipeline is:

1. tokenize the input text
2. run the model with attention and hidden-state outputs enabled
3. average attention across layers and heads
4. compute normalized attention received by each token
5. compute value-vector norms from the selected BERT layer
6. classify candidate attention sinks using attention received and value norm
7. generate an experimental corrected attention tensor for inspection

The current classification rule is intentionally simple:

- token index `0`, usually `[CLS]`, is treated as a beneficial sink candidate
- high attention received with lower value norm is treated as detrimental
- everything else is neutral

Special tokens such as `[CLS]` and `[SEP]` are included in the current prototype analysis. This is intentional for now because it makes structural-token behavior visible.

## About the correction step

SinkFix includes an experimental attention redistribution step. When a token is classified as detrimental, the prototype can suppress attention to that token in the attention tensor and renormalize the remaining attention.

This is not model training, fine-tuning, or a proven improvement to BERT output quality. The corrected attention tensor is currently diagnostic evidence for future research, not a claim that the model has been fixed.

## What this is not

- not a model-training pipeline
- not a replacement for fine-tuning
- not a claim that attention editing improves downstream task quality yet
- not a production interpretability platform
- not currently designed for autoregressive language models

## Learning goals

I am using this project to learn:

- how attention behaves inside transformer models
- how to inspect attention tensors and hidden states
- how to design a small ML research workflow
- how to build a backend around model analysis
- how to present ML behavior clearly in a frontend
- how to communicate limitations without overclaiming

## Tech stack

- PyTorch
- Hugging Face Transformers
- FastAPI
- Next.js
- TypeScript
- Tailwind CSS

## Run locally

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn backend.api.main:app --reload
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend at:

```text
http://localhost:3000
```

The frontend expects the backend at:

```text
http://localhost:8000
```

## API usage

Call the analysis endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_name":"google-bert/bert-base-uncased","text":"The wheels on the bus go round and round."}'
```

`POST /api/analyze` returns:

- `token_list`: model tokens
- `classifications`: one label per token
- `att_received_scores`: normalized attention received by each token
- `value_norms`: normalized value-vector norm per token
- `corrected_att_scores`: experimental redistributed attention tensor

The full corrected attention tensor is dense and large. The frontend currently focuses on the smaller token-level diagnostic view.

## Checks

Backend syntax check:

```bash
python -m compileall backend
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## Current limitations

- The value-norm extraction is BERT-specific.
- Thresholds are experimental and need evaluation.
- Special tokens like `[CLS]` and `[SEP]` can dominate attention.
- The correction step is not fed back into the model to measure output improvement.
- The endpoint currently loads the model per request, which is acceptable for experimentation but inefficient for production.
- Autoregressive language models are not supported yet.

## Documentation

- [Future directions](FUTURE_DIRECTIONS.md)

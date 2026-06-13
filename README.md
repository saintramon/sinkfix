# SinkFix

SinkFix is a deployed full-stack app for inspecting attention sinks in BERT-style transformer models.

The project is best described as an attention interpretability tool. It helps answer:

> Which tokens receive unusually concentrated attention, and do those tokens look structural, useful, or suspicious?

SinkFix takes a Hugging Face model name and input text, runs the model with attention and hidden-state outputs enabled, and returns a token-level diagnostic report.

## Live App

SinkFix is deployed at:

- https://www.sinkfix.xyz
- https://sinkfix.xyz

## Current Status

SinkFix currently works as a deployed attention diagnostics app:

- Backend API built with FastAPI
- Frontend built with Next.js, React, TypeScript, and Tailwind CSS
- Default model input set to `google-bert/bert-base-uncased`
- Analysis designed around BERT-style encoder internals
- Results shown as a token-level diagnostic table in the frontend

The app focuses on internal attention behavior. It does not claim to fully explain why a model produced a specific final prediction.

## What The App Shows

For each input, SinkFix returns:

- model tokens from the tokenizer
- normalized attention received by each token
- normalized value-vector norm for each token
- a token classification: `beneficial`, `neutral`, or `detrimental`
- summary counts by classification
- the strongest attention receiver
- the top attention sinks
- a full table of token-level diagnostics

Special tokens such as `[CLS]` and `[SEP]` are included in the report. That is intentional in the current version because structural-token behavior is part of what the project is inspecting.

## Backend Method

The current backend pipeline is:

1. Load the requested model and tokenizer with Hugging Face Transformers.
2. Tokenize the input text.
3. Run the model with attention and hidden-state outputs enabled.
4. Average attention across layers and heads.
5. Compute normalized attention received by each token.
6. Compute normalized value-vector norms from BERT layer `0`.
7. Detect tokens above the attention threshold.
8. Classify detected sink candidates.

The classification rule is intentionally simple:

- token index `0`, usually `[CLS]`, is classified as `beneficial` when evaluated at early layer depth
- high attention received with lower value norm is classified as `detrimental`
- everything else is classified as `neutral`

The frontend displays the token-level diagnostics as summary cards, top sinks, and a full results table.

## What This Is Not

- not a training or fine-tuning pipeline
- not a model repair system
- not a claim that attention diagnostics fully explain model decisions
- not an ML monitoring system
- not currently designed for autoregressive language models

## Tech Stack

Backend:

- Python
- FastAPI
- Hugging Face Transformers
- PyTorch

Frontend:

- Next.js
- React
- TypeScript
- Tailwind CSS

## Project Structure

```text
backend/
  api/
    main.py       FastAPI app and CORS setup
    routes.py     analysis endpoint
    schemas.py    request and response models
  ml/
    utils.py          model loading and attention extraction
    sink_detector.py  attention sink detection
    classifier.py     sink classification rule

frontend/
  app/                 Next.js routes
  src/features/analysis/
    api/               frontend API request helper
    components/        analysis form and results UI
    types/             TypeScript response types
```

## Run Locally

Install backend dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch is not already installed in your environment, install the build appropriate for your machine from the official PyTorch instructions.

Start the backend:

```bash
uvicorn backend.api.main:app --reload
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

Start the frontend:

```bash
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

Allowed frontend origins can be configured with:

```bash
FRONTEND_ORIGINS=https://www.sinkfix.xyz,https://sinkfix.xyz,http://localhost:3000
```

## API Usage

Call the analysis endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_name":"google-bert/bert-base-uncased","text":"The wheels on the bus go round and round."}'
```

Request body:

```json
{
  "model_name": "google-bert/bert-base-uncased",
  "text": "The wheels on the bus go round and round."
}
```

Response fields:

- `token_list`: model tokens
- `classifications`: one label per token
- `att_received_scores`: normalized attention received by each token
- `value_norms`: normalized value-vector norm per token

## Frontend Flow

The input page submits a model name and text to the backend. On success, the frontend stores the latest response in `sessionStorage` and navigates to `/results`.

The results page reads that stored response and renders:

- total token count
- classification counts
- strongest attention receiver
- top five attention sinks
- full token table

Refreshing or opening `/results` without a stored response shows an empty-state message.

## Checks

Backend syntax check:

```bash
python -m compileall backend
```

Frontend lint:

```bash
cd frontend
npm run lint
```

Frontend production build:

```bash
cd frontend
npm run build
```

## Current Limitations

- The value-vector extraction assumes BERT internals at `model.encoder.layer[...]`.
- The model is loaded on every request, which is slow and inefficient.
- Classification thresholds are heuristic and may need validation for broader model coverage.
- The frontend only keeps the latest result in browser `sessionStorage`.
- Autoregressive language models are not supported.
- The project currently inspects attention behavior, not full causal explanations of model predictions.

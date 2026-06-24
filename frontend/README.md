# SinkFix Frontend

This is the Next.js frontend for SinkFix, an attention-sink explainability tool for transformer models.

The frontend lets a user submit text and inspect the backend's token-level analysis in a readable diagnostic view.

## Current UI

The interface currently supports:

- entering a Hugging Face model name
- entering text to analyze
- submitting the input to the FastAPI backend
- showing loading and error states
- displaying summary counts for token classifications
- displaying an averaged attention heatmap
- displaying a token table with attention received, value norm, and classification
- exporting the displayed result as JSON or CSV

## Backend requirement

Start the backend from the repository root before using the frontend:

```bash
uvicorn backend.api.main:app --reload
```

The frontend expects the backend at:

```text
http://localhost:8000
```

## Run locally

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

## Checks

Run linting:

```bash
npm run lint
```

Run a production build:

```bash
npm run build
```

## Product direction

The frontend should make transformer attention behavior easier to understand. It should not imply that SinkFix retrains or permanently fixes the model.

Near-term improvements should focus on:

- example inputs for guided demos
- curated compatible BERT model selection
- strongest-sink explanations
- clearer empty/loading/error states
- screenshots for portfolio review
- readable handling for long token sequences

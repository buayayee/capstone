# Capstone — Deferment Fraud Checker (Docker-first)

This project can run fully in Docker (no local Python/venv required).

## Prerequisites

- Docker Desktop running
- Files present in:
  - `directives/` (instruction `.docx` / `.pdf`)
  - `submissions/` (supporting docs to check)

## 1) Build the image

```bash
docker compose build
```

## 2) Run checker on all submissions

Writes results to `output/results.csv`.

```bash
docker compose run --rm checker
```

## 3) Run checker on one document

```bash
docker compose run --rm checker python check.py \
  --instructions "directives/Instructions (Directive).docx" \
  --document submissions/my_doc.pdf
```

## 4) Start API service

```bash
docker compose up api
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Detached mode:

```bash
docker compose up -d api
```

## One-time API key setup

Put your keys in the project-root `.env` file once. Docker Compose auto-loads it, so you do not need to `export` them every time.

Starter files included:

- `.env` for your local machine
- `.env.example` as a shareable template

Edit `.env` and fill in the values:

```bash
nano .env
```

Expected keys:

```bash
GROQ_API_KEY="your-key"
GROQ_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
AWS_DEFAULT_REGION="us-east-1"
DATAGOV_API_KEY="..."
```

Then just run Docker normally.

## Verify OCR dependency inside Docker

`onnxruntime` is required for the OCR backend. Quick check:

```bash
docker compose run --rm checker python -c "import onnxruntime; print(onnxruntime.__version__)"
```

## Troubleshooting

- If you changed `requirements.txt`, rebuild:

```bash
docker compose build --no-cache
```

- If output looks stale, remove and rerun checker:

```bash
rm -f output/results.csv
docker compose run --rm checker
```
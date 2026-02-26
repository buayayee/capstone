# ─────────────────────────────────────────────────────────────────────────────
# Capstone Project — Deferment Fraud Checker
# Base image: slim Debian Python so apt-get is available for system tools
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# tesseract-ocr : OCR engine for scanned PDFs
# poppler-utils : pdf2image backend (pdftoppm)
# libgl1        : required by some Pillow builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Volumes (mounted at runtime — no rebuild needed when you add documents) ───
# directives/ : your instructions .docx or .pdf
# submissions/ : PDFs to check
# output/      : results CSV written here
VOLUME ["/app/directives", "/app/submissions", "/app/output"]

# ── Default command: run the CLI checker ─────────────────────────────────────
# Override with `docker run ... python -m uvicorn api.main:app` for the API
CMD ["python", "check.py", "--help"]

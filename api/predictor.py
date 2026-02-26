"""
predictor.py
------------
Loads the fine-tuned DistilBERT model once at startup and exposes
a predict() function for inference. Keeps the model in memory
to avoid reloading on every request (critical for 100k+ calls/day).
"""

import re
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_DIR = Path(__file__).parent.parent / "models" / "distilbert-fraud-classifier"
MAX_TOKEN_LENGTH = 512

# Threshold above which a document is considered fraudulent
FRAUD_THRESHOLD = 0.5
# Above this fraud score → REJECT; between FRAUD_THRESHOLD and REVIEW_THRESHOLD → FLAG
REVIEW_THRESHOLD = 0.75

# ── Load model once at module import ─────────────────────────────────────────
_classifier = None


def load_model():
    global _classifier
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}. "
            "Please run training/train.py first."
        )
    print(f"[Predictor] Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    _classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        top_k=None,  # return scores for ALL labels
        device=-1,   # CPU; change to 0 if GPU available
    )
    print("[Predictor] Model loaded successfully.")


def _detect_flags(text: str) -> list[str]:
    """
    Rule-based heuristics to flag specific suspicious patterns.
    These supplement the ML score with explainable reasons.
    """
    flags = []
    text_lower = text.lower()

    # Check for suspiciously future/old dates
    year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    for year in year_matches:
        yr = int(year)
        if yr < 2000 or yr > 2030:
            flags.append(f"Suspicious year detected: {year}")

    # Check for common copy-paste artifacts
    if text.count("  ") > 20:
        flags.append("Excessive whitespace detected (possible copy-paste manipulation)")

    # Check for known misspellings of real institutions
    known_misspellings = [
        "singaporr", "polyechnic", "universiti of", "technologiy",
    ]
    for mis in known_misspellings:
        if mis in text_lower:
            flags.append(f"Possible institution name misspelling: '{mis}'")

    # Check if document is suspiciously short (too little content)
    word_count = len(text.split())
    if word_count < 30:
        flags.append(f"Document too short ({word_count} words) — may be incomplete")

    return flags


def predict(document_text: str, application_id: str = "") -> dict:
    """
    Run fraud classification on the provided document text.

    Returns a dict matching FraudCheckResponse schema.
    """
    if _classifier is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Run inference
    results = _classifier(document_text)[0]  # list of {label, score} dicts
    score_map = {r["label"]: r["score"] for r in results}

    fraud_score = score_map.get("fraudulent", 0.0)
    genuine_score = score_map.get("genuine", 1.0)
    predicted_label = "fraudulent" if fraud_score >= FRAUD_THRESHOLD else "genuine"

    # Determine recommendation
    if fraud_score >= REVIEW_THRESHOLD:
        recommendation = "REJECT"
    elif fraud_score >= FRAUD_THRESHOLD:
        recommendation = "FLAG_FOR_REVIEW"
    else:
        recommendation = "APPROVE"

    # Rule-based flags for explainability
    flags = _detect_flags(document_text)

    return {
        "application_id": application_id,
        "label": predicted_label,
        "fraud_score": round(fraud_score, 4),
        "genuine_score": round(genuine_score, 4),
        "recommendation": recommendation,
        "flags": flags,
    }

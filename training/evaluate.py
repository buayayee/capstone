"""
evaluate.py
-----------
Loads the fine-tuned model and runs a full evaluation on the test set.
Outputs: accuracy, F1, precision, recall, and a confusion matrix.

Usage:
    python evaluate.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

sys.path.append(str(Path(__file__).parent))
import config


def evaluate():
    print(f"[1/4] Loading fine-tuned model from: {config.MODEL_OUTPUT_DIR}")
    if not config.MODEL_OUTPUT_DIR.exists():
        print("[ERROR] Model not found. Run train.py first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(config.MODEL_OUTPUT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(config.MODEL_OUTPUT_DIR))

    # Build inference pipeline (runs on CPU by default)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=config.MAX_TOKEN_LENGTH,
        device=-1,  # -1 = CPU; change to 0 for GPU
    )

    print(f"[2/4] Loading test data from: {config.DATA_CSV}")
    df = pd.read_csv(config.DATA_CSV).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    _, test_texts, _, test_labels = train_test_split(
        texts, labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    print(f"[3/4] Running inference on {len(test_texts)} test samples...")
    results = classifier(test_texts, batch_size=32)
    label_map = {"genuine": 0, "fraudulent": 1}
    predictions = [label_map[r["label"]] for r in results]

    print(f"\n[4/4] Evaluation Results")
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(test_labels, predictions):.4f}")
    print(f"F1 Score  : {f1_score(test_labels, predictions, average='binary'):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, predictions, target_names=["Genuine", "Fraudulent"]))

    # ── Confusion Matrix Plot ──────────────────────────────────────────────
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Fraudulent"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix – Deferment Fraud Classifier")
    plt.tight_layout()
    output_img = config.MODEL_OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(output_img)
    print(f"\nConfusion matrix saved to: {output_img}")
    plt.show()


if __name__ == "__main__":
    evaluate()

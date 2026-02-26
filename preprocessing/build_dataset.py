"""
build_dataset.py
----------------
Scans data/genuine/ and data/fraudulent/ folders,
extracts text from each document, and produces a
labeled CSV at data/labeled/dataset.csv.

CSV format:
    text, label
    "...extracted text...", 0   <- genuine
    "...extracted text...", 1   <- fraudulent

Usage:
    python build_dataset.py
"""

import csv
import os
import sys
from pathlib import Path

# Allow imports from parent directory
sys.path.append(str(Path(__file__).parent))
from extract_text import extract_text

GENUINE_DIR = Path(__file__).parent.parent / "data" / "genuine"
FRAUDULENT_DIR = Path(__file__).parent.parent / "data" / "fraudulent"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "labeled" / "dataset.csv"

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def collect_documents(folder: Path, label: int) -> list[dict]:
    """Collect all supported documents from a folder and assign a label."""
    records = []
    for file in folder.iterdir():
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            print(f"  Processing: {file.name} (label={label})")
            try:
                text = extract_text(str(file))
                if text:
                    records.append({"text": text, "label": label})
                else:
                    print(f"  [WARN] Empty text extracted from {file.name}, skipping.")
            except Exception as e:
                print(f"  [ERROR] Failed to process {file.name}: {e}")
    return records


def build_dataset():
    print("[1/3] Collecting genuine documents...")
    genuine_records = collect_documents(GENUINE_DIR, label=0)
    print(f"      Found {len(genuine_records)} genuine documents.\n")

    print("[2/3] Collecting fraudulent documents...")
    fraudulent_records = collect_documents(FRAUDULENT_DIR, label=1)
    print(f"      Found {len(fraudulent_records)} fraudulent documents.\n")

    all_records = genuine_records + fraudulent_records

    if not all_records:
        print("[ERROR] No documents found. Place PDFs/images in data/genuine/ and data/fraudulent/")
        return

    print(f"[3/3] Writing dataset to {OUTPUT_CSV}...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nDone! Dataset saved: {OUTPUT_CSV}")
    print(f"  Total samples : {len(all_records)}")
    print(f"  Genuine (0)   : {len(genuine_records)}")
    print(f"  Fraudulent (1): {len(fraudulent_records)}")


if __name__ == "__main__":
    build_dataset()

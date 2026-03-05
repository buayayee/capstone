"""
prepare_training_data.py
------------------------
Data preparation pipeline for DistilBERT training.

IMPORTANT — What belongs in each folder:
  data/genuine/     <- Documents KNOWN to be genuine/legitimate.
                       These are pre-verified, correctly-issued supporting docs.
  data/fraudulent/  <- Documents KNOWN to be fraudulent/fake.
                       These are tampered, forged, or fabricated docs.

  !! submissions/ documents do NOT go here directly. !!
     Submissions are UNKNOWN until checked. Only move a submission into
     data/genuine/ or data/fraudulent/ AFTER you have confirmed its true label.

Steps:
  1. Extract text from data/genuine/ (label=0) and data/fraudulent/ (label=1).
  2. Write data/labeled/dataset.csv with (text, label) columns.
  3. Augment: synthetically generate more fraudulent samples from genuine ones
     to balance the classes. Writes data/labeled/dataset_augmented.csv.

After this script finishes, run:
    python training/train.py

Usage:
    python prepare_training_data.py
    python prepare_training_data.py --multiplier 5
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import string
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
GENUINE_DIR = BASE_DIR / "data" / "genuine"
FRAUDULENT_DIR = BASE_DIR / "data" / "fraudulent"
LABELED_DIR = BASE_DIR / "data" / "labeled"
DATASET_CSV = LABELED_DIR / "dataset.csv"
AUGMENTED_CSV = LABELED_DIR / "dataset_augmented.csv"

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".docx"}

sys.path.insert(0, str(BASE_DIR / "preprocessing"))
from extract_text import extract_text


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Build dataset CSV from labeled folders
# ══════════════════════════════════════════════════════════════════════════════

def collect_documents(folder: Path, label: int) -> list[dict]:
    records = []
    files = [f for f in sorted(folder.iterdir())
             if f.name != ".gitkeep" and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        return records
    for file in files:
        print(f"  Extracting text: {file.name} (label={label})")
        try:
            text = extract_text(str(file))
            if text.strip():
                records.append({"text": text, "label": label})
            else:
                print(f"  [WARN] Empty text from {file.name}, skipping.")
        except Exception as e:
            print(f"  [ERROR] {file.name}: {e}")
    return records


def build_dataset(dataset_csv: Path) -> tuple[int, int]:
    print("[1/2] Collecting labeled documents...")
    genuine_records = collect_documents(GENUINE_DIR, label=0)
    fraudulent_records = collect_documents(FRAUDULENT_DIR, label=1)
    print(f"      Genuine (0): {len(genuine_records)} | Fraudulent (1): {len(fraudulent_records)}")

    if not genuine_records and not fraudulent_records:
        raise RuntimeError(
            "\nNo training documents found.\n"
            "  Place KNOWN genuine docs in:     data/genuine/\n"
            "  Place KNOWN fraudulent docs in:  data/fraudulent/\n"
            "  Do NOT put submission/ docs here until you have verified their true label."
        )

    all_records = genuine_records + fraudulent_records
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    with open(dataset_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"  Dataset CSV saved: {dataset_csv}")
    return len(genuine_records), len(fraudulent_records)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Augment with synthetic fraudulent samples
# ══════════════════════════════════════════════════════════════════════════════

_FAKE_INSTITUTIONS = [
    "Singapore Polytechnic of Technology",
    "National University of Singaporr",
    "Nanyang Technological Universiti",
    "Republic Polyechnic",
    "Temasek Polytechnics",
]
_REAL_INSTITUTIONS = [
    "National University of Singapore", "NUS",
    "Nanyang Technological University", "NTU",
    "Singapore Management University", "SMU",
    "Singapore Polytechnic", "Ngee Ann Polytechnic",
    "Temasek Polytechnic", "Republic Polytechnic",
]
_DATE_PATTERN = re.compile(r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b")


def _mutate_dates(text: str) -> str:
    def replace_date(m):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year += random.choice([-3, -2, -1, 1, 2, 3])
        day = max(1, min(day + random.randint(-5, 5), 28))
        month = max(1, min(month + random.randint(-2, 2), 12))
        return f"{day:02d}/{month:02d}/{year}"
    return _DATE_PATTERN.sub(replace_date, text)


def _mutate_institution(text: str) -> str:
    fake = random.choice(_FAKE_INSTITUTIONS)
    for inst in _REAL_INSTITUTIONS:
        if inst in text:
            return text.replace(inst, fake, 1)
    return text


def _corrupt_characters(text: str, rate: float = 0.01) -> str:
    chars = list(text)
    n = max(1, int(len(chars) * rate))
    for i in random.sample(range(len(chars)), min(n, len(chars))):
        if chars[i].isalpha():
            chars[i] = random.choice(string.ascii_letters)
    return "".join(chars)


def _augment_sample(text: str) -> str:
    mutations = [_mutate_dates, _mutate_institution, _corrupt_characters]
    for fn in random.sample(mutations, k=random.randint(1, len(mutations))):
        text = fn(text)
    return text


def augment_dataset(dataset_csv: Path, augmented_csv: Path, multiplier: int) -> None:
    rows: list[dict] = []
    genuine_rows: list[dict] = []

    with open(dataset_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if int(row["label"]) == 0:
                genuine_rows.append(row)

    synthetic = [
        {"text": _augment_sample(row["text"]), "label": 1}
        for row in genuine_rows
        for _ in range(multiplier)
    ]

    all_rows = rows + synthetic
    random.shuffle(all_rows)

    with open(augmented_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(all_rows)

    original_fraud = sum(1 for r in rows if int(r["label"]) == 1)
    print(f"  Original: {len(rows)} ({len(genuine_rows)} genuine, {original_fraud} fraudulent)")
    print(f"  New synthetic fraudulent: {len(synthetic)}")
    print(f"  Total samples: {len(all_rows)}")
    print(f"  Augmented CSV saved: {augmented_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare labeled training data for DistilBERT.")
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Synthetic fraudulent samples per genuine doc (default: 5). "
                             "Increase if you have few fraudulent examples.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Capstone -- Training Data Preparation")
    print("=" * 60)
    print(f"\n  Genuine docs folder  : {GENUINE_DIR}")
    print(f"  Fraudulent docs folder: {FRAUDULENT_DIR}")
    print()

    # Step 1
    genuine_count, fraudulent_count = build_dataset(DATASET_CSV)

    # Step 2
    print(f"\n[2/2] Augmenting dataset (multiplier={args.multiplier})...")
    augment_dataset(DATASET_CSV, AUGMENTED_CSV, multiplier=args.multiplier)

    print("\n" + "=" * 60)
    print("  Data preparation complete!")
    print(f"  Raw CSV      : {DATASET_CSV}")
    print(f"  Augmented CSV: {AUGMENTED_CSV}")
    print("=" * 60)
    print("\n  Next step -- train the model:")
    print("    C:\\cap_venv\\Scripts\\python.exe training/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

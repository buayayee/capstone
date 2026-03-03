"""
prepare_training_data.py
------------------------
One-shot data preparation pipeline for DistilBERT training.

Steps:
  1. Seed genuine training data - copies all supporting docs from
     submissions/ case folders into data/genuine/ (with unique names).
  2. Build dataset CSV - extracts text from data/genuine/ and
     data/fraudulent/, writes data/labeled/dataset.csv.
  3. Augment dataset - synthetically generates fraudulent samples
     from the genuine ones, writes data/labeled/dataset_augmented.csv.

After this script finishes, run:
    python training/train.py

Usage:
    python prepare_training_data.py
    python prepare_training_data.py --submissions submissions --multiplier 5 --skip-seed
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import string
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SUBMISSIONS_DIR = BASE_DIR / "submissions"
GENUINE_DIR = BASE_DIR / "data" / "genuine"
FRAUDULENT_DIR = BASE_DIR / "data" / "fraudulent"
LABELED_DIR = BASE_DIR / "data" / "labeled"
DATASET_CSV = LABELED_DIR / "dataset.csv"
AUGMENTED_CSV = LABELED_DIR / "dataset_augmented.csv"

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".docx"}

sys.path.insert(0, str(BASE_DIR / "preprocessing"))
from extract_text import extract_text


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Seed genuine data from submissions/
# ══════════════════════════════════════════════════════════════════════════════

def seed_genuine_data(submissions_dir: Path, genuine_dir: Path) -> int:
    """
    Copy all supported document files from submissions/<case>/ folders
    into data/genuine/ with unique names (case1_ChildBirth.pdf etc.).

    Skips files already present (safe to re-run).
    Returns the count of newly copied files.
    """
    genuine_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    case_dirs = sorted(
        [d for d in submissions_dir.iterdir() if d.is_dir() and d.name != ".gitkeep"]
    )
    if not case_dirs:
        print("  [WARN] No case folders found under submissions/. Nothing to seed.")
        return 0

    for case_dir in case_dirs:
        for doc in sorted(case_dir.iterdir()):
            if doc.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            dest_name = f"{case_dir.name}_{doc.name}"
            dest = genuine_dir / dest_name
            if dest.exists():
                skipped += 1
                continue
            shutil.copy2(doc, dest)
            print(f"  Copied: {case_dir.name}/{doc.name} → data/genuine/{dest_name}")
            copied += 1

    print(f"  Seed complete — copied: {copied}, already present (skipped): {skipped}")
    return copied


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Build dataset CSV
# ══════════════════════════════════════════════════════════════════════════════

def collect_documents(folder: Path, label: int) -> list[dict]:
    records = []
    for file in sorted(folder.iterdir()):
        if file.name == ".gitkeep" or file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
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


def build_dataset(dataset_csv: Path) -> int:
    print("[2/3] Collecting genuine documents...")
    genuine_records = collect_documents(GENUINE_DIR, label=0)
    print(f"      {len(genuine_records)} genuine documents.\n")

    print("      Collecting fraudulent documents (manual, if any)...")
    fraudulent_records = collect_documents(FRAUDULENT_DIR, label=1)
    print(f"      {len(fraudulent_records)} fraudulent documents.\n")

    all_records = genuine_records + fraudulent_records
    if not all_records:
        raise RuntimeError(
            "No documents found. Ensure submissions/ has PDFs or place docs in data/genuine/."
        )

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    with open(dataset_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"  Dataset CSV saved: {dataset_csv}")
    print(f"  Total: {len(all_records)} | Genuine: {len(genuine_records)} | Fraudulent: {len(fraudulent_records)}")
    return len(all_records)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Augment dataset with synthetic fraudulent samples
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
    parser = argparse.ArgumentParser(description="Prepare training data for DistilBERT fine-tuning.")
    parser.add_argument("--submissions", default="submissions", help="Path to submissions folder")
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Synthetic fraudulent samples per genuine doc (default: 5)")
    parser.add_argument("--skip-seed", action="store_true",
                        help="Skip copying from submissions/ (if data/genuine/ is already populated)")
    args = parser.parse_args()

    submissions_dir = Path(args.submissions)

    print("=" * 60)
    print("  Capstone — Training Data Preparation")
    print("=" * 60)

    # ── Step 1 ─────────────────────────────────────────────────────────────
    if args.skip_seed:
        print("\n[1/3] Seeding skipped (--skip-seed).")
    else:
        print(f"\n[1/3] Seeding genuine data from: {submissions_dir}")
        seed_genuine_data(submissions_dir, GENUINE_DIR)

    # ── Step 2 ─────────────────────────────────────────────────────────────
    print(f"\n[2/3] Building dataset CSV...")
    total = build_dataset(DATASET_CSV)

    # ── Step 3 ─────────────────────────────────────────────────────────────
    print(f"\n[3/3] Augmenting dataset (multiplier={args.multiplier})...")
    augment_dataset(DATASET_CSV, AUGMENTED_CSV, multiplier=args.multiplier)

    # ── Done ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Data preparation complete!")
    print(f"  Genuine source docs  : {GENUINE_DIR}")
    print(f"  Raw dataset CSV      : {DATASET_CSV}")
    print(f"  Augmented CSV        : {AUGMENTED_CSV}")
    print("=" * 60)
    print("\n  Next step — train the model:")
    print("    python training/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

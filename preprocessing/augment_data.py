"""
augment_data.py
---------------
Synthetically generates additional fraudulent samples by
mutating genuine documents. This helps balance the dataset
when you have far fewer fraudulent examples than genuine ones.

Mutation strategies:
  1. Date manipulation   - alter enrollment/validity dates
  2. Name swapping       - swap names with placeholders
  3. Institution spoofing- replace institution name with a fake one
  4. Random character corruption - simulate OCR artifacts on fake docs

Usage:
    python augment_data.py --input ../data/labeled/dataset.csv --output ../data/labeled/dataset_augmented.csv --multiplier 3
"""

import argparse
import csv
import random
import re
import string
from pathlib import Path

# Fake institutions to substitute in spoofed docs
FAKE_INSTITUTIONS = [
    "Singapore Polytechnic of Technology",  # plausible but fake
    "National University of Singaporr",     # typo-based fake
    "Nanyang Technological Universiti",
    "Republic Polyechnic",
    "Temasek Polytechnics",
]

# Common date patterns to mutate
DATE_PATTERN = re.compile(r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b")


def mutate_dates(text: str) -> str:
    """Shift dates forward/backward randomly to simulate date manipulation."""
    def replace_date(match):
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        # Randomly shift year by 1-3 years
        year += random.choice([-3, -2, -1, 1, 2, 3])
        # Clamp day and month to valid ranges
        day = max(1, min(day + random.randint(-5, 5), 28))
        month = max(1, min(month + random.randint(-2, 2), 12))
        return f"{day:02d}/{month:02d}/{year}"
    return DATE_PATTERN.sub(replace_date, text)


def mutate_institution(text: str) -> str:
    """Replace known real institution names with a fake one."""
    fake = random.choice(FAKE_INSTITUTIONS)
    real_institutions = [
        "National University of Singapore", "NUS",
        "Nanyang Technological University", "NTU",
        "Singapore Management University", "SMU",
        "Singapore Polytechnic", "Ngee Ann Polytechnic",
        "Temasek Polytechnic", "Republic Polytechnic",
    ]
    for inst in real_institutions:
        if inst in text:
            return text.replace(inst, fake, 1)
    return text


def corrupt_characters(text: str, corruption_rate: float = 0.01) -> str:
    """Randomly corrupt a small percentage of characters to simulate document tampering."""
    chars = list(text)
    num_to_corrupt = max(1, int(len(chars) * corruption_rate))
    indices = random.sample(range(len(chars)), min(num_to_corrupt, len(chars)))
    for i in indices:
        if chars[i].isalpha():
            chars[i] = random.choice(string.ascii_letters)
    return "".join(chars)


def augment_sample(text: str) -> str:
    """Apply a random combination of mutations to generate a fraudulent-looking document."""
    mutations = [mutate_dates, mutate_institution, corrupt_characters]
    # Apply 1-3 random mutations
    selected = random.sample(mutations, k=random.randint(1, len(mutations)))
    for mutation in selected:
        text = mutation(text)
    return text


def augment_dataset(input_csv: str, output_csv: str, multiplier: int = 3):
    """
    Read genuine samples (label=0) from input_csv, augment them as fraudulent (label=1),
    and write everything to output_csv.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    rows = []
    genuine_rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if int(row["label"]) == 0:
                genuine_rows.append(row)

    augmented = []
    for row in genuine_rows:
        for _ in range(multiplier):
            augmented_text = augment_sample(row["text"])
            augmented.append({"text": augmented_text, "label": 1})

    all_rows = rows + augmented
    random.shuffle(all_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(all_rows)

    original_fraud = sum(1 for r in rows if int(r["label"]) == 1)
    print(f"Augmentation complete!")
    print(f"  Original samples     : {len(rows)} ({len(genuine_rows)} genuine, {original_fraud} fraudulent)")
    print(f"  Augmented fraudulent : {len(augmented)}")
    print(f"  Total output samples : {len(all_rows)}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset with synthetic fraudulent samples.")
    parser.add_argument("--input", default="../data/labeled/dataset.csv")
    parser.add_argument("--output", default="../data/labeled/dataset_augmented.csv")
    parser.add_argument("--multiplier", type=int, default=3, help="How many fake samples to generate per genuine doc")
    args = parser.parse_args()

    augment_dataset(args.input, args.output, args.multiplier)

"""
check.py
--------
Main entry point. Given:
  1. An instructions PDF (your directive/rulebook)
  2. One or more submission PDFs

Extracts text from each and returns APPROVE / REJECT with reasons.

Usage:
  Single document:
    python check.py --instructions directives/instructions.pdf --document submissions/doc1.pdf

  Entire folder of documents:
    python check.py --instructions directives/instructions.pdf --folder submissions/

  Save results to CSV:
    python check.py --instructions directives/instructions.pdf --folder submissions/ --output results.csv
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow imports from preprocessing/
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))
from extract_text import extract_text
from rule_parser import FRAUD_SIGNAL_PATTERNS, InstructionRules, parse_instructions


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    filename: str
    verdict: str                      # "APPROVE" or "REJECT"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extracted_text_preview: str = ""  # first 300 chars for reference


# ---------------------------------------------------------------------------
# Core checking logic
# ---------------------------------------------------------------------------

def _check_required_keywords(text: str, rules: InstructionRules) -> list[str]:
    """
    Universal checks every deferment document must pass.
    - 'name'  : looks for the word 'name' OR a person-name pattern (Mr/Ms/Dr + word)
    - 'date'  : looks for the word 'date' OR an actual date format (e.g. 5 Dec 2024)
    Returns list of failed checks.
    """
    text_lower = text.lower()
    missing = []

    # Check for 'name' (literal) OR salutation pattern suggesting a person is mentioned
    name_present = (
        "name" in text_lower
        or bool(re.search(r"\b(mr|ms|mrs|dr|mdm|miss)\.?\s+[a-z]+", text_lower))
        or bool(re.search(r"\b(i/c|nric|s\d{7}[a-z])\b", text_lower, re.IGNORECASE))
    )
    if not name_present:
        missing.append("applicant name or identity")

    # Check for 'date' (literal) OR a recognisable date pattern
    date_present = (
        "date" in text_lower
        or bool(re.search(
            r"\b(\d{1,2}[\s\-/]\w+[\s\-/]\d{2,4}"   # 4 Dec 2024 or 04-Dec-24
            r"|\d{1,2}/\d{1,2}/\d{2,4}"              # 04/12/2024
            r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b"  # December 2024
            r")",
            text_lower
        ))
    )
    if not date_present:
        missing.append("issue date")

    return missing


def _check_domain_category(text: str, rules: InstructionRules) -> tuple[str | None, list[str]]:
    """
    Check whether the document matches at least one deferment category.
    Returns (matched_category_name, []) on success, or (None, [warning]) on failure.
    """
    text_lower = text.lower()
    for category, keywords in rules.domain_groups.items():
        if any(kw in text_lower for kw in keywords):
            return category, []
    categories = ", ".join(rules.domain_groups.keys())
    return None, [f"Document does not clearly belong to any known deferment category ({categories})"]


def _check_institution(text: str, rules: InstructionRules) -> bool:
    """Return True if any recognised institution name is found in the document."""
    text_lower = text.lower()
    return any(inst in text_lower for inst in rules.recognised_institutions)


def _check_dates(text: str, rules: InstructionRules) -> list[str]:
    """
    Find all 4-digit years in the document and flag any outside the valid range.
    """
    flags = []
    min_year, max_year = rules.valid_year_range
    years_found = re.findall(r"\b(20\d{2})\b", text)
    for year_str in set(years_found):
        yr = int(year_str)
        if yr < min_year or yr > max_year:
            flags.append(f"Year {yr} is outside the valid range ({min_year}-{max_year})")
    return flags


def _check_fraud_signals(text: str) -> list[str]:
    """
    Scan the document text for obvious fraud signal patterns.
    """
    signals = []
    text_lower = text.lower()
    for pattern in FRAUD_SIGNAL_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            signals.append(f"Suspicious pattern detected: '{match.group(0).strip()}'")
    return signals


def _check_document_length(text: str) -> list[str]:
    """Flag documents that are suspiciously short (likely incomplete or blank)."""
    word_count = len(text.split())
    if word_count < 30:
        return [f"Document is too short ({word_count} words) — may be incomplete or blank"]
    return []


def check_document(pdf_path: str, rules: InstructionRules) -> CheckResult:
    """
    Run all checks on a single PDF and return a CheckResult.
    """
    filename = Path(pdf_path).name
    reject_reasons = []
    warnings = []

    # ── Step 1: Extract text ──────────────────────────────────────────────────
    try:
        text = extract_text(pdf_path)
    except RuntimeError as e:
        # RuntimeError means a tool (Poppler/Tesseract) is missing — not a doc problem
        return CheckResult(
            filename=filename,
            verdict="WARNING",
            reasons=[f"Could not process (tool missing): {e}"],
        )
    except Exception as e:
        return CheckResult(
            filename=filename,
            verdict="WARNING",
            reasons=[f"Could not extract text from document: {e}"],
        )

    preview = text[:300].replace("\n", " ")

    # -- Step 2: Document too short? ------------------------------------------
    reject_reasons.extend(_check_document_length(text))

    # -- Step 3: Universal required fields present? ---------------------------
    missing_keywords = _check_required_keywords(text, rules)
    if missing_keywords:
        reject_reasons.append(
            f"Missing required fields: {', '.join(missing_keywords)}"
        )

    # -- Step 4: Matches at least one deferment category? --------------------
    matched_category, category_warnings = _check_domain_category(text, rules)
    if matched_category:
        warnings.append(f"Category matched: {matched_category}")
    else:
        warnings.extend(category_warnings)

    # -- Step 5: Recognised institution? --------------------------------------
    if not _check_institution(text, rules):
        warnings.append(
            "No recognised institution name found -- verify manually"
        )

    # -- Step 6: Date validity (warn only, don't reject for old dates) --------
    date_flags = _check_dates(text, rules)
    if date_flags:
        warnings.extend([f"NOTE: {f}" for f in date_flags])

    # -- Step 7: Fraud signals ------------------------------------------------
    fraud_signals = _check_fraud_signals(text)
    reject_reasons.extend(fraud_signals)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = "REJECT" if reject_reasons else "APPROVE"

    return CheckResult(
        filename=filename,
        verdict=verdict,
        reasons=reject_reasons,
        warnings=warnings,
        extracted_text_preview=preview,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_result(result: CheckResult):
    if result.verdict == "APPROVE":
        verdict_display = "[APPROVE]"
    elif result.verdict == "REJECT":
        verdict_display = "[REJECT] "
    else:
        verdict_display = "[WARNING] could not process"
    print(f"\n" + "-"*60)
    print(f"  File    : {result.filename}")
    print(f"  Verdict : {verdict_display}")
    if result.reasons:
        print(f"  Reasons :")
        for r in result.reasons:
            print(f"    - {r}")
    if result.warnings:
        print(f"  Notes:")
        for w in result.warnings:
            print(f"    ! {w}")
    print(f"  Preview : {result.extracted_text_preview[:120]}...")


def save_results_to_csv(results: list[CheckResult], output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "verdict", "reasons", "warnings"])
        for r in results:
            writer.writerow([
                r.filename,
                r.verdict,
                " | ".join(r.reasons),
                " | ".join(r.warnings),
            ])
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check deferment supporting PDFs against the instructions directive."
    )
    parser.add_argument(
        "--instructions", required=True,
        help="Path to the instructions/directive file (.docx or .pdf)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--document",
        help="Path to a single submission file to check (.pdf, .jpg, .png, etc.)"
    )
    group.add_argument(
        "--folder",
        help="Path to a folder of submission files to check in bulk (.pdf, .jpg, .png, etc.)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="(Optional) Save results to this CSV file path"
    )
    args = parser.parse_args()

    # ── Parse instructions ────────────────────────────────────────────────────
    rules = parse_instructions(args.instructions)

    # ── Collect PDFs to check ─────────────────────────────────────────────────
    if args.document:
        pdf_files = [args.document]
    else:
        folder = Path(args.folder)
        image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"]
        pdf_files = sorted(folder.glob("*.pdf"))
        for pattern in image_exts:
            pdf_files += sorted(folder.glob(pattern))
        if not pdf_files:
            print(f"[ERROR] No supported files found in: {folder}")
            sys.exit(1)
        print(f"\nFound {len(pdf_files)} file(s) in {folder}")

    # ── Run checks ────────────────────────────────────────────────────────────
    results = []
    approved = 0
    rejected = 0
    warnings = 0

    for pdf in pdf_files:
        result = check_document(str(pdf), rules)
        print_result(result)
        results.append(result)
        if result.verdict == "APPROVE":
            approved += 1
        elif result.verdict == "REJECT":
            rejected += 1
        else:
            warnings += 1

    # -- Summary ---------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  SUMMARY: {len(results)} document(s) checked")
    print(f"  [APPROVE]  : {approved}")
    print(f"  [REJECT]   : {rejected}")
    print(f"  [WARNING]  : {warnings}  (install Poppler to process scanned PDFs)")
    print("="*60)

    # -- Optional CSV export --------------------------------------------------
    if args.output:
        save_results_to_csv(results, args.output)


if __name__ == "__main__":
    main()

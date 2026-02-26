"""
rule_parser.py
--------------
Parses your instructions/directive document (.docx or .pdf) and extracts
a set of validation rules.

The instructions document typically defines:
  - What fields a valid document MUST contain (e.g. "must include full name")
  - What institutions are recognised
  - Required date ranges or validity periods
  - Any red flags to look out for

This module reads those rules so check.py can apply them to every submitted PDF.
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from extract_text import extract_text

# ── Recognised institutions (extend this list as needed) ─────────────────────
RECOGNISED_INSTITUTIONS = [
    # Universities
    "national university of singapore", "nus",
    "nanyang technological university", "ntu",
    "singapore management university", "smu",
    "singapore institute of technology", "sit",
    "singapore university of technology and design", "sutd",
    "singapore university of social sciences", "suss",
    # Polytechnics
    "singapore polytechnic",
    "ngee ann polytechnic",
    "temasek polytechnic",
    "republic polytechnic",
    "nyp", "nanyang polytechnic",
    # ITEs
    "ite", "institute of technical education",
    # Overseas (common deferment cases)
    "university", "college", "institute", "polytechnic", "academy",
]

# -- Universal keywords: every deferment doc MUST contain these ---------------
UNIVERSAL_REQUIRED_KEYWORDS = [
    "name",    # applicant's name must appear
    "date",    # a date must be present
]

# -- Domain keyword groups: document must match AT LEAST ONE ------------------
# Each group represents one valid deferment category.
# A document is accepted if it clearly belongs to any one category.
# It is NOT penalised for missing categories that don't apply to it.
DOMAIN_KEYWORD_GROUPS = {
    "study (local/overseas)": ["study", "enrol", "student", "course", "university",
                                "polytechnic", "college", "school", "degree"],
    "employment/work":        ["employ", "work", "company", "organisation", "overseas",
                                "posting", "secondment"],
    "medical internship":     ["housemanship", "medical", "doctor", "hospital",
                                "clinical", "mbbs", "intern"],
    "legal pupillage":        ["pupillage", "lawyer", "advocate", "bar", "legal"],
    "professional exam":      ["accounting", "examination", "professional",
                                "registration", "corporate", "attainment"],
    "marriage":               ["marriage", "matrimon", "civil", "spouse", "wed"],
    "childbirth":             ["birth", "child", "newborn", "infant",
                                "maternity", "paternity", "baby"],
}

# -- Patterns that raise suspicion --------------------------------------------
FRAUD_SIGNAL_PATTERNS = [
    r"\b(photoshop|edited|modified)\b",
    r"\b(fake|forged|falsif)\b",
    r"(.)\1{6,}",   # repeated characters (copy-paste corruption)
]


class InstructionRules:
    """Holds the validation rules applied to every submission document."""

    def __init__(
        self,
        required_keywords: list[str],
        domain_groups: dict,
        recognised_institutions: list[str],
        valid_year_range: tuple[int, int],
        raw_instructions: str,
    ):
        self.required_keywords = required_keywords      # must appear in every doc
        self.domain_groups = domain_groups              # doc must match >= 1 group
        self.recognised_institutions = recognised_institutions
        self.valid_year_range = valid_year_range
        self.raw_instructions = raw_instructions


def _extract_year_range(text: str) -> tuple[int, int]:
    """
    Returns the valid year range for SUBMISSION documents.
    We use a rolling window (3 years ago → 5 years ahead of today)
    rather than dates found in the directive itself, because the directive
    may be an old policy document with dates like 2005 that would incorrectly
    reject all modern submissions.
    """
    import datetime
    current_year = datetime.date.today().year
    return (current_year - 3, current_year + 5)


def parse_instructions(instructions_path: str) -> InstructionRules:
    """
    Read the instructions document (.docx or .pdf) and return an InstructionRules
    object containing all derived validation rules.
    """
    print(f"[Instructions] Parsing rules from: {instructions_path}")
    raw_text = extract_text(instructions_path)
    valid_year_range = _extract_year_range(raw_text)
    print(f"  Universal checks  : {UNIVERSAL_REQUIRED_KEYWORDS}")
    print(f"  Domain categories : {list(DOMAIN_KEYWORD_GROUPS.keys())}")
    print(f"  Valid year range  : {valid_year_range[0]} - {valid_year_range[1]}")
    return InstructionRules(
        required_keywords=UNIVERSAL_REQUIRED_KEYWORDS,
        domain_groups=DOMAIN_KEYWORD_GROUPS,
        recognised_institutions=RECOGNISED_INSTITUTIONS,
        valid_year_range=valid_year_range,
        raw_instructions=raw_text,
    )

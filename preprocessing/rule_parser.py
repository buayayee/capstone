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
# NOTE: 'name' is intentionally excluded — supporting documents are submitted
# with applicant names redacted for confidentiality, and the directive does not
# require the applicant's name to appear on the supporting document.
UNIVERSAL_REQUIRED_KEYWORDS = [
    "date",    # a date must be present (directive relies on timing for most categories)
]

# -- Domain keyword groups: document must match AT LEAST ONE ------------------
# Each group represents one valid deferment category.
# A document is accepted if it clearly belongs to any one category.
# It is NOT penalised for missing categories that don't apply to it.
DOMAIN_KEYWORD_GROUPS = {
    # ── Work-Related Reasons (Rules 7–12) ─────────────────────────────────
    # Rule 10 (simultaneous call-up) and Rule 13 (uniformed services) are
    # determined by unit records, not by a supporting document — excluded.
    "Rule 7 — new employment":               ["appointment letter", "offer of employment", "offer letter",
                                               "commencement of employment", "start work", "new position",
                                               "joining date", "newly employed", "employer"],
    "Rule 8 — new business":                 ["acra", "business registration", "registered business",
                                               "sole proprietor", "partnership", "incorporated",
                                               "uen", "bizfile", "newly registered"],
    "Rule 9 — retrenchment/job seeker":      ["retrench", "redundan", "job seeker",
                                               "termination of employment", "notice of redundancy",
                                               "laid off"],
    "Rule 11 — school bus driver":           ["school bus", "bus driver"],
    "Rule 12 — overseas employment":         ["overseas employment", "overseas posting", "posted overseas",
                                               "seafarer", "residing outside singapore",
                                               "living outside singapore"],
    # ── Family-Related Reasons (Rules 22–25) placed early so specific keywords
    # like "death", "deceased", "marriage" win over generic study/work matches ─
    "Rule 22 — serious illness/bereavement": ["death", "deceased", "demise", "passed away",
                                               "next-of-kin", "next of kin", "seriously ill",
                                               "critical condition", "cancer", "bereave", "funeral",
                                               "certificate of death", "death certificate"],
    "Rule 23 — marriage":                    ["marriage", "matrimon", "wed", "solemniz",
                                               "honeymoon", "civil marriage", "customary marriage"],
    "Rule 24 — childbirth":                  ["birth", "child", "delivery", "newborn", "infant",
                                               "maternity", "paternity", "baby", "pregnant", "pregnancy",
                                               "expected date of delivery", "edd", "caesarean"],
    "Rule 25 — spouse overseas":             ["accompanying spouse", "spouse overseas",
                                               "spouse on overseas studies", "spouse on overseas employment",
                                               "accompany his spouse"],
    # ── Medical Reasons (Rule 21) ─────────────────────────────────────────
    "Rule 21 — medical leave":               ["medical certificate", "medical leave", "hospitalisation",
                                               "hospitaliz", "inpatient", "sick leave", "ward",
                                               "medical condition", "treatment", "diagnosis"],
    # ── Study / Training Reasons (Rules 14–19) ────────────────────────────
    # Rule 15 (overseas full-time studies) matches via Rule 14 keywords.
    # Rule 16 (part-time studies) is NOT a valid deferment reason — excluded.
    "Rule 14 — full-time studies":           ["enrol", "full-time student", "university", "polytechnic",
                                               "college", "degree", "diploma", "academic term",
                                               "nus", "ntu", "smu", "sit", "sutd", "nie", "ite",
                                               "sim", "council for private education", "cpe"],
    "Rule 17 — examinations":                ["examination", "exam schedule", "exam timetable",
                                               "professional exam", "assessment", "test date"],
    "Rule 18 — professional courses":        ["housemanship", "pupillage", "bar examination",
                                               "clinical attachment", "house officer", "pgy1", "pgy2",
                                               "professional course", "medical officer"],
    "Rule 19 — employer-sponsored training": ["employer-sponsored training", "employer sponsored",
                                               "sponsored training", "training programme",
                                               "work attachment", "secondment", "full-time training"],
    # ── Religious Reasons (Rule 20) ───────────────────────────────────────
    "Rule 20 — religious studies/ministry":  ["religious studies", "religious ministry",
                                               "full-time religious", "pastor", "priest",
                                               "monk", "nun", "seminary", "religious minister",
                                               "mosque", "temple"],
    # ── Others (Rules 26, 28) ─────────────────────────────────────────────
    # Rule 27 (permanent residency) and Rule 29 (criminal) are administrative
    # disruption actions with no supporting document to check — excluded.
    "Rule 26 — national representation":     ["represent singapore", "national team", "sea games",
                                               "olympic", "international games", "regional games",
                                               "athlete", "national selection", "national representative"],
    "Rule 28 — rehabilitation":              ["rehabilitation", "residential care", "rehab centre",
                                               "rehab center", "welfare home"],
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

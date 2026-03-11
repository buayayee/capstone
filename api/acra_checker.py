"""
acra_checker.py
---------------
Validates SD company / employer details against the ACRA open data API
(data.gov.sg dataset: d_3f960c10fed6145404ca7b821f263b87 — "Entities Registered with ACRA").

Checks performed:
  1. UEN / company name extraction from OCR text (regex).
  2. Live lookup in the ACRA dataset:
       - uen_status_desc : "Registered" or "Deregistered"
       - entity_name     : canonical company name
       - entity_type_desc: "Local Company", "Sole Proprietorship/ Partnership",
                           "Limited Liability Partnership", "Foreign Company Branch", etc.
  3. Rule 8 window determination:
       - Local Company / Local SP / Local Partnership → 6 months
       - Overseas Partnership                        → 6 months
       - Overseas Sole Proprietorship                → 9 months

If the entity is Deregistered the verdict is "FRAUD" (SD cannot be authenticated).
If the entity is not found the verdict is "WARN" (manual check recommended).

Requires:
  - pip install requests
  - DATAGOV_API_KEY environment variable (optional; raises rate limits)
"""

import os
import re
import json
import requests
import urllib3
from typing import Optional

# difflib for company name fuzzy matching
from difflib import SequenceMatcher

# Suppress SSL warnings that occur on corporate networks with proxy certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
_DATASET_ID = "d_3f960c10fed6145404ca7b821f263b87"
_ACRA_URL   = f"https://data.gov.sg/api/action/datastore_search?resource_id={_DATASET_ID}"

# Rules that involve a company / employer — ACRA check is triggered for these only
ACRA_APPLICABLE_RULES = {"Rule 7", "Rule 8", "Rule 9", "Rule 11", "Rule 12", "Rule 19"}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Singapore UEN formats:
#   Standard  : 8-10 digits + 1 uppercase letter  (e.g. 199600940G, 53250767C)
#   T/S/R-type: T|S|R + 2 digits + 2 letters + 4 digits + 1 letter (statutory boards, etc.)
_UEN_RE = re.compile(
    r'\b([0-9]{8,10}[A-Z]|[TRS][0-9]{2}[A-Z]{2}[0-9]{4}[A-Z])\b'
)

# Government / public institutions that will NOT appear in ACRA as private entities.
# If the extracted company name contains any of these tokens, skip the ACRA check.
_GOVT_TOKENS: frozenset[str] = frozenset([
    # Full-form ministry / govt body names (NOT short acronyms like MOH/MOE/MOM
    # which can appear in genuine ACRA-registered company names, e.g. "MOH Holdings Pte Ltd")
    "MINISTRY OF HEALTH", "MINISTRY OF EDUCATION", "MINISTRY OF MANPOWER",
    "MINISTRY OF DEFENCE", "MINISTRY OF HOME AFFAIRS",
    "MINDEF", "REPUBLIC OF SINGAPORE", "NATIONAL SERVICE",
    "GOVERNMENT OF SINGAPORE",
    # Note: "HOSPITAL" removed — many hospitals ARE ACRA-registered private companies
    # (e.g. KK Women's and Children's Hospital Pte Ltd under SingHealth cluster).
    # _COMPANY_RE already requires a corporate suffix; real govt hospitals (SGH etc.)
    # appear without "Pte Ltd" so they won't be matched anyway.
    "NATIONAL UNIVERSITY", "UNIVERSITY", "POLYTECHNIC", "INSTITUTE OF TECHNICAL",
    "AUTHORITY", "COUNCIL",
    "CENTRAL PROVIDENT", "CPF",
    "SUBORDINATE COURT", "STATE COURT", "SUPREME COURT", "JUDICATURE",
    "ITE ", "NUS ", "NTU ", "SMU ", "SUTD ", "SIT ",
    "SINGHEALTH", "NUHS", "NHG",
])

# Company name: runs of capitalised (title-case OR ALL-CAPS) words ending with a
# known SG entity suffix.  Each word must START with an uppercase letter so that
# lowercase prose ("this letter confirms...") is not accidentally consumed.
# Optional parenthesised qualifier like "(Singapore)" or "(S) Pte Ltd" is allowed.
# Uses re.IGNORECASE only on the entity-suffix alternation, not the word tokens,
# so surrounding lowercase sentence text is never pulled in.
_COMPANY_RE = re.compile(
    r'\b((?:[A-Z][A-Za-z0-9&\'\-\.]*[^\S\n]+|'          # word (title or CAPS) + space
    r'\([A-Z][A-Za-z]*\)[^\S\n]*){1,8}'                  # or (Qualifier) 
    r'(?:Pte\.?[^\S\n]?Ltd\.?|Sdn\.?[^\S\n]?Bhd\.?'
    r'|PTE\.?[^\S\n]?LTD\.?|SDN\.?[^\S\n]?BHD\.?'
    r'|LIMITED|LTD\.?|LLP|LLC|CORP\.?|INC\.?'
    r'|ENTERPRISE|Enterprise'
    r'|TRADING|Trading'
    r'|SERVICES|Services'
    r'|SOLUTIONS|Solutions'
    r'|HOLDINGS|Holdings'
    r'|GROUP|Group'
    r'|TECHNOLOGIES|Technologies'
    r'|MANAGEMENT|Management'
    r'|CONSULTANCY|Consultancy'
    r'|INDUSTRIES|Industries'
    r'|LOGISTICS|Logistics'
    r'|CONSTRUCTION|Construction'
    r'|ENGINEERING|Engineering))\b'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_uen(text: str) -> Optional[str]:
    """Return the first UEN found in OCR text, or None."""
    m = _UEN_RE.search(text)
    return m.group(1).strip() if m else None


def extract_company_name(text: str) -> Optional[str]:
    """
    Return the first likely private-sector company name found in OCR text, or None.
    Returns None for government / public institution names (hospitals, ministries,
    universities etc.) that will not appear in ACRA as private entities.
    """
    # Legal entity suffixes that do NOT count as substantive words
    _SUFFIX_WORDS = frozenset([
        "PTE", "PTE.", "LTD", "LTD.", "LIMITED", "LLP", "LLC", "SDN", "BHD",
        "CORP", "CORP.", "INC", "INC.", "ENTERPRISE", "ENTERPRISES",
        "TRADING", "SERVICES", "SOLUTIONS", "HOLDINGS", "HOLDING",
        "GROUP", "TECHNOLOGIES", "TECHNOLOGY", "MANAGEMENT", "CONSULTANCY",
        "INDUSTRIES", "LOGISTICS", "CONSTRUCTION", "ENGINEERING",
        "(SINGAPORE)", "(S)",
    ])
    for m in _COMPANY_RE.finditer(text):
        name = m.group(1).strip()
        name_upper = name.upper()
        # Skip if the name contains any government/public institution token
        if any(tok in name_upper for tok in _GOVT_TOKENS):
            continue
        # Require at least 2 substantive (non-suffix) words to avoid false positives
        # like bare "Management Pte Ltd" or "Solutions Group"
        core_words = [
            w for w in name_upper.split()
            if w not in _SUFFIX_WORDS
        ]
        if len(core_words) < 2:
            continue
        return name
    return None


def _headers() -> dict:
    key = os.environ.get("DATAGOV_API_KEY", "")
    return {"x-api-key": key} if key else {}


def _query(params: dict) -> list[dict]:
    """Call the ACRA datastore_search endpoint and return records list."""
    try:
        resp = requests.get(
            _ACRA_URL, params=params,
            headers=_headers(), timeout=10, verify=False,
        )
        resp.raise_for_status()
        return resp.json().get("result", {}).get("records", []) or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Company name fuzzy matching helpers
# ---------------------------------------------------------------------------
# Suffixes to strip before comparing (so "ABC PTE LTD" ≈ "ABC")
_STRIP_SUFFIXES = re.compile(
    r'\s+(PTE\.?\s?LTD\.?|SDN\.?\s?BHD\.?|LIMITED|LTD\.?|LLP|LLC|CORP\.?|INC\.?'
    r'|ENTERPRISE|TRADING|SERVICES|SOLUTIONS|HOLDINGS|GROUP|TECHNOLOGIES'
    r'|MANAGEMENT|CONSULTANCY|INDUSTRIES|LOGISTICS|CONSTRUCTION|ENGINEERING'
    r'|SINGAPORE|S\'PORE)\s*$',
    re.IGNORECASE,
)


def _normalize_name(name: str) -> str:
    """
    Deep normalization for USE AS SEARCH QUERY ONLY:
    Uppercase + remove punctuation + strip legal suffixes (e.g. PTE LTD, ENTERPRISE).
    Returns the core business name token(s) for broad ACRA recall.
    """
    n = name.upper().strip()
    n = re.sub(r'[.\-,\'\"&()]', ' ', n)
    for _ in range(3):
        n2 = _STRIP_SUFFIXES.sub('', n).strip()
        if n2 == n:
            break
        n = n2
    return re.sub(r'\s+', ' ', n).strip()


def _light_normalize(name: str) -> str:
    """
    Light normalization for SCORING only:
    Uppercase + collapse whitespace + strip punctuation — does NOT strip entity suffixes.
    Preserves "ENTERPRISE", "TRADING" etc. so different entities score differently.
    """
    n = name.upper().strip()
    n = re.sub(r'[.\-,\'\"&()]+', ' ', n)
    return re.sub(r'\s+', ' ', n).strip()


def _name_score(query: str, candidate: str) -> float:
    """
    Return a similarity score 0.0–1.0 between two company name strings.
    Uses LIGHT normalization (keeps entity suffixes) so that
    "DYNASOURCE TRADING" does not score high against "DYNASOURCE ENTERPRISE PTE LTD".
    """
    q = _light_normalize(query)
    c = _light_normalize(candidate)

    if not q or not c:
        return 0.0

    # Fast exact match
    if q == c:
        return 1.0

    # Substring containment
    if q in c or c in q:
        return 0.90

    # Character-level similarity
    char_sim = SequenceMatcher(None, q, c).ratio()

    # Word overlap
    q_words = set(q.split())
    c_words = set(c.split())
    union    = q_words | c_words
    overlap  = q_words & c_words
    word_sim = len(overlap) / len(union) if union else 0.0

    return 0.5 * char_sim + 0.5 * word_sim


# Singapore postal code: always 6 digits
_POSTAL_RE = re.compile(r'\bS(?:ingapore)?\s*(\d{6})\b|\b(\d{6})\b')


def extract_postal_code(text: str) -> Optional[str]:
    """Return the first 6-digit Singapore postal code found in OCR text, or None."""
    for m in _POSTAL_RE.finditer(text):
        code = m.group(1) or m.group(2)
        # SG postal codes start with 01-82 (valid districts)
        if code and 1 <= int(code[:2]) <= 82:
            return code
    return None


# Minimum confidence to accept a name-only match
_NAME_MATCH_THRESHOLD = 0.70


def _search_by_name(name: str, postal_code: Optional[str] = None) -> Optional[dict]:
    """
    Query ACRA by company name using full-text search (`q=`).
    Strategy:
      1. Search with full normalized core name (e.g. 'TECH DATA DISTRIBUTION').
      2. Pick the best candidate by name score above _NAME_MATCH_THRESHOLD.
      3. If nothing qualifies but a postal_code was supplied:
         a. Try a SECOND broader search using only the first 2 significant words
            (e.g. 'TECH DATA') and accept whichever result has a matching
            reg_postal_code — handles restructured/rebranded entities.
         b. Try a DIRECT postal-code filter lookup and accept the result with the
            best name score — handles foreign company branches whose registered
            address in ACRA may differ from their letterhead address.
    """
    core = _normalize_name(name)
    rows = _query({"q": core, "limit": 20})

    # Pass 1: score-based match on full core name
    best_record: Optional[dict] = None
    best_score  = 0.0
    for row in rows:
        score = _name_score(name, row.get("entity_name", ""))
        if score > best_score:
            best_score  = score
            best_record = row
    if best_score >= _NAME_MATCH_THRESHOLD:
        return best_record

    if postal_code:
        # Pass 2a: broader 2-word prefix search + postal tiebreaker
        core_words = core.split()
        if len(core_words) >= 2:
            short_query = " ".join(core_words[:2])
            broad_rows = _query({"q": short_query, "limit": 50})
            for row in broad_rows:
                if row.get("reg_postal_code", "").strip() == postal_code.strip():
                    return row   # postal code is definitive

        # Pass 2b: direct postal code filter — returns all entities at this address
        postal_rows = _query({"filters": json.dumps({"reg_postal_code": postal_code}), "limit": 10})
        if postal_rows:
            # Only accept if name similarity is reasonable — same address confirms same entity,
            # but we need at least partial name overlap to avoid false matches in shared buildings
            best_postal = max(postal_rows, key=lambda r: _name_score(name, r.get("entity_name", "")))
            if _name_score(name, best_postal.get("entity_name", "")) >= 0.50:
                return best_postal

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_acra(document_text: str, rule_category: str | None = None) -> dict:
    """
    Extract UEN / company name from OCR text, query ACRA, and return:

    {
        "uen"         : str | None,
        "entity_name" : str | None,
        "uen_status"  : "Registered" | "Deregistered" | "Not found" | "Skipped",
        "entity_type" : str | None,
        "rule8_months": int | None,   # applicable window in months (Rule 8 only)
        "verdict"     : "PASS" | "FRAUD" | "WARN" | "SKIP",
        "notes"       : [str],
    }
    """
    result: dict = {
        "uen":          None,
        "entity_name":  None,
        "uen_status":   "Skipped",
        "entity_type":  None,
        "rule8_months": None,
        "verdict":      "SKIP",
        "notes":        [],
    }

    uen    = extract_uen(document_text)
    name   = extract_company_name(document_text)
    postal = extract_postal_code(document_text)

    if not uen and not name:
        result["notes"].append(
            "[ACRA] No UEN or company name detected in document — ACRA check skipped."
        )
        return result

    result["uen"] = uen

    # --- Try lookup by UEN (exact match, most reliable) ---
    record: Optional[dict] = None
    matched_by = None
    if uen:
        rows = _query({"filters": json.dumps({"uen": uen}), "limit": 1})
        if rows:
            record = rows[0]
            matched_by = "UEN"

    # --- Fallback: name-based fuzzy search (with postal code tiebreaker) ---
    if record is None and name:
        record = _search_by_name(name, postal_code=postal)
        if record:
            matched_by = "name" if _name_score(name, record.get("entity_name", "")) >= _NAME_MATCH_THRESHOLD else "postal code"

    # --- No match in ACRA database ---
    if record is None:
        result["uen_status"] = "Not found" if (uen or name) else "Skipped"
        if uen:
            result["verdict"] = "WARN"
            result["notes"].append(
                f"[ACRA] UEN {uen} NOT FOUND in ACRA database "
                f"(Name on SD: {name or 'N/A'}). "
                "Could be an overseas entity or UEN was OCR-misread — manual verification required."
            )
        elif name:
            result["verdict"] = "WARN"
            postal_hint = f" (Postal: {postal})" if postal else ""
            result["notes"].append(
                f"[ACRA] Company '{name}'{postal_hint} not found in ACRA database. "
                "May be a trading name, former name, or foreign branch — manual verification required. "
                "Check data.gov.sg/datasets/d_b1d2b840ab9e993570c037b706b39bb8/view for foreign entities."
            )
        else:
            result["notes"].append(
                "[ACRA] No UEN or company name detected in document — ACRA check skipped."
            )
        # For Rule 8 entities not in ACRA, flag for manual window determination
        if rule_category and "8" in rule_category:
            result["notes"].append(
                "[ACRA] Rule 8: entity not confirmed in ACRA — possibly overseas. "
                "Window is 6 months (local/partnership) or 9 months (overseas sole proprietorship). "
                "Reviewer must confirm entity type."
            )
        return result

    # --- Record found ---
    result["uen"]         = record.get("uen", uen)
    result["entity_name"] = record.get("entity_name")
    status      = record.get("uen_status_desc",   "")
    entity_type = record.get("entity_type_desc",  "")
    result["uen_status"]  = status
    result["entity_type"] = entity_type

    match_label = (
        f"UEN {result['uen']}" if matched_by == "UEN"
        else f"postal code {postal} match (SD name: '{name}')" if matched_by == "postal code"
        else f"name match ('{name}')"
    )

    if "Deregistered" in status:
        result["verdict"] = "FRAUD"
        result["notes"].append(
            f"[ACRA] DEREGISTERED: '{result['entity_name']}' ({match_label}) "
            f"is no longer a registered entity ({entity_type}). "
            "The SD issuer cannot be verified — flag as potentially fraudulent."
        )
    else:
        result["verdict"] = "PASS"
        result["notes"].append(
            f"[ACRA] Verified: '{result['entity_name']}' ({match_label}) "
            f"— Status: {status} | Type: {entity_type}."
        )

    # --- Rule 8: determine applicable registration window ---
    if rule_category and "8" in rule_category:
        is_foreign = "Foreign" in entity_type
        is_sp      = "Sole Proprietorship" in entity_type

        if not is_foreign:
            # All local entities: 6 months
            months = 6
            result["notes"].append(
                f"[ACRA] Rule 8 window: 6 months (local entity — {entity_type})."
            )
        elif is_sp:
            # Overseas sole proprietorship: 9 months
            months = 9
            result["notes"].append(
                "[ACRA] Rule 8 window: 9 months (overseas sole proprietorship)."
            )
        else:
            # Overseas partnership / other: 6 months
            months = 6
            result["notes"].append(
                f"[ACRA] Rule 8 window: 6 months (overseas partnership / {entity_type})."
            )
        result["rule8_months"] = months

    return result

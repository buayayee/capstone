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
from concurrent.futures import ThreadPoolExecutor, as_completed

# difflib for company name fuzzy matching
from difflib import SequenceMatcher

# Suppress SSL warnings that occur on corporate networks with proxy certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
# "Entities Registered with ACRA" — private companies, LLPs, sole proprietorships, etc.
_DATASET_ID  = "d_3f960c10fed6145404ca7b821f263b87"
_ACRA_URL    = f"https://data.gov.sg/api/action/datastore_search?resource_id={_DATASET_ID}"

# "Entities Registered with Other" — statutory boards, societies, trade unions,
# government bodies (e.g. CAAS, MAS, HDB, CPF Board, Registry of Societies, etc.)
_OTHER_DATASET_ID = "d_b1d2b840ab9e993570c037b706b39bb8"
_OTHER_ACRA_URL   = f"https://data.gov.sg/api/action/datastore_search?resource_id={_OTHER_DATASET_ID}"

# Rules that involve a company / employer — ACRA check is triggered for these only
ACRA_APPLICABLE_RULES = {"Rule 7", "Rule 8", "Rule 9", "Rule 11", "Rule 12", "Rule 19"}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Singapore UEN formats:
#   Standard  : 8-10 digits + 1 uppercase letter  (e.g. 199600940G, 53250767C)
#   T/S/R-type: T|S|R + 2 digits + 2 letters + 4 digits + 1 letter (statutory boards, etc.)
_UEN_RE = re.compile(
    # Standard numeric UEN: 8-10 digits then one uppercase letter,
    # not preceded by a digit (avoids matching inside longer numbers),
    # not followed by a digit (avoids partial UEN from longer digit strings).
    # The trailing letter CAN be followed by more letters (e.g. 'KGSTReg').
    r'(?<![0-9])([0-9]{8,10}\s?[A-Z])(?![0-9])'
    r'|'
    # Entity-type prefixes: T/R/S + 2 digits + 2 letters + 4 digits + 1 letter
    r'(?<![A-Z0-9])([TRS][0-9]{2}[A-Z]{2}[0-9]{4}[A-Z])(?![A-Z0-9])'
)

# Government / public institutions that will NOT appear in ACRA as private entities.
# If the extracted company name contains any of these tokens, skip the ACRA check.
_GOVT_TOKENS: frozenset[str] = frozenset([
    # Full-form ministry / govt body names (NOT short acronyms like MOH/MOE/MOM
    # which can appear in genuine ACRA-registered company names, e.g. "MOH Holdings Pte Ltd")
    "MINISTRY OF HEALTH", "MINISTRY OF EDUCATION", "MINISTRY OF MANPOWER",
    "MINISTRY OF DEFENCE", "MINISTRY OF HOME AFFAIRS", "MINISTRY OF FOREIGN AFFAIRS",
    "MINISTRY OF LAW", "MINISTRY OF FINANCE", "MINISTRY OF CULTURE",
    "MINISTRY OF SUSTAINABILITY", "MINISTRY OF SOCIAL", "MINISTRY OF DIGITAL",
    "MINISTRY OF TRANSPORT", "MINISTRY OF NATIONAL DEVELOPMENT",
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

# Regex that inserts a space before an uppercase letter that immediately follows
# a lowercase letter or a digit — the most common OCR merging pattern.
# Examples:
#   'SKhynixAsiaPte Ltd'  → 'SKhynix Asia Pte Ltd'
#   'BritoilOffshoreServices'  → 'Britoil Offshore Services'
#   'StraitsView#12-07Marina'  → 'Straits View#12-07 Marina'
_OCR_MERGE_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')

# Split tokens where OCR fuses an acronym prefix with a lowercase word.
# Example: 'SKhynix' -> 'SK hynix'
_OCR_ACRONYM_WORD_RE = re.compile(r'\b([A-Z]{2,4})([a-z][A-Za-z0-9]*)\b')


def _fix_ocr_spacing(text: str) -> str:
    """
    Insert missing spaces in OCR-merged text.
    Only inserts a space at lowercase→UPPERCASE or digit→UPPERCASE boundaries
    so that legitimate all-caps acronyms (e.g. 'SK', 'BRITOIL') are preserved.
    Also collapses multiple spaces introduced by the split.
    """
    fixed = _OCR_ACRONYM_WORD_RE.sub(r'\1 \2', text)
    fixed = _OCR_MERGE_RE.sub(' ', fixed)
    return re.sub(r'  +', ' ', fixed)


def extract_uen(text: str) -> Optional[str]:
    """Return the first UEN found in OCR text, or None."""
    m = _UEN_RE.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    return re.sub(r'\s+', '', raw).strip()


def extract_company_name(text: str) -> Optional[str]:
    """
    Return the first likely private-sector company name found in OCR text, or None.
    Returns None for government / public institution names (hospitals, ministries,
    universities etc.) that will not appear in ACRA as private entities.

    Applies OCR spacing normalisation first so that merged letterhead text like
    'SKhynixAsiaPte Ltd' is split into 'SK hynix Asia Pte Ltd' before matching.
    """
    # Fix OCR-merged text before running any regex
    text = _fix_ocr_spacing(text)

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


def _query(params: dict, url: str | None = None) -> list[dict]:
    """Call a ACRA datastore_search endpoint and return records list."""
    try:
        resp = requests.get(
            url or _ACRA_URL, params=params,
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

# Minimum confidence when using postal-code fallback.
# Postal code alone is not definitive in shared addresses; require some name overlap.
_POSTAL_FALLBACK_THRESHOLD = 0.60


def _search_by_name(name: str, postal_code: Optional[str] = None,
                    url: str | None = None) -> Optional[dict]:
    """
    Query an ACRA datastore_search endpoint by company name using full-text search (`q=`).
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
    _url = url or _ACRA_URL
    core      = _normalize_name(name)    # suffix-stripped — e.g. "CIVIL AVIATION AUTHORITY OF"
    full_name = _light_normalize(name)   # full name preserved — e.g. "CIVIL AVIATION AUTHORITY OF SINGAPORE"

    # Pass -1 (exact): try entity_name exact filter using the uppercased full name.
    # This is the most reliable path for statutory boards, ministries and govt bodies
    # whose names don't respond well to CKAN full-text ranking.
    exact_rows = _query({"filters": json.dumps({"entity_name": full_name}), "limit": 1}, url=_url)
    if exact_rows:
        candidate = exact_rows[0]
        # Require the candidate's name to START WITH the query to avoid false
        # positives like "Singapore Polytechnic Cru" matching "Singapore Polytechnic"
        _cand_upper = candidate.get("entity_name", "").upper()
        if _cand_upper == full_name or _cand_upper.startswith(full_name + " "):
            if _name_score(name, candidate.get("entity_name", "")) >= _NAME_MATCH_THRESHOLD:
                return candidate

    # Pass 0: search using the full (non-stripped) name first.
    # This is critical for statutory boards / govt bodies whose canonical name includes
    # "Singapore", "Authority", "Council" etc. (which _normalize_name would strip).
    rows_full = _query({"q": full_name, "limit": 20}, url=_url) if full_name != core else []
    for row in rows_full:
        if _name_score(name, row.get("entity_name", "")) >= _NAME_MATCH_THRESHOLD:
            return row

    # Pass 0b: try with just the first 2 significant words of the original name.
    # The CKAN full-text search ranks short distinctive queries better than long ones —
    # e.g. "civil aviation" reliably surfaces CAAS whereas "civil aviation authority" does not.
    _STOP = frozenset(["OF", "THE", "AND", "FOR", "IN", "ON", "BY", "TO", "OR", "A", "AN",
                       "AT", "WITH", "FROM"])
    sig_words = [w for w in full_name.split() if w not in _STOP]
    if len(sig_words) >= 2:
        short_query = " ".join(sig_words[:2])
        if short_query not in {core, full_name}:
            rows_short = _query({"q": short_query, "limit": 20}, url=_url)
            for row in rows_short:
                if _name_score(name, row.get("entity_name", "")) >= _NAME_MATCH_THRESHOLD:
                    return row

    rows = _query({"q": core, "limit": 20}, url=_url)

    # Pass 1: score-based match on full core name
    best_record: Optional[dict] = None
    best_score  = 0.0
    for row in rows:
        score = _name_score(name, row.get("entity_name", ""))
        # Guard: reject if candidate is clearly a sub-entity of the query name
        # (e.g. "Singapore Polytechnic Cru" should not match "Singapore Polytechnic")
        _cand = row.get("entity_name", "").upper()
        _qry  = full_name
        if len(_cand) > len(_qry) + 4 and _cand.startswith(_qry):
            continue
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
            broad_rows = _query({"q": short_query, "limit": 50}, url=_url)
            for row in broad_rows:
                if row.get("reg_postal_code", "").strip() == postal_code.strip():
                    if _name_score(name, row.get("entity_name", "")) >= _POSTAL_FALLBACK_THRESHOLD:
                        return row

        # Pass 2b: direct postal code filter — returns all entities at this address
        postal_rows = _query({"filters": json.dumps({"reg_postal_code": postal_code}), "limit": 10}, url=_url)
        if postal_rows:
            # Only accept if name similarity is reasonable — same address confirms same entity,
            # but we need at least partial name overlap to avoid false matches in shared buildings
            best_postal = max(postal_rows, key=lambda r: _name_score(name, r.get("entity_name", "")))
            if _name_score(name, best_postal.get("entity_name", "")) >= _POSTAL_FALLBACK_THRESHOLD:
                return best_postal

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_acra(document_text: str, rule_category: str | None = None,
               llm_company_hint: str | None = None) -> dict:
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

    # --- LLM-hint preferred path: use AI-identified issuer name when provided ---
    _using_hint = False
    if not uen and llm_company_hint:
        # Fix OCR-merged spacing in the LLM hint (LLM may echo back the raw OCR string)
        hint_fixed = _fix_ocr_spacing(llm_company_hint)
        # Strip trailing parenthetical acronyms/short-forms e.g. "(CAAS)", "(ROM)", "(S)"
        # before querying ACRA — the registry stores full names only
        hint_name = re.sub(r'\s*\([^)]{1,10}\)\s*$', '', hint_fixed).strip()
        # Also strip trailing country qualifiers like ", SINGAPORE" or "(SINGAPORE)"
        hint_name = re.sub(r',?\s*SINGAPORE\s*$', '', hint_name, flags=re.IGNORECASE).strip()

        # --- Compound issuer split ---
        # Contracts name two parties: "Caterpillar S.A.R.L. Singapore Branch and P&A Link Pte Ltd".
        # The LLM may echo the full "between X and Y" string as the ISSUER.
        # Guard: split only when the SECOND (or later) candidate itself contains an
        # entity-suffix keyword AND is >= 2 words — that signals it is a real standalone
        # company name, not a mid-name conjunction like "Research and Development Pte Ltd".
        _entity_suffix_re = re.compile(
            r'\b(pte\.?\s*ltd\.?|sdn\.?\s*bhd\.?|s\.?a\.?r\.?l\.?|llp|llc|corp|inc'
            r'|holdings|group|services|consultancy|authority|ministry|board'
            r'|hospital|clinic|school|university|college|branch)\b',
            re.IGNORECASE,
        )
        _compound_sep = re.compile(r'(?i)\s+(?:and|&)\s+(?=[A-Z])')
        _hint_candidates = [c.strip() for c in _compound_sep.split(hint_name) if c.strip()]
        if len(_hint_candidates) > 1 and (
            # First candidate must be >= 2 words (a 1-word prefix is just part of a name)
            len(_hint_candidates[0].split()) >= 2
            and all(
                # Each non-first candidate must have an entity suffix AND be >= 2 words
                _entity_suffix_re.search(c) and len(c.split()) >= 2
                for c in _hint_candidates[1:]
            )
        ):
            # Try each candidate in ACRA; use the first that resolves.
            _resolved_name = None
            for _cand in _hint_candidates:
                _test_main  = _search_by_name(_cand, postal)
                _test_other = _search_by_name(_cand, postal, _OTHER_ACRA_URL)
                if _test_main or _test_other:
                    _resolved_name = _cand
                    break
            # Heuristic fallback (ACRA resolution above found nothing):
            # prefer the candidate with an entity suffix that is NOT the first one —
            # in "Company A engages Company B Pte Ltd", the local registrable entity
            # is usually B. If all/none have suffixes, fall back to the first.
            if not _resolved_name:
                _suffixed = [c for c in _hint_candidates if _entity_suffix_re.search(c)]
                _resolved_name = _suffixed[-1] if _suffixed else _hint_candidates[0]
            hint_name = _resolved_name

        if hint_name:
            name = hint_name
            _using_hint = True

    if not uen and not name:
        result["notes"].append(
            "[ACRA] No UEN or company name detected in document — ACRA check skipped."
        )
        return result

    result["uen"] = uen

    # --- Try lookup by UEN (exact match, most reliable) ---
    # Search both the main ACRA dataset and the "Entities Registered with Other" dataset
    # (which covers statutory boards, societies, trade unions, government bodies, etc.)
    record: Optional[dict] = None
    matched_by   = None
    _other_record = False   # True when match came from the "Other" dataset

    # UEN and name lookups query both datasets in parallel to halve HTTP round-trip time.
    _params_uen = {"filters": json.dumps({"uen": uen}), "limit": 1} if uen else None

    if uen:
        with ThreadPoolExecutor(max_workers=2) as _pool:
            _f_main  = _pool.submit(_query, _params_uen)
            _f_other = _pool.submit(_query, _params_uen, _OTHER_ACRA_URL)
            rows, other_rows = _f_main.result(), _f_other.result()
        if rows:
            record    = rows[0]
            matched_by = "UEN"
        elif other_rows:
            record        = other_rows[0]
            matched_by    = "UEN"
            _other_record = True

    # --- Fallback: name-based fuzzy search (with postal code tiebreaker) ---
    # Run both dataset searches in parallel.
    if record is None and name:
        with ThreadPoolExecutor(max_workers=2) as _pool:
            _f_main  = _pool.submit(_search_by_name, name, postal)
            _f_other = _pool.submit(_search_by_name, name, postal, _OTHER_ACRA_URL)
            _r_main, _r_other = _f_main.result(), _f_other.result()
        # Prefer main ACRA over Other Registry (private companies checked first)
        if _r_main:
            record     = _r_main
            matched_by = "name" if _name_score(name, record.get("entity_name", "")) >= _NAME_MATCH_THRESHOLD else "postal code"
        elif _r_other:
            record        = _r_other
            matched_by    = "name" if _name_score(name, record.get("entity_name", "")) >= _NAME_MATCH_THRESHOLD else "postal code"
            _other_record = True

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
            hint_label = " [AI-identified]" if _using_hint else ""
            result["notes"].append(
                f"[ACRA] Company '{name}'{hint_label}{postal_hint} not found in ACRA or Other Registry (statutory boards / govt bodies). "
                "May be a foreign entity, trading name, or former name — manual verification required."
            )
        else:
            result["notes"].append(
                "[ACRA] No UEN or company name detected in document — ACRA check skipped."
            )
        # For Rule 8 entities not in ACRA, flag for manual window determination
        if rule_category and re.search(r'\bRule\s*8\b', rule_category, re.IGNORECASE):
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

    _ai_tag = " [AI-identified]" if _using_hint else ""
    _db_tag = " [Other Registry]" if _other_record else ""
    match_label = (
        f"UEN {result['uen']}" if matched_by == "UEN"
        else f"postal code {postal} match (SD name: '{name}'{_ai_tag})" if matched_by == "postal code"
        else f"name match ('{name}'{_ai_tag})"
    )

    if "Deregistered" in status:
        result["verdict"] = "FRAUD"
        result["notes"].append(
            f"[ACRA{_db_tag}] DEREGISTERED: '{result['entity_name']}' ({match_label}) "
            f"is no longer a registered entity ({entity_type}). "
            "The SD issuer cannot be verified — flag as potentially fraudulent."
        )
    else:
        result["verdict"] = "PASS"
        _registry_label = "Other Registry (Statutory/Govt Body)" if _other_record else "ACRA"
        result["notes"].append(
            f"[{_registry_label}] Verified: '{result['entity_name']}' ({match_label}) "
            f"— Status: {status} | Type: {entity_type}."
        )

    # --- Rule 8: determine applicable registration window ---
    if rule_category and re.search(r'\bRule\s*8\b', rule_category, re.IGNORECASE):
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

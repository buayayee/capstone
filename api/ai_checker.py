"""
ai_checker.py
-------------
Uses Amazon Bedrock (Claude 3 Haiku) to:
  1. Detect whether a supporting document is fraudulent or tampered.
  2. Check whether it satisfies the NS deferment directive.

Requires:
  - pip install boto3
  - AWS credentials via ~/.aws/credentials  (run: aws configure)
  - Model access MUST be enabled in AWS Console:
      https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess
      → Modify model access → check "Claude 3 Haiku" → Submit

Usage (from check.py with --ai flag):
  check.py --instructions directives/... --folder submissions/ --ai
"""

import os
import re
import json
import time
from datetime import datetime
from typing import Any

try:
    import boto3
    from botocore.config import Config as BotocoreConfig
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    BotocoreConfig = None
    ClientError = None

try:
    from groq import Groq
except ImportError:
    Groq = None
from dateutil.relativedelta import relativedelta
from acra_checker import check_acra, _fix_ocr_spacing

# Groq / Bedrock backend selection
# Default to a higher-quota direct model to reduce daily token-limit failures.
# You can override via GROQ_MODEL in .env/.shell.
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Amazon Bedrock — Claude 3 Haiku via cross-region inference
# Uses the "us.*" inference profile so calls are routed across US regions,
# matching the quota: "Cross-region model inference tokens per minute for
# Anthropic Claude 3 Haiku".
MODEL = "us.anthropic.claude-3-haiku-20240307-v1:0"
_BEDROCK_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Bedrock has no per-minute token cap on paid accounts, so no pacing needed.
# We keep a tiny gap (2s) purely to avoid thundering-herd in batch mode.
# ---------------------------------------------------------------------------
_LLM_MIN_GAP   = 2.0
_last_llm_call = time.monotonic() - _LLM_MIN_GAP

# ---------------------------------------------------------------------------
# Python-side temporal check — bypasses LLM date arithmetic entirely
# ---------------------------------------------------------------------------

# Keywords that indicate a date is a DATE OF BIRTH — must be excluded from rule window checks
_DOB_KEYWORDS: list[str] = [
    "date of birth", "dob", "d.o.b.", "born on", "birth date",
    "birthdate", "place of birth", "nationality", "nric",
]

# Keywords that anchor a date to a specific deferment rule
_RULE_DATE_KEYWORDS: dict[str, list[str]] = {
    "Rule 7":  ["commencement", "commence", "joined", "employment start",
                "start date", "reporting date"],
    "Rule 8":  ["incorporation", "incorporated", "registration date",
                "registered", "acra", "business commencement",
                "date of incorporation", "date of registration"],
    "Rule 9":  ["retrench", "last day of employment", "last working day",
                "termination", "redundancy"],
    "Rule 17": ["examination", "exam date", "test date"],
    "Rule 22": ["death", "passed away", "serious illness", "diagnosis",
                "admitted", "hospitalised"],
    "Rule 23": ["solemnization", "solemnisation", "marriage", "wedding"],
    "Rule 24": ["expected date of delivery", "edd", "due date",
                "confinement", "estimated delivery"],
    # Rule 14: STUDENT letters from EDUCATIONAL INSTITUTIONS — NOT employer letters
    "Rule 14": ["programme", "program", "semester", "academic", "pursuing",
               "enrol", "matriculat", "full-time student", "full time student",
               "diploma", "degree", "university", "college"],
    # Rule 18: Professional vocational training (housemanship, pupillage, chambering)
    "Rule 18": ["housemanship", "house officer", "houseman", "pgy1", "pgy2",
               "pupillage", "pupil", "chambering", "posting", "rotation",
               "clinical posting", "professional course", "vocational"],
    # Rule 19: Employer writing on behalf of employee for structured workplace training
    "Rule 19": ["sponsored training", "employer-sponsored", "training period",
               "staff training", "on behalf of our employee", "secondment",
               "training programme", "mandatory training"],
    # Rule 12: Overseas employment / civil service posting
    "Rule 12": ["posted overseas", "overseas posting", "posting to", "assigned to",
               "embassy", "high commission", "consulate", "foreign mission",
               "overseas employment", "overseas work", "overseas contract"],
    # Rule 11: School bus driver
    "Rule 11": ["school bus", "bus driver", "sole driver"],
    "Rule 26": ["competition", "tournament", "games", "national representation",
               "selected to represent", "sports association"],
}

_MONTHS_LONG  = (r'January|February|March|April|May|June|July|August|'
                 r'September|October|November|December')
_MONTHS_SHORT = r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'

_DATE_PATTERNS: list[tuple[str, str]] = [
    # day + month name + year (standard and normalised OCR)
    (rf'\b(\d{{1,2}})\s+({_MONTHS_LONG})\s+(\d{{4}})\b',   '%d %B %Y'),
    (rf'\b(\d{{1,2}})\s+({_MONTHS_SHORT})\.?\s+(\d{{4}})\b', '%d %b %Y'),
    # ISO and numeric
    (r'\b(\d{4})-(\d{2})-(\d{2})\b',  '%Y-%m-%d'),
    (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', '%d/%m/%Y'),
    # month name + year only (OCR often omits the day, e.g. "August 2023")
    (rf'\b({_MONTHS_LONG})\s+(\d{{4}})\b',   '%B %Y'),
    (rf'\b({_MONTHS_SHORT})\s+(\d{{4}})\b',  '%b %Y'),
]

_OCR_MONTH_RE = re.compile(rf'(?i)({_MONTHS_LONG}|{_MONTHS_SHORT})')


def _normalize_ocr_dates(text: str) -> str:
    """Insert spaces around month names to fix OCR-merged date strings.
    Handles cases like 'inAugust2023' → 'in August 2023' and
    '16December2024' → '16 December 2024' and 'from13January2025to' → 'from 13 January 2025 to'.
    """
    # 1. Space around month names
    text = _OCR_MONTH_RE.sub(r' \1 ', text)
    # 2. Space before 1-2 digit day numbers immediately after letters (e.g. 'from13')
    text = re.sub(r'([A-Za-z])(\d{1,2})(?=\s)', r'\1 \2', text)
    # 3. Space after 4-digit year numbers immediately followed by letters (e.g. '2023and')
    text = re.sub(r'(\d{4})([A-Za-z])', r'\1 \2', text)
    # 4. Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text


def _extract_dates(text: str) -> list[tuple[datetime, str, str]]:
    """Return list of (datetime, context_window, raw_date_string).
    Normalises OCR-merged date strings (e.g. 'inAugust2023') before matching.
    """
    normalised = _normalize_ocr_dates(text)
    results: list[tuple[datetime, str, str]] = []
    seen: set[str] = set()
    for pattern, fmt in _DATE_PATTERNS:
        for m in re.finditer(pattern, normalised, re.IGNORECASE):
            raw = m.group(0).strip()
            s, e = m.start(), m.end()
            # Use the normalised text for context (close enough for keyword matching)
            context = normalised[max(0, s - 250): e + 250].replace('\n', ' ')
            try:
                dt = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            key = dt.strftime('%Y-%m-%d')
            if key not in seen:
                seen.add(key)
                results.append((dt, context, raw))
    return results


def _python_temporal_check(
    document_text: str,
    windows: dict,   # {rule_label: (description, start_dt|None, end_dt|None)}
    ict_start_dt: datetime | None = None,
    identified_category: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Compare dates extracted from the document against their required windows.
    Violation → ('REJECT', violations, notes).
    All pass / no match → ('PASS', [], notes).
    This runs entirely in Python to avoid LLM date arithmetic errors.

    identified_category: the rule/category the LLM assigned to this document
    (e.g. "Rule 14 - Full-Time Studies"). When provided, only the matching rule
    is checked so that keyword overlap between categories does not cause false
    violations (e.g. study letters triggering Rule 7 employment checks).
    """
    date_hits = _extract_dates(document_text)
    if not date_hits:
        return "PASS", [], [
            "[Python temporal check] No dates found in document — LLM verdict kept."
        ]

    # Derive the single active rule from the LLM-identified category, if any.
    # Only run date-window checks for that rule to avoid cross-category false flags.
    _active_rule: str | None = None
    if identified_category:
        m = re.search(r'Rule\s*(\d+)', identified_category, re.IGNORECASE)
        if m:
            _active_rule = f"Rule {m.group(1)}"

    violations: list[str] = []
    passes: list[str] = []

    for dt, context, raw in date_hits:
        ctx_lower = context.lower()
        for rule_label, (desc, win_start, win_end) in windows.items():
            if win_start is None or win_end is None:
                continue  # Rule 14: window defined by academic period, not a range
            # Skip rules that don't match the identified category
            if _active_rule and rule_label != _active_rule:
                continue
            keywords = _RULE_DATE_KEYWORDS.get(rule_label, [])
            if not any(kw in ctx_lower for kw in keywords):
                continue  # date not contextually related to this rule
            # --- skip dates that look like DOBs (context contains birth-related keywords) ---
            if any(dob_kw in ctx_lower for dob_kw in _DOB_KEYWORDS):
                continue
            # --- for Rule 22: skip dates that are clearly DOBs (>18 years before ICT start) ---
            if rule_label == "Rule 22" and ict_start_dt and (ict_start_dt - dt).days > 365 * 18:
                continue
            # --- do the comparison in Python, not the LLM ---
            if win_start <= dt <= win_end:
                passes.append(
                    f"[Python] {rule_label}: {raw} [OK] "
                    f"({win_start:%d %b %Y} - {win_end:%d %b %Y})"
                )
            else:
                badge = "AFTER the window end" if dt > win_end else "BEFORE the window start"
                violations.append(
                    f"{rule_label}: date {raw} ({dt:%d %b %Y}) is {badge}. "
                    f"Required window: {win_start:%d %b %Y} – {win_end:%d %b %Y}."
                )

    # --- Special Rule 14 check: ICT start must fall within the academic period ---
    # Only run if LLM actually classified this document as Rule 14
    if ict_start_dt and "Rule 14" in windows and (_active_rule is None or _active_rule == "Rule 14"):
        r14_kws = _RULE_DATE_KEYWORDS.get("Rule 14", [])
        r14_dates: list[tuple[datetime, str]] = [
            (dt, raw) for dt, ctx, raw in date_hits
            if any(kw in ctx.lower() for kw in r14_kws)
        ]
        if r14_dates:
            earliest_dt, earliest_raw = min(r14_dates, key=lambda x: x[0])
            latest_dt,   latest_raw   = max(r14_dates, key=lambda x: x[0])
            if ict_start_dt < earliest_dt:
                violations.append(
                    f"Rule 14: ICT start {ict_start_dt:%d %b %Y} is BEFORE "
                    f"the earliest programme date found ({earliest_raw}, {earliest_dt:%d %b %Y}). "
                    "The academic period has not yet started at ICT start."
                )
            elif ict_start_dt > latest_dt:
                violations.append(
                    f"Rule 14: ICT start {ict_start_dt:%d %b %Y} is AFTER "
                    f"the latest programme date found ({latest_raw}, {latest_dt:%d %b %Y}). "
                    "The academic period has already ended before ICT start."
                )
            else:
                passes.append(
                    f"[Python] Rule 14: ICT start {ict_start_dt:%d %b %Y} [OK] "
                    f"within programme period ({earliest_dt:%d %b %Y} - {latest_dt:%d %b %Y})"
                )

    if violations:
        return "REJECT", violations, [f"[Python date FAIL] {v}" for v in violations]
    if passes:
        return "PASS", [], passes
    return "PASS", [], [
        "[Python temporal check] No rule-relevant dates matched — LLM verdict kept."
    ]

_client: Any = None
_groq_client: Any = None

# Disable botocore's own ThrottlingException retry so only our explicit loop
# fires — prevents 4× quota drain per invocation attempt.
_BOTO_CFG = BotocoreConfig(retries={"max_attempts": 1, "mode": "standard"}) if BotocoreConfig else None


def _get_bedrock_client():
    global _client
    if _client is None:
        if boto3 is None:
            raise EnvironmentError(
                "boto3 is not installed for Bedrock fallback. Set GROQ_API_KEY or install boto3."
            )
        key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret  = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        region  = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        # If env vars are set, use them explicitly; otherwise let boto3 use
        # ~/.aws/credentials + ~/.aws/config (set via aws configure) automatically.
        if key_id and secret:
            _client = boto3.client(
                "bedrock-runtime",
                region_name=region or _BEDROCK_REGION,
                aws_access_key_id=key_id,
                aws_secret_access_key=secret,
                config=_BOTO_CFG,
            )
        else:
            # Fall back to boto3 default credential chain + config region
            # region_name=None lets boto3 read region from ~/.aws/config
            _client = boto3.client("bedrock-runtime", region_name=region, config=_BOTO_CFG)
    return _client


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        if Groq is None:
            raise EnvironmentError("groq package is not installed.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _preferred_backend() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return "bedrock"


def _invoke_groq(prompt: str) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        temperature=0.0,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def _invoke_bedrock(prompt: str) -> str:
    client = _get_bedrock_client()

    for attempt in range(4):
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            })
            response = client.invoke_model(
                modelId=MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            resp_body = json.loads(response["body"].read())
            return resp_body["content"][0]["text"].strip()
        except Exception as e:
            if ClientError is not None and isinstance(e, ClientError):
                code = e.response["Error"]["Code"]
                if code == "ThrottlingException":
                    wait = 20 * (attempt + 1)
                    print(f"  [Bedrock throttle] waiting {wait}s before retry {attempt + 1}/4...")
                    time.sleep(wait)
                    continue
            raise

    raise RuntimeError("Bedrock ThrottlingException after 4 retries")


def check_with_groq(
    document_text: str,
    directive_text: str,
    filename: str = "",
    valid_categories: list[str] | None = None,
    ict_start: str | None = None,
    ict_end: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Ask a Groq-hosted LLM to:
      1. Assess whether the document is authentic or fraudulent.
      2. Check whether it satisfies the NS deferment directive.

    valid_categories: list of known-valid deferment categories parsed from the
    directive (e.g. ['study', 'medical', 'childbirth', ...]).  Passing these in
    prevents the LLM from missing categories that fall outside the 6000-char
    directive snippet.

    Returns:
        (verdict, reject_reasons, notes)
        verdict is one of:
          "APPROVE"  — genuine document that satisfies the directive
          "REJECT"   — genuine but does not meet eligibility requirements
          "WARNING"  — flagged suspicious but not conclusively fraudulent
    """
    backend = _preferred_backend()

    # =========================================================================
    # LEVEL 1 — ACRA COMPANY REGISTRY CHECK (runs before everything else)
    # If the issuing entity is deregistered, the SD cannot be authentic.
    # We short-circuit immediately and skip the LLM to save time & quota.
    # =========================================================================
    _acra_early = check_acra(document_text, rule_category=None)
    if _acra_early["verdict"] == "FRAUD":
        fraud_note = _acra_early["notes"][0]
        return (
            "REJECT",
            [fraud_note],
            ["[ACRA] Document rejected at first-level registry check — LLM analysis skipped."]
            + _acra_early["notes"],
        )

    directive_snippet = directive_text[:6000]
    # Fix OCR spacing artefacts (e.g. "SKhynixAsiaPte Ltd" → "SK Hynix Asia Pte Ltd")
    # before feeding text to the LLM and the definitive ACRA lookup.
    document_text = _fix_ocr_spacing(document_text)
    doc_snippet = document_text[:4000]

    # exposed to the Python temporal override after the LLM call
    _py_windows: dict | None = None
    _ict_s_dt: datetime | None = None

    if ict_start and ict_end:
        # Pre-compute eligibility windows so the LLM only needs to compare dates,
        # not calculate them (LLMs are unreliable at date arithmetic).

        def _fmt(d: datetime) -> str:
            return d.strftime("%d %b %Y")

        try:
            ict_s = datetime.strptime(ict_start, "%d %b %Y")
            ict_e = datetime.strptime(ict_end, "%d %b %Y")
            duration_days = (ict_e - ict_s).days + 1

            windows = {
                "Rule 7":  (f"commencement date must be between {_fmt(ict_s - relativedelta(months=3))} and {ict_start}",
                            ict_s - relativedelta(months=3), ict_s),
                "Rule 8":  (f"registration date must be between {_fmt(ict_s - relativedelta(months=6))} and {ict_start}",
                            ict_s - relativedelta(months=6), ict_s),
                "Rule 9":  (f"last employment date must be between {_fmt(ict_s - relativedelta(months=6))} and {ict_start}",
                            ict_s - relativedelta(months=6), ict_s),
                "Rule 14": (f"academic period must include {ict_start}", None, None),
                "Rule 17": (f"exam date must be between {ict_start} and {_fmt(ict_e + relativedelta(weeks=2))}",
                            ict_s, ict_e + relativedelta(weeks=2)),
                "Rule 22": (f"death/illness date must be between {_fmt(ict_s - relativedelta(days=30))} and {_fmt(ict_e + relativedelta(days=30))}",
                            ict_s - relativedelta(days=30), ict_e + relativedelta(days=30)),
                "Rule 23": (f"solemnisation date must be between {ict_start} and {_fmt(ict_e + relativedelta(weeks=1))}",
                            ict_s, ict_e + relativedelta(weeks=1)),
                "Rule 24": (f"EDD must be between {_fmt(ict_s - relativedelta(weeks=8))} and {_fmt(ict_s + relativedelta(weeks=4))}",
                            ict_s - relativedelta(weeks=8), ict_s + relativedelta(weeks=4)),
                "Rule 26": (f"competition date must be between {ict_start} and {_fmt(ict_s + relativedelta(months=6))}",
                            ict_s, ict_s + relativedelta(months=6)),
            }

            _py_windows = windows  # store for Python override after LLM call
            _ict_s_dt = ict_s
            window_lines = "\n".join(f"  {r}: {desc}" for r, (desc, _, _) in windows.items())

            # Build a concrete worked example using actual Rule 8 window
            r8_start = ict_s - relativedelta(months=6)
            r8_example_pass = _fmt(r8_start + relativedelta(months=3))  # mid-window
            r8_example_fail_past = _fmt(r8_start - relativedelta(months=3))  # before window
            r8_example_fail_future = _fmt(ict_s + relativedelta(months=1))   # after ICT start

            temporal_note = (
                f"== ICT / TRAINING DATES ==\n"
                f"  ICT Start    : {ict_start}  ({ict_s.strftime('%B %Y')})\n"
                f"  ICT End      : {ict_end}  ({ict_e.strftime('%B %Y')})\n"
                f"  Duration     : {duration_days} days\n\n"
                f"== PRE-COMPUTED ELIGIBILITY WINDOWS ==\n"
                f"{window_lines}\n\n"
                "CRITICAL DATE REASONING RULES:\n"
                f"  1. The year {ict_s.year + 1} is AFTER the year {ict_s.year}. "
                f"The year {ict_s.year - 1} is BEFORE the year {ict_s.year}.\n"
                f"  2. Any date in {ict_s.year + 1} is MORE THAN ONE YEAR AFTER the ICT start "
                f"({ict_start}). It cannot be within any window that ends at or before {ict_start}.\n"
                "  3. A date is only VALID if it is NUMERICALLY BETWEEN the two window dates.\n"
                f"  4. A date AFTER {ict_start} cannot satisfy a window that ends on {ict_start}.\n\n"
                f"WORKED EXAMPLE (Rule 8 — registration must be between "
                f"{_fmt(r8_start)} and {ict_start}):\n"
                f"  - {r8_example_pass}  → VALID  (falls within the window)\n"
                f"  - {r8_example_fail_past}  → INVALID (before the window start)\n"
                f"  - {r8_example_fail_future}  → INVALID (AFTER the ICT start; later = greater year/month)\n\n"
                "INSTRUCTION: Locate the key date(s) in the SD. For each one, determine whether\n"
                "it falls BETWEEN the two boundary dates of its window (BOTH endpoints inclusive).\n"
                "If the date is outside the window, set VERDICT to REJECT.\n"
                "In NOTES, write: 'Date found in SD: [date]. Window: [window]. Result: PASS/FAIL.'"
            )
        except Exception:
            temporal_note = (
                f"ICT Start: {ict_start}  |  ICT End: {ict_end}\n"
                "Compare the SD's key dates against the ICT window. "
                "REJECT if dates are clearly incompatible with the eligibility window."
            )
    else:
        temporal_note = (
            "== ICT / TRAINING DATES ==\n"
            "  ICT Start : NOT PROVIDED\n"
            "  ICT End   : NOT PROVIDED\n\n"
            "== TEMPORAL ASSUMPTION RULES (NO ICT DATES GIVEN) ==\n"
            "  TIMING IS ASSUMED VALID. Your ONLY task is to verify DOCUMENT CONTENT:\n"
            "  - Does the document have the required letterhead / issuing authority?\n"
            "  - Does it contain the required dates / statements for its category?\n"
            "  - Are the required fields present and legible?\n\n"
            "  DO NOT REJECT for any of the following when ICT dates are unknown:\n"
            "  * Date outside an eligibility window (window cannot be computed)\n"
            "  * ICT not overlapping with a training / academic / marriage period\n"
            "  * Employment commencement too far from ICT start\n"
            "  * Exam date not within ICT window\n\n"
            "  ASSUMPTION TABLE (assume ICT falls within validity window):\n"
            "  Rule 7  (New Employment)      — assume ICT is within 3 months of commencement\n"
            "  Rule 8  (New Business)        — assume ICT is within 6 months of registration\n"
            "  Rule 9  (Retrenchment)        — assume ICT is within 6 months of last employment\n"
            "  Rule 11 (School Bus Driver)   — assume ICT falls within the bus contract period\n"
            "  Rule 12 (Overseas Employment) — assume ICT falls within the overseas posting period\n"
            "  Rule 14 (Studies)             — assume ICT falls within the academic period\n"
            "  Rule 17 (Examinations)        — assume ICT overlaps with exam dates\n"
            "  Rule 18 (Professional Course) — assume ICT overlaps with training period\n"
            "  Rule 19 (Sponsored Training)  — assume ICT overlaps with training period\n"
            "  Rule 22 (Illness/Bereavement) — assume ICT is within 30 days of event\n"
            "  Rule 23 (Marriage)            — assume ICT is within 1 week of solemnisation\n"
            "  Rule 24 (Childbirth)          — assume ICT is near the expected delivery date\n"
            "  Rule 26 (National Rep.)       — assume ICT overlaps with competition dates\n"
        )

    # Per-category SD requirements derived from the GenAI deferment prompt directive.
    # These define exactly what a valid SD must contain for each category,
    # and the eligibility window conditions (if ICT dates are provided).
    _ict_known = bool(ict_start and ict_end)
    _timing_note = (
        "  NOTE: ICT dates ARE known — check both content AND eligibility windows below."
        if _ict_known else
        "  NOTE: ICT dates are NOT known — check CONTENT ONLY. IGNORE all ELIGIBILITY WINDOW "
        "and INVALID-if-date-outside-window criteria. Assume timing is valid per the table above."
    )
    SD_REQUIREMENTS = f"""
== SUPPORTING DOCUMENT (SD) REQUIREMENTS PER CATEGORY ==
For each category, check: (A) SD document content, and (B) eligibility window ONLY if ICT dates given.
{_timing_note}

Rule 7 — New Employment:
  REQUIRED: Official company letterhead, commencement date of employment specified,
            employer signature.
  ELIGIBILITY WINDOW: ICT start date must be within 3 months of employment commencement date
                      (i.e. commencement date is less than 3 months before ICT start).
                      NS activity duration must be longer than 7 days.
  INVALID if: No letterhead, no start date, or ICT start is more than 3 months after commencement.

Rule 8 — Newly Established Business:
  REQUIRED: Document/letterhead from ACRA or equivalent overseas authority,
            date of business registration specified.
  ELIGIBILITY WINDOW: Business registration date must be within 6 months before ICT start
                      (local business). For overseas: 6 months (partnership) or 9 months
                      (sole proprietorship) before ICT start.
  INVALID if: No ACRA document, or registration date is NOT within the required window before ICT.

Rule 9 — Retrenchment / Job Seeker:
  REQUIRED: (1) Document from employer confirming retrenchment or redundancy,
            (2) Last date of employment clearly stated.
  ELIGIBILITY WINDOW: Last date of employment must be within 6 months before ICT start date.
  INVALID if: No employer document, no last employment date,
              or last employment date is more than 6 months before ICT start.
  CRITICAL — DO NOT reject for any of the following — they are NOT required by policy:
    * Explicit written reason for retrenchment
    * Employer signature or company stamp
    * Specific retrenchment clause reference
    * HR department letterhead (any employer document suffices)
  A redundancy/retrenchment notice that states the last day of employment is SUFFICIENT.

Rule 11 — School Bus Driver:
  REQUIRED: Employer letterhead confirming role as school bus driver, school bus contract
            period stated, and that the individual is the sole/only driver on that route.
  INVALID if: No letterhead, no contract period, or another driver can cover the route.

Rule 12 — Overseas Employment:
  REQUIRED: Official letter from employer or government agency confirming that the person
            is CURRENTLY EMPLOYED / POSTED OVERSEAS (including civil service overseas postings,
            embassy/high commission assignments, or private sector overseas work contracts).
            Overseas location and period must be stated.
  APPLIES TO: Singapore civil servants posted abroad (MFA embassy staff, SAF/Home Team overseas
              attachments), private sector employees working overseas.
  ELIGIBILITY WINDOW: ICT start date must fall within the overseas posting/contract period.
  APPROVE if: Official letter states the person is posted/employed overseas during the ICT window.
  INVALID if: No official letter, no overseas location, or posting period does not cover ICT dates.

--- CRITICAL DISAMBIGUATION: Rule 12 vs Rule 26 ---
  * Rule 12 = The person is WORKING or LIVING OVERSEAS (employment, civil service posting,
    embassy assignment). Document is from an EMPLOYER or GOVERNMENT AGENCY confirming posting.
    Key phrases: "posting to", "assigned to", "is employed at", "contract period", embassy, mission.
  * Rule 26 = The person is REPRESENTING SINGAPORE in a SPORTS COMPETITION or GAMES event.
    Document is from a SPORTS BODY or NATIONAL SPORTS ASSOCIATION. Key phrases: "selected to
    represent", "Games", "championship", "tournament", "competition".
  DO NOT confuse an overseas work/civil service posting (Rule 12) with national representation
  in a sporting event (Rule 26). Ministry of Foreign Affairs letters = Rule 12 always.

--- CRITICAL DISAMBIGUATION: Rule 14 vs Rule 18 vs Rule 19 ---
The single most important signal is WHO WROTE THE LETTER:
  * Letter from an EDUCATIONAL INSTITUTION (university, polytechnic, ITE, school)
    confirming the person is a currently ENROLLED STUDENT → Rule 14
  * Letter from a PROFESSIONAL BODY or the training institution for post-graduate
    vocational training (housemanship, pupillage, chambering, articled clerkship) → Rule 18
  * Letter from the person's EMPLOYER (a company or government agency) requesting
    deferment because the EMPLOYEE must attend structured training → Rule 19

Do NOT classify as Rule 14 if:
  - The letterhead is from an employer / company / hospital group / government agency
  - The person is referred to as "Dr", "employee", "staff", "house officer" (not "student")
  - The letter requests deferment on behalf of an employee

Rule 14 — Full-Time Studies (Local or Overseas):
  REQUIRED: Letterhead from an EDUCATIONAL INSTITUTION, programme/course specified,
            academic period specified, confirms person is an enrolled full-time STUDENT.
  ELIGIBILITY WINDOW: ICT start date must fall within the academic period stated in the SD.
  INVALID if: No educational-institution letterhead, no course details, no academic period,
              ICT dates outside period, or letter is from an employer (use Rule 18 or 19 instead).

Rule 14 (Internship) — Compulsory Internship as part of degree:
  REQUIRED: Letterhead from an EDUCATIONAL INSTITUTION, internship dates specified,
            letter explicitly states the internship is mandatory/compulsory as part of the degree.
  ELIGIBILITY WINDOW: ICT start date must overlap with the internship period.
  INVALID if: Internship not stated as mandatory, no dates, or ICT outside internship period.

Rule 17 — Examinations (Part-Time Study / Exam):
  REQUIRED: Letterhead from an education institution, programme/course specified,
            examination dates explicitly stated.
  ELIGIBILITY WINDOW: One or more exam dates must fall within ICT start and ICT end dates
                      (or within 2 weeks after ICT end).
  INVALID if: No exam dates, or no exam dates fall within/near the ICT window.

Rule 18 — Professional Courses (Housemanship, Pupillage, Articled Clerkship, etc.):
  APPLIES TO: Post-graduate vocational training where attendance is mandatory and
              missing it causes serious career consequences (e.g. must repeat entire posting).
  EXAMPLES: Medical housemanship (PGY1/PGY2 House Officer hospital rotations),
            legal pupillage/chambering, pharmacy internship required for registration.
  REQUIRED: Letter from the employer or professional body stating the nature of the
            mandatory training, the training period/posting dates, and career impact of absence.
  ELIGIBILITY WINDOW: ICT start date must overlap with the posted training period.
  APPROVE if: Letter from employer/hospital group (e.g. MOHH, SingHealth, NHG) states
              the person is a House Officer undergoing mandatory PGY rotations and cannot
              be absent for more than a specified number of days — this is SUFFICIENT.
  INVALID if: No letter, no training period dates, or ICT outside training period.

Rule 19 — Employer-Sponsored Training:
  APPLIES TO: An EMPLOYER writing on behalf of an EMPLOYEE who must attend structured
              full-time training (e.g. mandatory company training, secondment, overseas course).
  REQUIRED: Employer letterhead, training programme named, training dates specified,
            confirms training is full-time and attendance is mandatory.
  ELIGIBILITY WINDOW: ICT start date must overlap with the training period.
  APPROVE if: Employer letter states employee must attend full-time structured training
              during the ICT window and absence would have serious consequences.
  INVALID if: No letterhead, no training dates, purely on-the-job informal training,
              or ICT does not overlap with the stated training period.

Rule 20 — Full-Time Religious Studies / Ministry:
  REQUIRED: Letter from a recognised religious institution confirming full-time role/studies.
  INVALID if: No letter from religious institution, or role is part-time.

Rule 21 — Medical Leave:
  REQUIRED: Document from government organisation or medical institution, medical condition
            specified, duration of medical leave stated.
  ELIGIBILITY WINDOW: Medical leave period must overlap with ICT start date.
  INVALID if: No institutional document, no medical condition, no duration, or no overlap.

Rule 22 — Serious Illness or Death of Next-of-Kin:
  REQUIRED: Document from government organisation or medical institution,
            date of death or serious illness diagnosis specified.
  ELIGIBILITY WINDOW: Date of death/illness should be within 30 days of ICT start date.
  INVALID if: No official document, no date, or event is outside the 30-day window.

Rule 23 — Marriage / Honeymoon:
  REQUIRED: Document specifying the solemnisation date (registration or customary marriage).
  ELIGIBILITY WINDOW: Solemnisation date must fall within ICT start date to 1 week after ICT end.
  INVALID if: No solemnisation date, or date is outside the window.

Rule 24 — Childbirth (Wife's Delivery):
  REQUIRED: Medical institution letterhead, estimated delivery date (EDD) specified,
            signed by a doctor, document dated within the past 1 month.
  ELIGIBILITY WINDOW: ICT start date must fall within 4 weeks BEFORE to 8 weeks AFTER the EDD.
  INVALID if: No EDD, no doctor signature, doc older than 1 month, or ICT outside the window.

Rule 25 — Spouse on Overseas Studies or Employment:
  REQUIRED: Business/education institution letterhead confirming spouse's overseas commitment,
            overseas location and period stated.
  INVALID if: No confirmation of spouse's overseas commitment, no dates.

Rule 26 — Representing Singapore in Games:
  REQUIRED: Official selection letter from sports body, competition dates stated.
  ELIGIBILITY WINDOW: ICT start must fall during or within 6 months before the competition.
  INVALID if: No official selection letter, no competition dates, or ICT outside 6-month window.

Rule 28 — Rehabilitation / Residential Care:
  REQUIRED: Letter from approved rehabilitation or residential care centre, period stated.
  INVALID if: No letter from an approved centre, or no period stated.
"""

    if valid_categories:
        categories_list = (
            "\n== KNOWN VALID DEFERMENT CATEGORIES ==\n"
            + "\n".join(f"  - {c}" for c in valid_categories)
            + "\n"
        )
    else:
        categories_list = ""

    prompt = f"""You are a NS (National Service) document verification clerk in Singapore.

YOUR ROLE: Evaluate whether the submitted Supporting Document (SD) is GENUINE and CONTAINS
ALL REQUIRED INFORMATION for its claimed category. You are NOT approving or rejecting the
deferment itself — that is the Commander's decision. You are only judging the SD.

A verdict of APPROVE means: "The SD is authentic and contains all required elements."
A verdict of REJECT means: "The SD is fraudulent, suspicious, or is missing required elements."
A verdict of WARNING means: "The SD appears genuine but has minor issues that need manual review."

=== TASK 1: FRAUD / AUTHENTICITY CHECK ===
Look for signs the document is fake or tampered:
- Misspelled institution names (e.g. "hospiel" instead of "hospital")
- Inconsistent, implausible, or impossible dates
- Grammar or phrasing that no real official letter would use
- Missing standard letterhead elements (organisation name, address, contact, logo)
- Content that looks AI-generated, copy-pasted, or edited

IMPORTANT: OCR garbling (jumbled letters, reversed text) is a SCAN issue, NOT fraud.
NOTE: Applicant names are intentionally REDACTED — a missing name is NOT a fraud signal.

=== CRITICAL DATE & CLASSIFICATION RULES (apply to ALL documents) ===
1. IGNORE any ICT, NS, IPPT, BMT, ORD, or reservist reporting dates mentioned INSIDE the
   SD itself. Use ONLY the ICT start/end dates provided in the TEMPORAL / DATE CHECK section.
2. LETTER ISSUE DATE is NOT a deferment event date. Many official letters contain a date at
   the top indicating when the letter was WRITTEN or ISSUED (e.g. "5 December 2024", "Dear Sir,"
   followed by a date, letterhead date, or "Date: ..."). This date is ONLY the document's
   creation date and must NEVER be used as the key event date (commencement date, exam date,
   wedding date, illness date, etc.). Always look past the issue date to find the actual
   event date described in the body of the letter.
3. Rule 7 (New Employment): The KEY DATE is the employment COMMENCEMENT date — the date
   the employee STARTED WORK. The letter issue/written date is NOT the commencement date.
   Look for phrases like "commencement date", "start date", "employed since", "joining date".
4. Rule 23 (Marriage): The KEY DATE is the actual SOLEMNISATION / WEDDING date — the date
   the couple was legally married. An ONLINE APPLICATION submission date or booking
   confirmation date is NOT the solemnisation date. Reject only if the actual wedding
   date falls outside the eligibility window.
5. Rule 22 (Bereavement/Illness): A date of birth (DOB) printed in a death certificate or
   medical document is NOT a programme or event start date. Focus only on the DATE OF DEATH
   or DATE OF DIAGNOSIS as the key event date.
6. Rule 14 (Full-Time Studies): The NSman must be ENROLLED AS A STUDENT. If the NSman is
   a TEACHER, LECTURER, TUTOR, INSTRUCTOR, or STAFF MEMBER at the institution, this does
   NOT qualify under Rule 14. REJECT with reason: "Applicant is not enrolled as a student;
   Rule 14 requires the NSman to be the enrolled student, not teaching/administrative staff."

=== TASK 2: SD CONTENT VALIDATION ==={categories_list}
{SD_REQUIREMENTS}
Step 1 — Identify which category this SD belongs to (use the categories list above).
Step 2 — Check that the SD contains ALL required elements for that category (see requirements above).
Step 3 — List any missing required elements as reasons for REJECT.

== DIRECTIVE REFERENCE ==
{directive_snippet}

== SUBMITTED DOCUMENT ==
Filename: {filename}
{doc_snippet}

== TEMPORAL / DATE CHECK ==
{temporal_note}

=== YOUR RESPONSE ===
Respond in EXACTLY this format (no extra text before or after):

FRAUD_ASSESSMENT: GENUINE or SUSPICIOUS
FRAUD_REASONS: <specific fraud signals, or "None">
ISSUER: <full name of the company, institution, or government authority that issued this document — exactly as it appears on the letterhead>
CATEGORY: <which deferment rule/category this SD belongs to>
VERDICT: APPROVE or REJECT or FRAUD
REASONS: <missing required SD elements if REJECT, fraud details if FRAUD, else "None">
NOTES: <key dates found in the SD, and any observations for the reviewing clerk>

Use FRAUD if the document appears forged or fabricated.
Use REJECT if the SD is genuine but is missing one or more required elements for its category.
Use APPROVE if the SD is genuine and contains all required elements for its category.
"""

    # Adaptive pacing: only sleep whatever gap remains since the last LLM call.
    # Single-case runs spend 15-25s on OCR + ACRA before reaching here, so the
    # sleep is typically 0. Batch runs still get the full 12s inter-call gap.
    global _last_llm_call
    _gap = _LLM_MIN_GAP - (time.monotonic() - _last_llm_call)
    if _gap > 0.05:
        print(f"  [Rate-limit pacing] sleeping {_gap:.1f}s ...")
        time.sleep(_gap)

    if backend == "groq":
        response_text = _invoke_groq(prompt)
    else:
        response_text = _invoke_bedrock(prompt)

    _last_llm_call = time.monotonic()
    verdict, reasons, notes, _llm_category, _llm_issuer = _parse_response(response_text)

    # --- Definitive ACRA check: always use LLM-extracted issuer as primary name ----
    # The LLM reads the clean document and extracts the ISSUER name reliably even
    # when raw OCR merges words (e.g. "SKhynixAsiaPte Ltd" → "SK Hynix Asia Pte Ltd").
    # We always run an authoritative ACRA lookup using the AI name when available,
    # falling back to the regex-based result only when the LLM found no issuer.
    if _llm_issuer:
        _acra_definitive = check_acra(
            document_text, rule_category=None, llm_company_hint=_llm_issuer
        )
        # Promote the AI-powered result unconditionally — it's always more reliable
        # than the raw-OCR regex pass.
        _acra_early = _acra_definitive
        # FRAUD escalation: override verdict immediately
        if _acra_definitive["verdict"] == "FRAUD":
            fraud_note = _acra_definitive["notes"][0]
            return (
                "REJECT",
                [fraud_note],
                ["[ACRA] Document rejected at registry check (AI-identified issuer)."]
                + _acra_definitive["notes"],
            )
    # -------------------------------------------------------------------------------

    # --- Python-side temporal override -------------------------------------------
    # The LLM cannot reliably do date arithmetic. After parsing its verdict we run
    # a pure-Python date comparison against the pre-computed eligibility windows.
    # If Python finds a violation, we override the LLM verdict unconditionally.
    # We pass the LLM-identified category so the check only fires for the matching rule.
    if _py_windows:
        py_v, py_reasons, py_notes = _python_temporal_check(
            document_text, _py_windows,
            ict_start_dt=_ict_s_dt,
            identified_category=_llm_category,
        )
        if py_v == "REJECT":
            verdict = "REJECT"
            # Keep category note at the front, then Python failure notes, then LLM reasons
            cat_notes = [n for n in notes if n.startswith("SD Category:")]
            other_notes = [n for n in notes if not n.startswith("SD Category:")]
            notes = cat_notes + py_notes + other_notes
            reasons = py_reasons + [r for r in reasons if r not in py_reasons]
        else:
            # Append Python pass/info notes after LLM notes
            notes += py_notes
    # ------------------------------------------------------------------------------

    # --- ACRA result notes (company verified at Level 1 before LLM) ---------------
    # Append the ACRA registry result to notes for the reviewing officer.
    # FRAUD was already caught above; here we only propagate PASS/WARN/SKIP notes.
    notes += _acra_early["notes"]

    # Rule 8 entity-type window supplement: re-use the early result's entity_type
    # to determine whether the 9-month overseas SP window applies.
    if _llm_category and re.search(r'\bRule\s*8\b', _llm_category, re.IGNORECASE) and _acra_early.get("entity_type"):
        entity_type = _acra_early["entity_type"]
        is_foreign = "Foreign" in entity_type
        is_sp      = "Sole Proprietorship" in entity_type
        if is_foreign and is_sp:
            notes.append(
                "[ACRA] Rule 8: entity is an overseas sole proprietorship — "
                "applicable registration window is 9 months (not the default 6)."
            )
        else:
            notes.append(
                f"[ACRA] Rule 8: applicable registration window is 6 months ({entity_type})."
            )
    # ------------------------------------------------------------------------------

    return verdict, reasons, notes


def _parse_response(response: str) -> tuple[str, list[str], list[str], str]:
    """Parse the LLM's structured response into (verdict, reasons, notes)."""
    verdict = "WARNING"
    found_verdict = False
    reasons: list[str] = []
    notes: list[str] = []
    fraud_reasons: list[str] = []
    fraud_assessment = ""
    category = ""
    issuer = ""

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("FRAUD_ASSESSMENT:"):
            fraud_assessment = line.split(":", 1)[1].strip().upper()
        elif line.startswith("FRAUD_REASONS:"):
            raw = line.split(":", 1)[1].strip()
            if raw and raw.lower() != "none":
                fraud_reasons = [r.strip() for r in raw.split(",") if r.strip()]
        elif line.startswith("ISSUER:"):
            issuer = line.split(":", 1)[1].strip()
        elif line.startswith("CATEGORY:"):
            category = line.split(":", 1)[1].strip()
        elif line.startswith("VERDICT:"):
            found_verdict = True
            v = line.split(":", 1)[1].strip().upper()
            if "APPROVE" in v:
                verdict = "APPROVE"
            elif "FRAUD" in v:
                verdict = "REJECT"  # FRAUD maps to REJECT for downstream compat
                reasons.insert(0, "Document appears fraudulent")
            elif "REJECT" in v:
                verdict = "REJECT"
        elif line.startswith("REASONS:"):
            raw = line.split(":", 1)[1].strip()
            if raw and raw.lower() != "none":
                reasons += [r.strip() for r in raw.split(",") if r.strip()]
        elif line.startswith("NOTES:"):
            raw = line.split(":", 1)[1].strip()
            if raw:
                notes.append(raw)

    # Surface fraud signals in notes
    if fraud_reasons:
        notes.append(f"Fraud signals: {'; '.join(fraud_reasons)}")

    # Prepend identified category to notes so it shows in output/CSV
    if category:
        notes.insert(0, f"SD Category: {category}")

    # If LLM is conflicted (suspicious doc but still approved) → flag for review
    if fraud_assessment == "SUSPICIOUS" and verdict == "APPROVE":
        verdict = "WARNING"
        notes.append("Document flagged as suspicious — manual review recommended")

    # Only trigger parse-failure fallback if the VERDICT line was never found
    if not found_verdict:
        notes.append(f"AI response could not be parsed: {response[:200]}")

    return verdict, reasons, notes, category, issuer

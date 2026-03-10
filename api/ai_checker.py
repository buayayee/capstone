"""
ai_checker.py
-------------
Uses the Groq API to:
  1. Detect whether a supporting document is fraudulent or tampered.
  2. Check whether it satisfies the NS deferment directive.

Requires:
  - pip install groq
  - Environment variable GROQ_API_KEY (https://console.groq.com/)

Usage (from check.py with --ai flag):
  check.py --instructions directives/... --folder submissions/ --ai
"""

import os
import re
import time
from datetime import datetime
from groq import Groq, RateLimitError
from dateutil.relativedelta import relativedelta

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ---------------------------------------------------------------------------
# Python-side temporal check — bypasses LLM date arithmetic entirely
# ---------------------------------------------------------------------------

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
    "Rule 14": ["programme", "program", "semester", "academic", "pursuing",
               "started", "enrol", "matriculat", "complete", "course",
               "study", "studies", "full-time", "full time", "diploma",
               "degree", "university", "college", "institution"],
    "Rule 26": ["competition", "tournament", "games", "national representation"],
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
) -> tuple[str, list[str], list[str]]:
    """
    Compare dates extracted from the document against their required windows.
    Violation → ('REJECT', violations, notes).
    All pass / no match → ('PASS', [], notes).
    This runs entirely in Python to avoid LLM date arithmetic errors.
    """
    date_hits = _extract_dates(document_text)
    if not date_hits:
        return "PASS", [], [
            "[Python temporal check] No dates found in document — LLM verdict kept."
        ]

    violations: list[str] = []
    passes: list[str] = []

    for dt, context, raw in date_hits:
        ctx_lower = context.lower()
        for rule_label, (desc, win_start, win_end) in windows.items():
            if win_start is None or win_end is None:
                continue  # Rule 14: window defined by academic period, not a range
            keywords = _RULE_DATE_KEYWORDS.get(rule_label, [])
            if not any(kw in ctx_lower for kw in keywords):
                continue  # date not contextually related to this rule
            # --- do the comparison in Python, not the LLM ---
            if win_start <= dt <= win_end:
                passes.append(
                    f"[Python] {rule_label}: {raw} ✓ "
                    f"({win_start:%d %b %Y} – {win_end:%d %b %Y})"
                )
            else:
                badge = "AFTER the window end" if dt > win_end else "BEFORE the window start"
                violations.append(
                    f"{rule_label}: date {raw} ({dt:%d %b %Y}) is {badge}. "
                    f"Required window: {win_start:%d %b %Y} – {win_end:%d %b %Y}."
                )

    # --- Special Rule 14 check: ICT start must fall within the academic period ---
    if ict_start_dt and "Rule 14" in windows:
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
                    f"[Python] Rule 14: ICT start {ict_start_dt:%d %b %Y} ✓ "
                    f"within programme period ({earliest_dt:%d %b %Y} – {latest_dt:%d %b %Y})"
                )

    if violations:
        return "REJECT", violations, [f"[Python date FAIL] {v}" for v in violations]
    if passes:
        return "PASS", [], passes
    return "PASS", [], [
        "[Python temporal check] No rule-relevant dates matched — LLM verdict kept."
    ]

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set.\n"
                "Get a free key at https://console.groq.com/ and run:\n"
                "  $env:GROQ_API_KEY = 'your-key-here'   (PowerShell)\n"
                "  export GROQ_API_KEY='your-key-here'   (bash/zsh)"
            )
        _client = Groq(api_key=api_key)
    return _client


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
    client = _get_client()

    directive_snippet = directive_text[:6000]
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
            "TEMPORAL NOTE: No ICT dates were provided. You cannot verify temporal overlap.\n"
            "Do NOT reject solely due to timing. Note key dates found in the SD in NOTES\n"
            "so a clerk can cross-check against MINDEF's records. Assume timing is valid."
        )

    # Per-category SD requirements derived from the GenAI deferment prompt directive.
    # These define exactly what a valid SD must contain for each category,
    # and the eligibility window conditions (if ICT dates are provided).
    SD_REQUIREMENTS = """
== SUPPORTING DOCUMENT (SD) REQUIREMENTS PER CATEGORY ==
For each category, check: (A) SD document content, and (B) eligibility window if ICT dates given.

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
  REQUIRED: Employer letterhead, explicit statement of retrenchment/redundancy,
            last date of employment specified.
  ELIGIBILITY WINDOW: Last date of employment must be within 6 months before ICT start date.
  INVALID if: No letterhead, no retrenchment confirmation, no last employment date,
              or last employment date is more than 6 months before ICT start.

Rule 14 — Full-Time Studies (Local or Overseas):
  REQUIRED: Letterhead from an education institution, programme/course specified,
            academic period specified, ideally states full-time programme.
  ELIGIBILITY WINDOW: ICT start date must fall within the academic period stated in the SD.
  INVALID if: No letterhead, no course details, no academic period, or ICT dates outside period.

Rule 14 (Internship) — Compulsory Internship as part of degree:
  REQUIRED: Letterhead from an education institution, internship dates specified,
            letter explicitly states the internship is mandatory/compulsory.
  ELIGIBILITY WINDOW: ICT start date must overlap with the internship period.
  INVALID if: Internship not stated as mandatory, no dates, or ICT outside internship period.

Rule 17 — Examinations (Part-Time Study / Exam):
  REQUIRED: Letterhead from an education institution, programme/course specified,
            examination dates explicitly stated.
  ELIGIBILITY WINDOW: One or more exam dates must fall within ICT start and ICT end dates
                      (or within 2 weeks after ICT end).
  INVALID if: No exam dates, or no exam dates fall within/near the ICT window.

Rule 18 — Professional Courses (Housemanship, Pupillage, etc.):
  REQUIRED: Letterhead from a recognised institution, nature of professional course stated,
            dates or period specified.
  ELIGIBILITY WINDOW: ICT start date must overlap with the course/training period.
  INVALID if: No letterhead, no course details, no dates, or ICT outside course period.

Rule 19 — Employer-Sponsored Training:
  REQUIRED: Business/government letterhead, training dates specified, full-time training confirmed.
  ELIGIBILITY WINDOW: ICT start date must overlap with the training period.
  INVALID if: No letterhead, no training dates, on-the-job training only, or ICT outside period.

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
CATEGORY: <which deferment rule/category this SD belongs to>
VERDICT: APPROVE or REJECT or FRAUD
REASONS: <missing required SD elements if REJECT, fraud details if FRAUD, else "None">
NOTES: <key dates found in the SD, and any observations for the reviewing clerk>

Use FRAUD if the document appears forged or fabricated.
Use REJECT if the SD is genuine but is missing one or more required elements for its category.
Use APPROVE if the SD is genuine and contains all required elements for its category.
"""

    # Proactive pacing: Groq free tier is ~30K TPM; each call uses ~5K tokens,
    # so wait 12s between calls to stay safely under the per-minute limit.
    time.sleep(12)

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0,
            )
            break
        except RateLimitError:
            wait = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s
            print(f"  [Rate limit] waiting {wait}s before retry {attempt + 1}/4...")
            time.sleep(wait)
    else:
        raise RuntimeError("Rate limit exceeded after 4 retries")

    response_text = response.choices[0].message.content.strip()
    verdict, reasons, notes = _parse_response(response_text)

    # --- Python-side temporal override -------------------------------------------
    # The LLM cannot reliably do date arithmetic. After parsing its verdict we run
    # a pure-Python date comparison against the pre-computed eligibility windows.
    # If Python finds a violation, we override the LLM verdict unconditionally.
    if _py_windows:
        py_v, py_reasons, py_notes = _python_temporal_check(document_text, _py_windows, ict_start_dt=_ict_s_dt)
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

    return verdict, reasons, notes


def _parse_response(response: str) -> tuple[str, list[str], list[str]]:
    """Parse the LLM's structured response into (verdict, reasons, notes)."""
    verdict = "WARNING"
    found_verdict = False
    reasons: list[str] = []
    notes: list[str] = []
    fraud_reasons: list[str] = []
    fraud_assessment = ""
    category = ""

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("FRAUD_ASSESSMENT:"):
            fraud_assessment = line.split(":", 1)[1].strip().upper()
        elif line.startswith("FRAUD_REASONS:"):
            raw = line.split(":", 1)[1].strip()
            if raw and raw.lower() != "none":
                fraud_reasons = [r.strip() for r in raw.split(",") if r.strip()]
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

    return verdict, reasons, notes

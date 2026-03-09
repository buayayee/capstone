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
import time
from groq import Groq, RateLimitError

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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

    if valid_categories:
        categories_hint = (
            "\n== KNOWN VALID DEFERMENT CATEGORIES (authoritative — from full directive) ==\n"
            + "\n".join(f"  - {c}" for c in valid_categories)
            + "\n\n"
            "== CATEGORY CLARIFICATIONS ==\n"
            "  - 'employment/work' covers: new job offers, overseas postings, secondments,\n"
            "    RETRENCHMENT / REDUNDANCY letters (job seeker after retrenchment), and\n"
            "    any letter confirming the applicant is employed or seeking employment.\n"
            "    A notice of redundancy or termination of employment IS a valid employment/work document.\n"
            "  - 'study (local/overseas)' covers: enrolment letters, internships tied to a degree,\n"
            "    student exchange programmes.\n"
            "  - 'medical internship' covers: housemanship (PGY1/PGY2), clinical attachments.\n"
            "  - 'childbirth' covers: birth of child, pregnancy/delivery certificates.\n"
        )
    else:
        categories_hint = ""

    prompt = f"""You are a senior NS (National Service) deferment clerk in Singapore with expertise in document fraud detection.

You must perform TWO checks on the submitted supporting document:

=== CHECK 1: AUTHENTICITY / FRAUD DETECTION ===
Look carefully for signs that the document may be fake or tampered with:
- Misspelled institution or hospital names (e.g. "hospiel" instead of "hospital")
- Inconsistent or implausible dates
- Unusual grammar or wording that real official letters would not use
- Claims that are too vague or too convenient (e.g. exemption for exactly the ICT period)
- Missing standard letterhead elements (address, registration number, contact details)
- Any text that looks copy-pasted or AI-generated in a suspicious way

IMPORTANT: If the text looks garbled, reversed, or poorly OCR-scanned, that is a TECHNICAL issue, NOT a fraud signal. Only flag SUSPICIOUS if the content itself is suspicious, not the scan quality.
Note: Applicant names are intentionally REDACTED for privacy — a missing name is NOT a fraud signal.

=== CHECK 2: DIRECTIVE COMPLIANCE ==={categories_hint}
Using the directive below, decide whether the document justifies an NS deferment.
If the document relates to any of the known valid categories listed above, it qualifies.

== DIRECTIVE / POLICY ==
{directive_snippet}

== SUBMITTED DOCUMENT ==
Filename: {filename}
{doc_snippet}

=== YOUR RESPONSE ===
Respond in EXACTLY this format (no extra text before or after):

FRAUD_ASSESSMENT: GENUINE or SUSPICIOUS
FRAUD_REASONS: <specific fraud signals found, or "None">
VERDICT: APPROVE or REJECT or FRAUD
REASONS: <specific reasons if REJECT or FRAUD, else "None">
NOTES: <brief summary of what the document is and your overall conclusion>

Use FRAUD as the verdict if you believe the document is forged or fabricated.
Use REJECT if the document appears genuine but does not meet the deferment criteria.
Use APPROVE if the document is genuine and satisfies a valid deferment reason.

SPECIAL NOTE — TEMPORAL CONDITIONS: Some categories (e.g. retrenchment within 6 months of callup,
childbirth within a certain window) have time-based eligibility requirements. The NS callup date
is NOT in the supporting document — it is in MINDEF's own records. You cannot verify temporal
conditions from the document alone. If the document is genuine and falls under a valid category,
APPROVE it and note the key date(s) in NOTES so a clerk can verify the time window.
"""

    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0,
            )
            break
        except RateLimitError:
            wait = 30 * (attempt + 1)  # 30s, 60s, 90s
            print(f"  [Rate limit] waiting {wait}s before retry {attempt + 1}/3...")
            time.sleep(wait)
    else:
        raise RuntimeError("Rate limit exceeded after 3 retries")

    response_text = response.choices[0].message.content.strip()
    return _parse_response(response_text)


def _parse_response(response: str) -> tuple[str, list[str], list[str]]:
    """Parse the LLM's structured response into (verdict, reasons, notes)."""
    verdict = "WARNING"
    found_verdict = False
    reasons: list[str] = []
    notes: list[str] = []
    fraud_reasons: list[str] = []
    fraud_assessment = ""

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("FRAUD_ASSESSMENT:"):
            fraud_assessment = line.split(":", 1)[1].strip().upper()
        elif line.startswith("FRAUD_REASONS:"):
            raw = line.split(":", 1)[1].strip()
            if raw and raw.lower() != "none":
                fraud_reasons = [r.strip() for r in raw.split(",") if r.strip()]
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

    # If LLM is conflicted (suspicious doc but still approved) → flag for review
    if fraud_assessment == "SUSPICIOUS" and verdict == "APPROVE":
        verdict = "WARNING"
        notes.append("Document flagged as suspicious — manual review recommended")

    # Only trigger parse-failure fallback if the VERDICT line was never found
    if not found_verdict:
        notes.append(f"AI response could not be parsed: {response[:200]}")

    return verdict, reasons, notes

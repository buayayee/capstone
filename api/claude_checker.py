"""
claude_checker.py
-----------------
Uses the Anthropic Claude API to judge whether a deferment supporting document
should be APPROVED or REJECTED, based on the actual directive text.

Requires:
  - pip install anthropic
  - Environment variable ANTHROPIC_API_KEY set to your API key
    (get one at https://console.anthropic.com/)

Usage (from check.py with --ai flag):
  check.py --instructions directives/... --folder submissions/ --ai
"""

import os
import re

import anthropic

# Model to use — claude-3-5-haiku is fast and cheap; swap to claude-opus-4 for
# higher accuracy on complex documents.
CLAUDE_MODEL = "claude-3-5-haiku-20241022"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Check env var first, fall back to hardcoded key
        #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
        api_key = os.environ.get("ANTHROPIC_API_KEY", "micdrop")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def check_with_claude(
    document_text: str,
    directive_text: str,
    filename: str = "",
) -> tuple[str, list[str], list[str]]:
    """
    Ask Claude to judge the document against the directive.

    Returns:
        (verdict, reject_reasons, notes)
        verdict is "APPROVE", "REJECT", or "WARNING"
    """
    client = _get_client()

    # Truncate very long texts to stay within token limits
    directive_snippet = directive_text[:6000]
    doc_snippet = document_text[:4000]

    prompt = f"""You are an NS (National Service) deferment application clerk in Singapore.

Your job is to check whether a supporting document submitted for an NS deferment application is VALID or should be REJECTED.

== DIRECTIVE / POLICY (what counts as valid) ==
{directive_snippet}

== SUBMITTED DOCUMENT ==
Filename: {filename}
{doc_snippet}

== YOUR TASK ==
Based on the directive above, decide if this supporting document is sufficient proof for a deferment.

Rules for APPROVAL:
- The document must be relevant to a valid deferment reason (full-time study, overseas employment, medical, marriage, childbirth, etc.)
- It must clearly identify the applicant (name or NRIC or some personal identifier)
- It must have a date issued
- It must appear genuine (issued by a recognisable institution or authority)

Rules for REJECTION:
- Document is clearly irrelevant to NS deferment
- Missing the applicant's identity
- Missing a date
- Obvious signs of tampering or forgery
- Too vague or incomplete to be useful

Respond in EXACTLY this format — no extra text:
VERDICT: APPROVE
REASONS: <leave blank if approving>
NOTES: <brief explanation of what the document is and why approved, or why rejected>

Or if rejecting:
VERDICT: REJECT
REASONS: <comma-separated list of specific reasons>
NOTES: <brief explanation>
"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()
    return _parse_claude_response(response_text)


def _parse_claude_response(response: str) -> tuple[str, list[str], list[str]]:
    """Parse Claude's structured response into (verdict, reasons, notes)."""
    verdict = "WARNING"
    reasons = []
    notes = []

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if "APPROVE" in v:
                verdict = "APPROVE"
            elif "REJECT" in v:
                verdict = "REJECT"
        elif line.startswith("REASONS:"):
            raw = line.split(":", 1)[1].strip()
            if raw:
                reasons = [r.strip() for r in raw.split(",") if r.strip()]
        elif line.startswith("NOTES:"):
            raw = line.split(":", 1)[1].strip()
            if raw:
                notes = [raw]

    if verdict == "WARNING":
        # Claude didn't follow format — fall back gracefully
        notes = [f"Claude response could not be parsed: {response[:200]}"]

    return verdict, reasons, notes

"""
groq_checker.py
---------------
Uses the Groq API to judge whether a deferment supporting document
should be APPROVED or REJECTED, based on the actual directive text.

Groq provides very fast LLM inference (LLaMA, Mixtral, etc.) for free
on the developer tier — no credit card required.

Requires:
  - pip install groq
  - Environment variable GROQ_API_KEY set to your API key
    (get one at https://console.groq.com/)

Usage (from check.py with --ai flag):
  check.py --instructions directives/... --folder submissions/ --ai
"""

import os
import re
#In this scenerio we will be using groq as it is a free api AI key.
from groq import Groq

# Model to use — llama-3.3-70b-versatile is accurate and fast on Groq.
# Other good options: "llama3-8b-8192" (faster/cheaper), "mixtral-8x7b-32768"
#Change model here
MODEL = "llama-3.3-70b-versatile"

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


def check(
    document_text: str,
    directive_text: str,
    filename: str = "",
) -> tuple[str, list[str], list[str]]:
    """
    Ask a Groq-hosted LLM to judge the document against the directive.

    Returns:
        (verdict, reject_reasons, notes)
        verdict is "APPROVE", "REJECT", or "DISQUALIFIED"
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
- It must have a date issued
- It must appear genuine (issued by a recognisable institution or authority)
- NOTE: Applicant names are intentionally redacted for privacy — do NOT reject a document solely because no name appears

Rules for REJECTION:
- Document is clearly irrelevant to NS deferment
- Missing a date entirely
- Obvious signs of tampering or forgery
- Too vague or incomplete to be useful (e.g. blank page)

Respond in EXACTLY this format — no extra text before or after:
VERDICT: APPROVE
REASONS: <leave blank if approving>
NOTES: <brief explanation of what the document is and why approved, or why rejected>

Or if rejecting:
VERDICT: REJECT
REASONS: <comma-separated list of specific reasons>
NOTES: <brief explanation>
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,   # deterministic — same doc should give same verdict
    )

    response_text = response.choices[0].message.content.strip()
    return _parse_response(response_text)


def _parse_response(response: str) -> tuple[str, list[str], list[str]]:
    """Parse the LLM's structured response into (verdict, reasons, notes)."""
    verdict = "DISQUALIFIED"
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

    if verdict == "DISQUALIFIED":
        # LLM didn't follow format — fall back gracefully
        notes = [f"Groq response could not be parsed: {response[:200]}"]

    return verdict, reasons, notes

"""
Agent 3 — Compliance (safetyChecks)

Model: meta-llama/llama-4-scout-17b-16e-instruct (Groq)
Rationale: Compliance review is a rule-matching task with a deterministic
disclaimer fallback. Scout is fast enough and the hard rules ensure 100%
accuracy on the disclaimer check regardless of LLM output.
"""

import json
import re
import os
import time
from typing import Any
from pydantic import ValidationError
from models.schemas import SafetyCheckOutput

REQUIRED_DISCLAIMER = (
    "All claims are subject to policy terms, coverage verification, and investigation. "
    "This is not a determination of coverage."
)

# Deterministic pattern checks for known violation phrases
_GUARANTEE_PATTERNS = [
    r"\byour claim (has been |is |will be )?(approved|accepted|settled|paid)\b",
    r"\byou (are|will be|are now) (fully |)covered\b",
    r"\bwe will (definitely |)pay\b",
    r"\byou will receive \$[\d,]+",
    r"\byour claim is (denied|rejected)\b",
    r"\b(this claim|this loss|this incident) is not covered\b",  # Scoped to claims context only
    r"\bnot eligible for (insurance |claims |this )coverage\b",
]

_MISLEAD_PATTERNS = [
    r"\byou should accept this (offer|amount|settlement)\b",
    r"\bwe (guarantee|promise) (to resolve|payment|settlement)\b",
    r"\bwill be resolved (within|in) \d+ (hours|days) guaranteed\b",
]


def _deterministic_violation_check(reply: str) -> list[str]:
    """Fast regex scan for obvious violations — runs before LLM."""
    violations = []
    r = reply.lower()
    for p in _GUARANTEE_PATTERNS:
        if re.search(p, r):
            violations.append("coverage_guarantee_or_denial")
            break
    for p in _MISLEAD_PATTERNS:
        if re.search(p, r):
            violations.append("misleading_language")
            break
    if REQUIRED_DISCLAIMER.lower() not in r:
        violations.append("missing_disclaimer")
    return list(set(violations))


SYSTEM_PROMPT = f"""\
You are an insurance compliance reviewer. Review the draft reply below and return ONLY a valid JSON object — no markdown, no explanation.

Check for these violation types:

1. "coverage_guarantee_or_denial"
   Flag if the reply says:
   • "Your claim is approved / accepted / settled / paid"
   • "You are covered" or "You will be covered"
   • "We will pay $X" or "You will receive $X"
   • "Your claim is denied" / "This is not covered" / "Not eligible for coverage"

2. "misleading_language"
   Flag if the reply:
   • Pressures the claimant: "You should accept this offer quickly"
   • Makes guaranteed timeline promises: "Resolved within 24 hours guaranteed"
   • Misrepresents policy terms

3. "missing_disclaimer"
   Flag if the reply does NOT contain (case-insensitive):
   "{REQUIRED_DISCLAIMER}"

RULES:
- If no violations: {{"compliance_pass": true, "violations": []}}
- If violations found: {{"compliance_pass": false, "violations": ["violation_type_1", ...]}}
- Only include violation type strings that actually apply.

Return EXACTLY the JSON object and nothing else.
"""


def run_safety_checks(
    reply: str,
    client: Any,
    max_retries: int = 3,
    token_tracker=None,
) -> SafetyCheckOutput:
    """
    Runs Agent 3 compliance check.
    Deterministic regex runs first as hard override.
    LLM is used for nuanced misleading language detection.
    Final result merges both.
    """
    # Run deterministic checks first (guaranteed accuracy on known patterns)
    det_violations = _deterministic_violation_check(reply)

    from api.llm_client import get_models
    model = get_models().safety_model

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Draft reply to review:\n{reply}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )
            latency_ms = (time.time() - t0) * 1000

            if token_tracker and response.usage:
                token_tracker.record(response.usage, model, "safetyChecks", latency_ms)

            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            result = SafetyCheckOutput(**data)

            # Merge deterministic violations with LLM violations (union)
            all_violations = list(set(result.violations) | set(det_violations))
            result.violations = all_violations
            result.compliance_pass = len(all_violations) == 0

            return result

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries - 1:
                # Fall back to pure deterministic result
                violations = det_violations
                return SafetyCheckOutput(
                    compliance_pass=len(violations) == 0,
                    violations=violations,
                )
            time.sleep(0.5 * (attempt + 1))

        except Exception as e:
            raise RuntimeError(f"safetyChecks API error: {e}")

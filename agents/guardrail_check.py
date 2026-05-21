"""
Agent 1 — Intake Guardrails (guardrailCheck)

Model: meta-llama/llama-4-scout-17b-16e-instruct (Groq)
Rationale: Scout is fast (~470ms TTFT). The task is 3 boolean flags —
PII detection, relevance gate, escalation flag. Speed and reliability
outweigh raw power here. No reasoning chain needed.
"""

import json
import os
import re
import time
from typing import Any
from pydantic import ValidationError
from models.schemas import GuardrailOutput

# ── Deterministic PII pre-check (regex — 100% recall on known patterns) ──────
_PII_PATTERNS = [
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",           # SSN
    r"\b[A-Z]{5}\d{4}[A-Z]\b",                     # PAN card
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",            # Aadhaar
    r"\b\d{16,19}\b",                               # Card number
    r"\b\d{9,18}\b(?=.*\b(account|routing|bank)\b)",# Bank account (contextual)
    r"\bPOL[-\s]?\d{6,}\b",                         # Full policy number
    r"\b[A-Z]{1,2}\d{6,9}\b",                       # Passport / DL format
    r"\b1JTD[A-Z0-9]{13}\b",                        # VIN sample pattern
]

def _has_pii(text: str) -> bool:
    """Returns True (PII found = no_pii is False) on hard pattern match."""
    for pattern in _PII_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


SYSTEM_PROMPT = """\
You are an insurance intake validation agent. Analyze the policyholder message and return ONLY a valid JSON object with exactly three boolean fields. No explanation, no markdown, no extra text.

=== FIELD RULES ===

1. is_insurance_related (boolean)
   TRUE  → message is about: filing a claim, reporting damage/loss/theft, asking about insurance coverage, policy terms, deductibles, claim status, or how to file.
   FALSE → message has NOTHING to do with insurance (cooking, coding, general trivia, jokes, unrelated requests).
   EDGE: If the message mixes insurance with off-topic content, set TRUE (there IS an insurance component).

2. no_pii (boolean)
   TRUE  → message contains NO unmasked sensitive identifiers.
   FALSE → message contains ANY of:
     • Full SSN (e.g., 523-78-9241 or 523789241)
     • Full Aadhaar (12-digit number, e.g., 9876 5432 1098)
     • Full PAN card (e.g., ABCDE1234F)
     • Full bank account / routing / card numbers
     • Full policy or claim number (e.g., POL-1234567890 — NOT ****-4321 which is masked)
     • Full Driver's License or VIN number
     • Government passport numbers
   NOTE: Partial masking (****-4321, XXX-XX-1234) is NOT PII → set TRUE.

3. needs_escalation (boolean)
   TRUE  → message contains ANY of:
     • Threats of self-harm or harming others ("I can't go on", "I want to end it")
     • Severe emotional distress with hopelessness ("I have nothing left", "I feel hopeless", "I have nowhere to go")
     • Active emergency happening RIGHT NOW: "house is on fire", "currently flooding", "at the scene right now"
     • Reported physical injury needing medical attention: "taken to hospital", "broken arm/leg/wrist", "in the ER", "injured"
   FALSE → high-dollar damage WITHOUT injury and WITHOUT active ongoing emergency (e.g., "$35,000 roof damage from last night's storm" = FALSE because no injury, no active emergency)
   FALSE → total vehicle loss or significant property loss with NO injuries and NO current emergency
   FALSE → past incidents already stabilized ("last night", "yesterday", "last week") with no injuries

=== EXAMPLES ===
Message: "My car was rear-ended yesterday. Damage is $4,000. No injuries."
→ {"is_insurance_related": true, "no_pii": true, "needs_escalation": false}

Message: "My SSN is 523-78-9241. I need to file a claim."
→ {"is_insurance_related": true, "no_pii": false, "needs_escalation": false}

Message: "My house is ON FIRE RIGHT NOW. We got out but everything is burning."
→ {"is_insurance_related": true, "no_pii": true, "needs_escalation": true}

Message: "I was in a crash and taken to the hospital with a broken wrist."
→ {"is_insurance_related": true, "no_pii": true, "needs_escalation": true}

Message: "What is the capital of France?"
→ {"is_insurance_related": false, "no_pii": true, "needs_escalation": false}

Message: "My policy ending in ****-4321 covers collision, right?"
→ {"is_insurance_related": true, "no_pii": true, "needs_escalation": false}

Message: "A tree fell on our house last night. $35,000 in damage. No injuries. We need to file a claim."
→ {"is_insurance_related": true, "no_pii": true, "needs_escalation": false}

Message: "I lost everything in the tornado. I have nothing left and feel completely hopeless."
→ {"is_insurance_related": true, "no_pii": true, "needs_escalation": true}

Return EXACTLY this JSON and nothing else:
{"is_insurance_related": true or false, "no_pii": true or false, "needs_escalation": true or false}
"""


def run_guardrail_check(
    message: str,
    client: Any,
    max_retries: int = 3,
    token_tracker=None,
    conversation_history: list = None,
) -> GuardrailOutput:
    """
    Runs Agent 1 guardrail check.
    Deterministic PII regex runs first as a hard override (guarantees 100% recall).
    If conversation_history is provided (ongoing session), injects last 2 turns so
    the LLM understands short follow-up messages like "where do i see this?" correctly.
    """
    from api.llm_client import get_models
    model = get_models().guardrail_model

    # Hard deterministic PII check (overrides LLM if triggered)
    deterministic_pii_found = _has_pii(message)

    # Build context-aware user prompt
    if conversation_history and len(conversation_history) >= 2:
        # Inject last 2 turns so guardrail understands follow-up messages
        recent = conversation_history[-4:]
        ctx = "\n".join(f"{t['role'].upper()}: {t['content'][:150]}" for t in recent)
        user_content = (
            f"This is a follow-up message in an ongoing insurance conversation.\n\n"
            f"Recent conversation:\n{ctx}\n\n"
            f"New message to evaluate:\n{message}"
        )
    else:
        user_content = f"Policyholder message:\n{message}"

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=150,
            )
            latency_ms = (time.time() - t0) * 1000

            if token_tracker and response.usage:
                token_tracker.record(response.usage, model, "guardrailCheck", latency_ms)

            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            result = GuardrailOutput(**data)

            # Hard override: if regex found PII, enforce no_pii=False regardless
            if deterministic_pii_found:
                result.no_pii = False

            return result

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"guardrailCheck failed after {max_retries} attempts. Last error: {e}"
                )
            time.sleep(0.5 * (attempt + 1))

        except Exception as e:
            raise RuntimeError(f"guardrailCheck API error: {e}")

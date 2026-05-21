"""
Agent 2.5 — Clarifier

Runs AFTER Agent 1 (guardrail) and BEFORE Agent 2 (claim parser).
Checks if the message is ambiguous and needs a follow-up question.

If the message clearly states what the customer needs → passes through (no question).
If intent or key details are missing → returns a targeted clarifying question.

This prevents Agent 2 from guessing when a human agent would ask instead.
"""

import json
import time
from typing import Any
from pydantic import ValidationError
from models.schemas import ClarifierOutput


SYSTEM_PROMPT = """\
You are an insurance intake assistant. Your job is to decide if a customer's message
provides enough information to process, or if a clarifying question is needed first.

Return ONLY a JSON object with these fields:
  needs_clarification: true if you need more info, false if message is clear enough
  clarification_question: the question to ask (only if needs_clarification=true, else null)
  confidence: "high" / "medium" / "low" — how confident you are about the customer's intent

=== WHEN TO ASK FOR CLARIFICATION ===
Ask when the message is truly ambiguous — you cannot determine the intent or what the customer needs:
- "I need help" (no detail at all)
- "I need to file something" (unclear what type)
- "I have a problem with my car" (unclear if claim or coverage question)

Do NOT ask for clarification if:
- The customer described an incident (accident, damage, theft) → it's a claim
- The customer asked a specific question about coverage → it's a coverage inquiry
- The customer mentioned a loss amount, injury, or police report → clear claim
- There's enough info to route the inquiry even if some details are missing

=== CLARIFICATION QUESTION STYLE ===
Be warm and specific. Offer clear options:
- "Are you filing a claim for recent damage, or do you have a question about your coverage?"
- "Is this about a vehicle (car/truck) or your home/property?"
- "Did this incident just happen, or are you following up on an existing claim?"

Return EXACTLY the JSON and nothing else.
"""


def run_clarifier(
    message: str,
    client: Any,
    conversation_history: list = None,
    max_retries: int = 2,
    token_tracker=None,
) -> ClarifierOutput:
    """
    Quick check: does this message need a follow-up question?
    Uses conversation history to avoid asking for info already provided.
    """
    from api.llm_client import get_models
    model = get_models().guardrail_model  # use fast model — this is a lightweight check

    # Build context from conversation history so we don't re-ask already-answered questions
    context = ""
    if conversation_history:
        recent = conversation_history[-4:]  # last 2 turns
        context = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in recent)
        context = f"\n\nRecent conversation:\n{context}"

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Customer message: {message}{context}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=150,
            )
            latency_ms = (time.time() - t0) * 1000
            if token_tracker and response.usage:
                token_tracker.record(response.usage, model, "clarifier", latency_ms)

            data = json.loads(response.choices[0].message.content.strip())
            return ClarifierOutput(**data)

        except Exception:
            if attempt == max_retries - 1:
                # On failure, don't block — just pass through
                return ClarifierOutput(needs_clarification=False)
            time.sleep(0.3)

    return ClarifierOutput(needs_clarification=False)

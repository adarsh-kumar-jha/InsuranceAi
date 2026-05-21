"""
Agent 4 — Fraud Detector

Runs AFTER Agent 3 (safety checks). Analyzes the claim for fraud risk signals.

Scoring dimensions:
  - Claim recency vs policy start date
  - Loss amount plausibility vs incident description
  - Claim frequency from same session
  - Inconsistencies between messages in the conversation
  - Suspicious language patterns

Returns:
  fraud_risk_score: Low / Medium / High
  fraud_flags:      list of specific red flags detected
  recommended_action: auto_approve / manual_review / flag_for_investigation
"""

import json
import re
import time
from typing import Any, Optional
from models.schemas import FraudCheckOutput as FraudDetectorOutput


SYSTEM_PROMPT = """\
You are a fraud detection specialist for an insurance company. Analyze the claim details
and conversation for indicators of fraud or abuse. Be objective — most claims are legitimate.

Return ONLY a JSON object with:
  fraud_risk_score: "Low" / "Medium" / "High"
  fraud_flags: list of specific red flags (empty list if none)
  recommended_action: one of "auto_approve" / "manual_review" / "flag_for_investigation"
  fraud_justification: 1-2 sentence explanation of your assessment

=== RED FLAGS TO CHECK ===
HIGH risk (flag for investigation if 2+):
- Claim filed within 30 days of policy start AND high loss amount
- Loss amount dramatically higher than typical for the described incident
- Customer unable or unwilling to provide basic details
- Multiple claims in short timeframe from same customer
- Inconsistent descriptions of the same incident across messages
- Unusually specific damage amounts without explanation

MEDIUM risk (manual review if any):
- Loss amount higher than typical (but possible)
- Vague descriptions with high dollar amounts
- Policy just started (60–90 days)
- Incident description sounds copied/template-like

LOW risk (normal indicators):
- Clear, consistent description
- Reasonable loss amounts for the incident type
- Customer provides specific details naturally
- First claim on the policy

=== TYPICAL LOSS RANGES ===
Auto fender bender: $500–$3,000
Auto total loss: $5,000–$30,000
Home water damage: $2,000–$15,000
Home fire damage: $10,000–$200,000
Theft (auto contents): $500–$5,000
Home theft: $1,000–$20,000

Be fair: most people are honest. Only flag genuine concerns.
"""


def run_fraud_detector(
    message: str,
    claim_data: dict,
    conversation_history: list = None,
    client: Any = None,
    session_claims_count: int = 0,
    token_tracker=None,
) -> FraudDetectorOutput:
    """
    Analyze the claim for fraud risk.
    
    Args:
        message: The current customer message
        claim_data: Output from claim parser (intent, loss_amount, priority, description)
        conversation_history: Previous turns in the session
        client: LLM client
        session_claims_count: Number of claims already filed in this session
        token_tracker: Token usage tracker
    """
    from api.llm_client import get_models
    model = get_models().guardrail_model  # fast model is fine for this

    # Build context string for LLM
    intent = claim_data.get("intent", {})
    if hasattr(intent, "value"):
        intent = intent.value
    elif isinstance(intent, dict):
        intent = intent.get("value", str(intent))

    loss = claim_data.get("estimated_loss_amount") or claim_data.get("loss_amount")
    priority = claim_data.get("priority", {})
    if hasattr(priority, "value"):
        priority = priority.value
    elif isinstance(priority, dict):
        priority = priority.get("value", str(priority))

    history_summary = ""
    if conversation_history and len(conversation_history) > 2:
        turns = conversation_history[-6:]
        history_summary = "\n".join(
            f"{t['role'].upper()}: {t['content'][:200]}" for t in turns
        )

    context = f"""
Claim type: {intent}
Estimated loss: {'${:,.0f}'.format(float(loss)) if loss else 'not stated'}
Priority assigned: {priority}
Claims in this session: {session_claims_count}
Customer message: {message}

Conversation context:
{history_summary if history_summary else 'No prior conversation.'}
    """.strip()

    # Rule-based pre-screening (fast, no LLM needed for obvious cases)
    rule_flags = []
    if session_claims_count >= 3:
        rule_flags.append(f"Multiple claims in single session ({session_claims_count} total)")
    if loss and float(loss) > 100000:
        rule_flags.append(f"Unusually high loss amount: ${float(loss):,.0f}")

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )
        latency_ms = (time.time() - t0) * 1000
        if token_tracker and response.usage:
            token_tracker.record(response.usage, model, "fraud_detector", latency_ms)

        data = json.loads(response.choices[0].message.content.strip())
        result = FraudDetectorOutput(**data)

        # Merge rule-based flags with LLM flags
        combined_flags = list(set(result.fraud_flags + rule_flags))
        result.fraud_flags = combined_flags

        # Escalate risk if rule-based found critical flags
        if rule_flags and result.fraud_risk_score == "Low":
            result.fraud_risk_score = "Medium"
            result.recommended_action = "manual_review"

        return result

    except Exception as e:
        # Fallback: return Low risk so we don't block legitimate claims
        return FraudDetectorOutput(
            fraud_risk_score="Low",
            fraud_flags=rule_flags,
            recommended_action="auto_approve",
            fraud_justification="Fraud analysis unavailable — proceeding with standard processing.",
        )

"""
Post-Processor — runs ONE LLM call after Agent 2 to generate three things:

  1. follow_up_questions  → 2-3 clickable questions the customer is likely to ask next
  2. evidence_checklist   → personalised list of documents/photos to gather
  3. settlement_range     → realistic payout estimate (only for actual claims with loss amount)

Using a single call for all three avoids extra API round-trips.
"""

import json
import time
from typing import Any, Optional
from models.schemas import PostProcessorOutput

SYSTEM_PROMPT = """\
You are an insurance claims assistant. Given the claim details below, return ONLY a JSON object.

Fields to return:
  follow_up_questions: array of exactly 2-3 short questions the customer is likely to ask next.
                       Make them specific to the claim type and situation.
  evidence_checklist:  array of items the customer needs to gather for their claim.
                       Be specific to the claim type (auto vs home vs coverage inquiry).
                       Empty array for coverage inquiries.
  settlement_range:    object with "min", "max", "note" fields — realistic payout estimate.
                       Only include if this is a claim (not coverage inquiry) AND loss_amount > 0.
                       Account for the deductible. If no policy data, estimate from typical ranges.
                       Set to null if not applicable.

=== FOLLOW-UP QUESTION EXAMPLES ===
Auto claim: "When will an adjuster contact me?", "Can I get a rental car?", "How long will repairs take?"
Home claim: "Will I need to vacate during repairs?", "Are temporary living expenses covered?", "What's my deductible?"
Coverage inquiry: "Does this apply to my current policy?", "How do I add this coverage?", "What's the premium increase?"

=== EVIDENCE CHECKLIST EXAMPLES ===
Auto claim: Police report (if applicable), Photos of all damage, Repair estimate from licensed shop,
            Other driver's insurance info (if accident), Witness contact information, Medical records (if injured)
Home claim: Photos/video of all damage, Contractor repair estimates (2-3 quotes), 
            Receipts for damaged items, Utility shutoff confirmation, Temporary repair receipts
Coverage inquiry: [] (no evidence needed)

=== SETTLEMENT RANGE EXAMPLES ===
Auto claim, loss $5,000, deductible $500: {"min": 3500, "max": 4500, "note": "After $500 deductible, subject to policy limits and investigation"}
Home claim, loss $25,000, deductible $2,000: {"min": 18000, "max": 23000, "note": "After deductible, range reflects typical adjustment factors"}

Return ONLY the JSON object, no other text.
"""


def run_post_processor(
    intent: str,
    priority: str,
    loss_amount: Optional[float],
    summary_response: str,
    client: Any,
    policy_deductible: Optional[float] = None,
    token_tracker=None,
    already_asked: set = None,
) -> PostProcessorOutput:
    """
    Single LLM call to generate follow-up questions, evidence checklist,
    and settlement estimate after the claim parser runs.
    """
    from api.llm_client import get_models
    model = get_models().guardrail_model  # fast model — lightweight output

    loss_str = f"${loss_amount:,.0f}" if loss_amount else "not stated"
    deductible_str = f"${policy_deductible:,.0f}" if policy_deductible else "unknown"

    user_msg = (
        f"Claim type: {intent}\n"
        f"Priority: {priority}\n"
        f"Estimated loss: {loss_str}\n"
        f"Policy deductible: {deductible_str}\n"
        f"Agent reply given: {summary_response[:200]}"
    )

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=400,
        )
        latency_ms = (time.time() - t0) * 1000
        if token_tracker and response.usage:
            token_tracker.record(response.usage, model, "postProcessor", latency_ms)

        data = json.loads(response.choices[0].message.content.strip())
        raw_questions = data.get("follow_up_questions", [])

        # Filter out questions the user already asked in this session
        if already_asked:
            raw_questions = [
                q for q in raw_questions
                if q.strip().lower() not in already_asked
            ]

        return PostProcessorOutput(
            follow_up_questions=raw_questions[:3],
            evidence_checklist=data.get("evidence_checklist", []),
            settlement_range=data.get("settlement_range"),
        )

    except Exception:
        result = _fallback(intent)
        if already_asked:
            result.follow_up_questions = [
                q for q in result.follow_up_questions
                if q.strip().lower() not in already_asked
            ]
        return result


def _fallback(intent: str) -> PostProcessorOutput:
    """Template-based fallback when LLM is unavailable."""
    fallbacks = {
        "Auto claim": PostProcessorOutput(
            follow_up_questions=[
                "When will an adjuster contact me?",
                "Can I get a rental car while mine is being repaired?",
                "How long does the claims process take?",
            ],
            evidence_checklist=[
                "Police report (if applicable)",
                "Photos of all vehicle damage",
                "Repair estimate from a licensed shop",
                "Other driver's insurance information",
            ],
            settlement_range=None,
        ),
        "Home claim": PostProcessorOutput(
            follow_up_questions=[
                "Will temporary living expenses be covered?",
                "How many repair quotes do I need?",
                "What is my deductible for this claim?",
            ],
            evidence_checklist=[
                "Photos and video of all damage",
                "At least 2 contractor repair estimates",
                "List of damaged personal property with values",
                "Receipts for any emergency/temporary repairs",
            ],
            settlement_range=None,
        ),
        "Coverage inquiry": PostProcessorOutput(
            follow_up_questions=[
                "Does this apply to my existing policy?",
                "How do I add or change my coverage?",
                "What would the premium change be?",
            ],
            evidence_checklist=[],
            settlement_range=None,
        ),
    }
    return fallbacks.get(intent, PostProcessorOutput(
        follow_up_questions=["How long does this process take?", "Who should I contact next?"],
        evidence_checklist=[],
        settlement_range=None,
    ))

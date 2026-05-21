"""
Agent 0.5 — Customer Sentiment Detector

Runs BEFORE Agent 2 (claim parser).
Detects the customer's emotional state and injects a tone instruction so
Agent 2 responds with the right level of empathy.

Sentiment levels:
  calm       → standard professional tone
  frustrated → extra acknowledgment, no bureaucratic language
  distressed → maximum empathy, prioritise emotional support first
"""

import re
import time
from typing import Any
from models.schemas import SentimentOutput

# ── Fast rule-based pre-screen (no LLM needed for obvious cases) ──────────────

_DISTRESSED_PATTERNS = re.compile(
    r"\b(devastated|destroyed|emergency|horrified|shaking|crying|don.t know what to do"
    r"|everything is gone|total loss|hospital|injured|hurt|death|died|fire|flood|collapse)\b",
    re.IGNORECASE,
)

_FRUSTRATED_PATTERNS = re.compile(
    r"\b(ridiculous|terrible|unacceptable|waiting for weeks|nobody.s helping|useless"
    r"|frustrated|angry|furious|pathetic|worst|been trying for|still no response"
    r"|keep getting ignored|no one is answering|already called|why is it taking)\b",
    re.IGNORECASE,
)

_TONE_INSTRUCTIONS = {
    "distressed": (
        "The customer is clearly in distress. Lead with deep empathy and emotional acknowledgment "
        "BEFORE any process information. Use warm, human language. Avoid all jargon. "
        "Keep sentences short. Reassure them that they are not alone."
    ),
    "frustrated": (
        "The customer is frustrated. Directly acknowledge their frustration in the opening sentence. "
        "Avoid corporate/bureaucratic language. Be concise and action-oriented. "
        "Tell them exactly what happens next."
    ),
    "calm": (
        "The customer is calm. Use a professional, warm, and concise tone. "
        "Focus on being helpful and informative."
    ),
}


def run_sentiment_agent(
    message: str,
    client: Any = None,
    token_tracker=None,
) -> SentimentOutput:
    """
    Detect customer sentiment. Uses rule-based patterns first (fast, no tokens).
    Falls back to LLM for ambiguous cases.
    """
    # Fast rule-based detection
    if _DISTRESSED_PATTERNS.search(message):
        sentiment = "distressed"
    elif _FRUSTRATED_PATTERNS.search(message):
        sentiment = "frustrated"
    else:
        # Use LLM for borderline cases — only if client is provided
        sentiment = _llm_sentiment(message, client, token_tracker) if client else "calm"

    tone = _TONE_INSTRUCTIONS[sentiment]
    return SentimentOutput(
        sentiment=sentiment,
        tone_instruction=tone,
        urgency_boost=sentiment == "distressed",
    )


def _llm_sentiment(message: str, client: Any, token_tracker) -> str:
    """Lightweight LLM call for ambiguous sentiment."""
    try:
        from api.llm_client import get_models
        model = get_models().guardrail_model

        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the customer's emotional tone in one word: "
                        "calm, frustrated, or distressed. "
                        "Reply with ONLY that one word, nothing else."
                    ),
                },
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        latency_ms = (time.time() - t0) * 1000
        if token_tracker and response.usage:
            token_tracker.record(response.usage, model, "sentimentAgent", latency_ms)

        raw = response.choices[0].message.content.strip().lower()
        if raw in ("calm", "frustrated", "distressed"):
            return raw
        return "calm"
    except Exception:
        return "calm"

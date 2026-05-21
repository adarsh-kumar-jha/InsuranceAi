"""
Input validation for the Insurance AI pipeline.
Runs before any LLM call to reject garbage, spam, and out-of-scope input.
"""

import re
import math
from typing import Optional
from config import cfg

MIN_LENGTH           = cfg.validation.min_length
MAX_LENGTH           = cfg.validation.max_length
MIN_WORDS            = cfg.validation.min_words
MIN_ALPHA_RATIO      = cfg.validation.min_alpha_ratio
MAX_REPEATED_CHAR_RATIO = 0.35
MAX_AVG_WORD_LENGTH  = 25


class ValidationError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution. Low entropy = repetitive/spam."""
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    total = len(text)
    return -sum((f / total) * math.log2(f / total) for f in freq.values())


def _most_repeated_char_ratio(text: str) -> float:
    """Ratio of the most-repeated character to total length."""
    if not text:
        return 0.0
    freq = {}
    for c in text.lower():
        freq[c] = freq.get(c, 0) + 1
    return max(freq.values()) / len(text)


def validate_input(message: str) -> tuple[bool, Optional[str]]:
    """
    Validates user input before sending to the LLM pipeline.

    Returns:
        (True, None) if valid
        (False, error_message) if invalid
    """
    if not message or not message.strip():
        return False, "Message cannot be empty."

    stripped = message.strip()

    # 1. Length checks
    if len(stripped) < MIN_LENGTH:
        return False, f"Message is too short. Please provide at least {MIN_LENGTH} characters describing your inquiry."

    if len(stripped) > MAX_LENGTH:
        return False, f"Message is too long ({len(stripped)} chars). Please keep it under {MAX_LENGTH} characters."

    # 2. Minimum word count
    words = stripped.split()
    if len(words) < MIN_WORDS:
        return False, "Please provide a more detailed description with at least 3 words."

    # 3. All special characters / numbers only
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars / len(stripped) < MIN_ALPHA_RATIO:
        return False, "Your message appears to contain mostly numbers or special characters. Please describe your insurance inquiry in plain text."

    # 4. Repeated character spam detection (e.g., "aaaaaaaaaa", "!!!!!!!")
    if _most_repeated_char_ratio(stripped) > MAX_REPEATED_CHAR_RATIO:
        return False, "Your message appears to contain repeated characters. Please describe your inquiry clearly."

    # 5. Low entropy (keyboard mashing / random characters)
    if _char_entropy(stripped) < cfg.validation.min_entropy and len(stripped) > 30:
        return False, "Your message appears to be random characters. Please describe your insurance inquiry."

    # 6. Excessive word length (no real words)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > MAX_AVG_WORD_LENGTH:
        return False, "Your message doesn't appear to contain recognizable words. Please describe your inquiry in plain language."

    # 7. Repeated word spam (e.g., "claim claim claim claim claim")
    unique_words = set(w.lower() for w in words)
    if len(words) >= 5 and len(unique_words) / len(words) < 0.25:
        return False, "Your message appears to contain repeated words. Please describe your insurance situation clearly."

    # 8. SQL/script injection patterns
    injection_patterns = [
        r"(select|insert|update|delete|drop|truncate)\s+\w+",
        r"<script[\s\S]*?>",
        r"javascript\s*:",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False, "Your message contains invalid content. Please describe your insurance inquiry in plain text."

    return True, None


def sanitize_input(message: str) -> str:
    """
    Sanitizes the message before processing:
    - Strips leading/trailing whitespace
    - Collapses multiple spaces
    - Removes control characters
    - Caps to MAX_LENGTH
    """
    cleaned = message.strip()
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    cleaned = re.sub(r" {3,}", "  ", cleaned)
    return cleaned[:MAX_LENGTH]

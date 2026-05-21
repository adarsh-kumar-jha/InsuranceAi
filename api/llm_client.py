"""
LLM Client Factory — delegates entirely to config.cfg.

All provider settings, API keys, base URLs, and model names live in .env
and are resolved once at startup by config.py. This module just exposes
the same get_client() / get_models() interface that agents rely on.
"""

from openai import OpenAI
from config import cfg


def get_client() -> OpenAI:
    """Returns an OpenAI-compatible client for the configured provider."""
    return OpenAI(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url)


def get_models():
    """Returns the per-agent model config from cfg.llm."""
    return cfg.llm

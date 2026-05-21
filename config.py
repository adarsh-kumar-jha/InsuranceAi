"""
config.py — Single source of truth for all project settings.

All values read from environment variables (via .env file).
Edit .env to change any setting — this file never needs to be touched.

Usage anywhere in the codebase:
    from config import cfg
    print(cfg.server_port)         # 8000
    print(cfg.llm.provider)        # "groq"
    print(cfg.llm.parser_model)    # "llama-3.3-70b-versatile"
    print(cfg.rag.top_k)           # 3
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


# ─── LLM / Provider config ────────────────────────────────────────────────────

_PROVIDER_DEFAULTS = {
    "groq": {
        "api_key_env":       "GROQ_API_KEY",
        "base_url":          "https://api.groq.com/openai/v1",
        "guardrail_model":   "meta-llama/llama-4-scout-17b-16e-instruct",
        "parser_model":      "llama-3.3-70b-versatile",
        "parser_fallback":   "meta-llama/llama-4-scout-17b-16e-instruct",
        "safety_model":      "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "openai": {
        "api_key_env":       "OPENAI_API_KEY",
        "base_url":          "https://api.openai.com/v1",
        "guardrail_model":   "gpt-4o-mini",
        "parser_model":      "gpt-4o-mini",
        "parser_fallback":   "gpt-4o-mini",
        "safety_model":      "gpt-4o-mini",
    },
    "together": {
        "api_key_env":       "TOGETHER_API_KEY",
        "base_url":          "https://api.together.xyz/v1",
        "guardrail_model":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "parser_model":      "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "parser_fallback":   "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "safety_model":      "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "fireworks": {
        "api_key_env":       "FIREWORKS_API_KEY",
        "base_url":          "https://api.fireworks.ai/inference/v1",
        "guardrail_model":   "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "parser_model":      "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "parser_fallback":   "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "safety_model":      "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    "ollama": {
        "api_key_env":       None,
        "base_url":          "http://localhost:11434/v1",
        "guardrail_model":   "llama3.3",
        "parser_model":      "llama3.3",
        "parser_fallback":   "llama3.2",
        "safety_model":      "llama3.3",
    },
    "custom": {
        "api_key_env":       "LLM_API_KEY",
        "base_url":          None,   # read from LLM_BASE_URL
        "guardrail_model":   None,   # must set GUARDRAIL_MODEL
        "parser_model":      None,   # must set PARSER_MODEL
        "parser_fallback":   None,
        "safety_model":      None,   # must set SAFETY_MODEL
    },
}


@dataclass
class LLMConfig:
    provider:              str
    api_key:               str
    base_url:              str
    guardrail_model:       str
    parser_model:          str
    parser_fallback_model: str
    safety_model:          str
    max_retries:           int
    temperature:           float

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = _env("LLM_PROVIDER", "groq").lower()
        if provider not in _PROVIDER_DEFAULTS:
            supported = ", ".join(_PROVIDER_DEFAULTS.keys())
            raise ValueError(f"Unknown LLM_PROVIDER='{provider}'. Supported: {supported}")

        defaults = _PROVIDER_DEFAULTS[provider]

        # Resolve base_url
        if provider == "custom":
            base_url = _env("LLM_BASE_URL")
            if not base_url:
                raise EnvironmentError("LLM_PROVIDER=custom requires LLM_BASE_URL in .env")
        else:
            base_url = defaults["base_url"]

        # Resolve API key
        key_env = defaults["api_key_env"]
        if key_env:
            api_key = _env(key_env)
            if not api_key:
                raise EnvironmentError(
                    f"Provider '{provider}' requires {key_env} to be set in .env"
                )
        else:
            api_key = "local"   # Ollama / local endpoints don't need a key

        def _model(env_var: str, default_key: str) -> str:
            override = _env(env_var)
            if override:
                return override
            val = defaults.get(default_key)
            if not val:
                raise EnvironmentError(
                    f"LLM_PROVIDER=custom requires {env_var} to be set in .env"
                )
            return val

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            guardrail_model       = _model("GUARDRAIL_MODEL",        "guardrail_model"),
            parser_model          = _model("PARSER_MODEL",           "parser_model"),
            parser_fallback_model = _model("PARSER_FALLBACK_MODEL",  "parser_fallback") or _model("PARSER_MODEL", "parser_model"),
            safety_model          = _model("SAFETY_MODEL",           "safety_model"),
            max_retries           = _env_int("LLM_MAX_RETRIES", 3),
            temperature           = _env_float("LLM_TEMPERATURE", 0.0),
        )


# ─── Server config ────────────────────────────────────────────────────────────

@dataclass
class ServerConfig:
    host:       str
    port:       int
    reload:     bool
    log_level:  str

    @classmethod
    def from_env(cls) -> "ServerConfig":
        # PORT is set by Render/Railway/Fly; SERVER_PORT is for local dev
        port = _env_int("PORT", 0) or _env_int("SERVER_PORT", 8000)
        return cls(
            host      = _env("SERVER_HOST", "0.0.0.0"),
            port      = port,
            reload    = _env_bool("SERVER_RELOAD", True),
            log_level = _env("SERVER_LOG_LEVEL", "info"),
        )


# ─── RAG config ───────────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    top_k:     int
    alpha:     float    # 0 = pure TF-IDF, 1 = pure BM25, 0.5 = equal blend
    min_score: float
    enabled:   bool

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            top_k     = _env_int("RAG_TOP_K", 3),
            alpha     = _env_float("RAG_ALPHA", 0.5),
            min_score = _env_float("RAG_MIN_SCORE", 0.01),
            enabled   = _env_bool("RAG_ENABLED", True),
        )


# ─── Input validation config ──────────────────────────────────────────────────

@dataclass
class ValidationConfig:
    min_length:       int
    max_length:       int
    min_words:        int
    min_alpha_ratio:  float
    min_entropy:      float

    @classmethod
    def from_env(cls) -> "ValidationConfig":
        return cls(
            min_length      = _env_int("VALIDATION_MIN_LENGTH", 10),
            max_length      = _env_int("VALIDATION_MAX_LENGTH", 2000),
            min_words       = _env_int("VALIDATION_MIN_WORDS", 3),
            min_alpha_ratio = _env_float("VALIDATION_MIN_ALPHA_RATIO", 0.3),
            min_entropy     = _env_float("VALIDATION_MIN_ENTROPY", 2.5),
        )


# ─── Root config object ───────────────────────────────────────────────────────

@dataclass
class ABConfig:
    variant:          str    # "A" or "B"
    enabled:          bool

    @classmethod
    def from_env(cls) -> "ABConfig":
        return cls(
            variant = _env("PROMPT_VARIANT", "A").upper(),
            enabled = _env_bool("AB_TESTING_ENABLED", False),
        )


@dataclass
class Config:
    llm:        LLMConfig
    server:     ServerConfig
    rag:        RAGConfig
    validation: ValidationConfig
    ab:         ABConfig

    @classmethod
    def load(cls) -> "Config":
        return cls(
            llm        = LLMConfig.from_env(),
            server     = ServerConfig.from_env(),
            rag        = RAGConfig.from_env(),
            validation = ValidationConfig.from_env(),
            ab         = ABConfig.from_env(),
        )

    def summary(self) -> str:
        return (
            f"Provider : {self.llm.provider} ({self.llm.base_url})\n"
            f"Models   : guardrail={self.llm.guardrail_model}\n"
            f"           parser={self.llm.parser_model}\n"
            f"           safety={self.llm.safety_model}\n"
            f"Server   : http://{self.server.host}:{self.server.port}\n"
            f"RAG      : enabled={self.rag.enabled}, top_k={self.rag.top_k}, alpha={self.rag.alpha}\n"
        )


# ─── Singleton — import this everywhere ──────────────────────────────────────
cfg = Config.load()


if __name__ == "__main__":
    print(cfg.summary())

"""
Global in-memory token tracker.
Records usage from every Groq API call across all agents per session.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


@dataclass
class AgentUsage:
    name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    total_latency_ms: float = 0.0


class TokenTracker:
    def __init__(self):
        self.reset()

    def record(
        self,
        usage,
        model: str,
        agent_name: str,
        latency_ms: float = 0.0,
    ):
        if usage is None:
            return

        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        tt = getattr(usage, "total_tokens", pt + ct) or (pt + ct)

        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        self.total_tokens += tt
        self.total_calls += 1

        # Per-model breakdown
        if model not in self.by_model:
            self.by_model[model] = ModelUsage()
        mu = self.by_model[model]
        mu.prompt_tokens += pt
        mu.completion_tokens += ct
        mu.total_tokens += tt
        mu.calls += 1

        # Per-agent breakdown
        if agent_name not in self.by_agent:
            self.by_agent[agent_name] = AgentUsage(name=agent_name)
        au = self.by_agent[agent_name]
        au.prompt_tokens += pt
        au.completion_tokens += ct
        au.total_tokens += tt
        au.calls += 1
        au.total_latency_ms += latency_ms

        # Rolling call log (last 50)
        self.call_log.append({
            "timestamp": time.time(),
            "agent": agent_name,
            "model": model,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "latency_ms": round(latency_ms, 1),
        })
        if len(self.call_log) > 50:
            self.call_log.pop(0)

    def reset(self):
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_calls: int = 0
        self.by_model: dict[str, ModelUsage] = {}
        self.by_agent: dict[str, AgentUsage] = {}
        self.call_log: list[dict] = []
        self.session_start: float = time.time()

    def get_stats(self) -> dict:
        avg_tokens_per_call = (
            round(self.total_tokens / self.total_calls, 1)
            if self.total_calls > 0 else 0
        )
        session_duration_s = round(time.time() - self.session_start, 1)

        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "avg_tokens_per_call": avg_tokens_per_call,
            "session_duration_s": session_duration_s,
            "by_model": {
                model: {
                    "prompt_tokens": mu.prompt_tokens,
                    "completion_tokens": mu.completion_tokens,
                    "total_tokens": mu.total_tokens,
                    "calls": mu.calls,
                }
                for model, mu in self.by_model.items()
            },
            "by_agent": {
                name: {
                    "prompt_tokens": au.prompt_tokens,
                    "completion_tokens": au.completion_tokens,
                    "total_tokens": au.total_tokens,
                    "calls": au.calls,
                    "avg_latency_ms": round(au.total_latency_ms / au.calls, 1) if au.calls > 0 else 0,
                }
                for name, au in self.by_agent.items()
            },
            "recent_calls": self.call_log[-10:],
        }


# Global singleton — shared across all requests in the server process
token_tracker = TokenTracker()

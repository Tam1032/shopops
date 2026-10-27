"""Model backends behind one interface.

`OpenAICompatLLM` covers every server that speaks the OpenAI chat-completions
tool-calling format: OpenAI, a self-hosted vLLM (`--enable-auto-tool-choice`),
Ollama, and Anthropic's OpenAI-compatible endpoint.
"""

import asyncio
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, exactly as the model produced it


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class LLM(Protocol):
    name: str

    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, max_retries: int = 3, timeout: float = 60, http_client=None,
                 rpm: float = 0):
        from openai import AsyncOpenAI

        self.name = model
        self.temperature = temperature
        # Client-side requests-per-minute cap (0 = off), shared by every concurrent run on this client.
        self.min_interval = 60 / rpm if rpm else 0.0
        self._next_slot = 0.0
        self._slot_lock = asyncio.Lock()
        # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
        self.client = AsyncOpenAI(base_url=base_url or None, api_key=api_key or "not-needed",
                                  max_retries=max_retries, timeout=timeout, http_client=http_client)

    @classmethod
    def from_env(cls, prefix: str = "LLM") -> "OpenAICompatLLM":
        return cls(
            model=os.environ[f"{prefix}_MODEL"],
            base_url=os.getenv(f"{prefix}_BASE_URL"),
            api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY"),
            # raise on low-tier accounts that hit 429 often; the SDK honours Retry-After
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "3")),
            rpm=float(os.getenv(f"{prefix}_RPM", "0")),
        )

    async def _wait_for_slot(self) -> None:
        if not self.min_interval:
            return
        async with self._slot_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        kwargs = {"model": self.name, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        await self._wait_for_slot()
        resp = await self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in (msg.tool_calls or [])]
        usage = Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens) if resp.usage else Usage()
        return LLMResponse(msg.content, calls, usage)


class ScriptedLLM:
    """Replays fixed responses; for tests. A step may be a callable that sees the messages."""

    name = "scripted"

    def __init__(self, steps: Sequence[LLMResponse | Callable[[list[dict]], LLMResponse]]):
        self.steps = list(steps)
        self.seen: list[list[dict]] = []

    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.seen.append(json.loads(json.dumps(messages)))
        if not self.steps:
            return LLMResponse(content="(script exhausted)")
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


def call(name: str, args: dict | str, id: str | None = None) -> ToolCall:
    """Shorthand for building scripted tool calls."""
    raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return ToolCall(id or f"call_{name}", name, raw)

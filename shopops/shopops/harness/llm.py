"""Model backends behind one interface.

`OpenAICompatLLM` covers every server that speaks the OpenAI chat-completions
tool-calling format: OpenAI, a self-hosted vLLM (`--enable-auto-tool-choice`),
Ollama, and Anthropic's OpenAI-compatible endpoint.
"""

import json
import os
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
                 temperature: float = 0.0, max_retries: int = 3, timeout: float = 60, http_client=None):
        from openai import AsyncOpenAI

        self.name = model
        self.temperature = temperature
        # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed",
                                  max_retries=max_retries, timeout=timeout, http_client=http_client)

    @classmethod
    def from_env(cls, prefix: str = "LLM") -> "OpenAICompatLLM":
        return cls(
            model=os.environ[f"{prefix}_MODEL"],
            base_url=os.getenv(f"{prefix}_BASE_URL"),
            api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )

    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        kwargs = {"model": self.name, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
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

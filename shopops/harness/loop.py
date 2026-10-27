"""A small tool-calling agent loop, written without an agent framework.

What it handles, step by step:
- tool schemas: every call is parsed and validated against the tool's JSON Schema;
  bad JSON or bad arguments go back to the model as an error so it can fix them
- retries: model-call failures are retried; tool errors are returned to the model
- confirmation: tools flagged `requires_confirmation` go through a Confirmer
  (approve, reject, edit the arguments, or defer to a human queue)
- guards: policy checks that can block a call before it runs
- budgets: max steps, per-tool timeout, and repeated identical calls
- tracing: every step goes to a Tracer
"""

import asyncio
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import jsonschema

from shopops.harness.llm import LLM, ToolCall, Usage
from shopops.harness.trace import Tracer


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], Awaitable[str]]
    requires_confirmation: bool = False
    final: bool = False  # calling it ends the run; its arguments become RunResult.final

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class PendingAction:
    tool: str
    arguments: dict
    reason: str = ""  # the model's text alongside the call, if any


@dataclass
class Decision:
    status: Literal["approved", "rejected", "deferred"]
    arguments: dict | None = None  # set when the approver edited the call
    note: str = ""


class Confirmer(Protocol):
    async def __call__(self, action: PendingAction) -> Decision: ...


async def auto_approve(action: PendingAction) -> Decision:
    return Decision("approved")


class DeferToQueue:
    """Records side-effect calls instead of running them; a human approves them later."""

    def __init__(self):
        self.queued: list[PendingAction] = []

    async def __call__(self, action: PendingAction) -> Decision:
        self.queued.append(action)
        return Decision("deferred", note=f"queued as action #{len(self.queued) - 1}")


# A guard returns None to allow the call, or a message explaining why it is blocked.
Guard = Callable[[str, dict], str | None]


@dataclass
class ToolEvent:
    step: int
    tool: str
    arguments: dict | None
    outcome: Literal["ok", "error", "invalid", "blocked", "rejected", "deferred", "unknown_tool"]
    result: str
    latency_ms: float = 0.0


@dataclass
class RunResult:
    final: dict | None
    stop_reason: Literal["final", "text", "max_steps", "loop_detected", "llm_error"]
    steps: int
    events: list[ToolEvent]
    messages: list[dict]
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    def calls(self, outcome: str | None = None) -> list[str]:
        return [e.tool for e in self.events if outcome is None or e.outcome == outcome]


class Agent:
    def __init__(self, llm: LLM, tools: list[ToolSpec], system: str, *,
                 confirmer: Confirmer = auto_approve, guards: list[Guard] = (),
                 max_steps: int = 10, tool_timeout: float = 20, llm_retries: int = 2,
                 max_repeats: int = 3, retry_backoff: float = 1.0, tracer: Tracer | None = None):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.system = system
        self.confirmer = confirmer
        self.guards = list(guards)
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.llm_retries = llm_retries
        self.max_repeats = max_repeats
        self.retry_backoff = retry_backoff
        self.tracer = tracer or Tracer()

    async def run(self, user_message: str) -> RunResult:
        t0 = time.perf_counter()
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user_message}]
        schemas = [t.openai_schema() for t in self.tools.values()]
        events: list[ToolEvent] = []
        usage = Usage()
        seen = Counter()
        final = None
        stop = "max_steps"
        step = 0
        nudged = False
        self.tracer.log("run_start", model=self.llm.name, user=user_message)

        while step < self.max_steps:
            step += 1
            resp = None
            for attempt in range(self.llm_retries + 1):
                try:
                    resp = await self.llm.complete(messages, schemas)
                    break
                except Exception as exc:  # network, rate limit after SDK retries, bad server reply
                    self.tracer.log("llm_error", step=step, attempt=attempt, error=repr(exc))
                    if attempt == self.llm_retries:
                        stop = "llm_error"
                    else:
                        await asyncio.sleep(self.retry_backoff * 2 ** attempt)
            if resp is None:
                break
            usage.add(resp.usage)
            self.tracer.log("llm", step=step, content=resp.content,
                            tool_calls=[(c.name, c.arguments) for c in resp.tool_calls],
                            prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens)

            assistant = {"role": "assistant", "content": resp.content}
            if resp.tool_calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                            "function": {"name": c.name, "arguments": c.arguments}}
                                           for c in resp.tool_calls]
            messages.append(assistant)

            if not resp.tool_calls:
                has_final_tool = any(t.final for t in self.tools.values())
                if has_final_tool and not nudged:
                    # Models sometimes answer in prose; one reminder usually fixes it.
                    nudged = True
                    messages.append({"role": "user", "content": "Finish by calling the final tool with your result."})
                    continue
                stop = "text"
                break

            loop_hit = False
            for tc in resp.tool_calls:
                key = (tc.name, tc.arguments)
                seen[key] += 1
                if seen[key] > self.max_repeats:
                    loop_hit = True
                event = await self._run_tool(step, tc, resp.content or "")
                events.append(event)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": event.result})
                self.tracer.log("tool", **event.__dict__)
                if event.outcome == "ok" and self.tools[tc.name].final:
                    final = event.arguments
            if final is not None:
                stop = "final"
                break
            if loop_hit:
                stop = "loop_detected"
                break

        result = RunResult(final, stop, step, events, messages, usage, (time.perf_counter() - t0) * 1000)
        self.tracer.log("run_end", stop_reason=stop, steps=step, final=final,
                        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                        latency_ms=round(result.latency_ms, 1))
        return result

    async def _run_tool(self, step: int, tc: ToolCall, reason: str) -> ToolEvent:
        spec = self.tools.get(tc.name)
        if spec is None:
            return ToolEvent(step, tc.name, None, "unknown_tool",
                             f"Error: there is no tool named {tc.name!r}. Available: {', '.join(self.tools)}")
        try:
            args = json.loads(tc.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            jsonschema.validate(args, spec.parameters)
        except (ValueError, jsonschema.ValidationError) as exc:
            msg = exc.message if isinstance(exc, jsonschema.ValidationError) else str(exc)
            return ToolEvent(step, tc.name, None, "invalid",
                             f"Error: invalid arguments for {tc.name}: {msg}. Fix them and call again.")

        for guard in self.guards:
            problem = guard(tc.name, args)
            if problem:
                return ToolEvent(step, tc.name, args, "blocked",
                                 f"Blocked by policy: {problem}. Rewrite and call again.")

        if spec.requires_confirmation:
            decision = await self.confirmer(PendingAction(tc.name, args, reason))
            if decision.status == "rejected":
                return ToolEvent(step, tc.name, args, "rejected",
                                 f"A human reviewer declined this action. {decision.note}".strip())
            if decision.status == "deferred":
                return ToolEvent(step, tc.name, args, "deferred",
                                 f"Action sent to a human for approval ({decision.note}). "
                                 "Treat it as done; do not call it again.")
            if decision.arguments is not None:
                args = decision.arguments

        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(spec.handler(args), self.tool_timeout)
            outcome = "ok"
        except TimeoutError:
            result, outcome = f"Error: {tc.name} timed out after {self.tool_timeout}s.", "error"
        except Exception as exc:
            result, outcome = f"Error from {tc.name}: {exc}", "error"
        return ToolEvent(step, tc.name, args, outcome, result, (time.perf_counter() - t0) * 1000)

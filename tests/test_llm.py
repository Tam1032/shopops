"""OpenAICompatLLM against a fake chat-completions server (same wire format as OpenAI / vLLM)."""

import json

import httpx

from shopops.harness.llm import OpenAICompatLLM


async def test_parses_tool_calls_and_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "get_order", "arguments": "{\"order_id\": \"O1003\"}"}}]}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 129},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = OpenAICompatLLM("qwen", base_url="http://vllm.local/v1", http_client=client)
    tools = [{"type": "function", "function": {"name": "get_order", "parameters": {"type": "object"}}}]
    resp = await llm.complete([{"role": "user", "content": "hi"}], tools)
    assert resp.tool_calls[0].name == "get_order" and json.loads(resp.tool_calls[0].arguments) == {"order_id": "O1003"}
    assert resp.usage.prompt_tokens == 120 and seen["tools"] == tools and seen["temperature"] == 0.0

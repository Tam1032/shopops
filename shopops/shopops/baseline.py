"""Keyword-rule "model" that speaks the same tool-calling protocol as a real LLM.

It is the floor in the eval table and lets the whole pipeline (harness, MCP,
eval, CI) run with no API key. It is not meant to be good.
"""

import json
import re

from shopops.harness.llm import LLMResponse, ToolCall

RULES = [  # first match wins: (topic, keywords)
    ("wrong_item", r"giao sai|nhận được .*mà|thiếu|wrong item|missing"),
    ("size_fit", r"size|chật|rộng|form"),
    ("delivery", r"giao|ship|chờ|trễ|chậm|delivery|shipping"),
    ("packaging", r"hộp|đóng gói|móp|bẹp|box"),
    ("service", r"tin nhắn|rep|tư vấn|trả lời"),
    ("quality", r"hỏng|rách|rò|phai|ra màu|chập chờn|giả|mẩn|bết|không như|khác ảnh|không có tiếng|broken"),
    ("praise", r"đẹp|tốt|ổn|hợp|great|good|chất lượng"),
]
ESCALATE = r"hỏng|rách|rò|ra màu|chập chờn|giả|mẩn|không có tiếng|hoàn tiền|trả hàng|đổi|giao sai|thiếu"


def _classify(review: dict) -> tuple[str, str, str]:
    text = review["text"].lower()
    topic = next((t for t, pat in RULES if re.search(pat, text)), "other")
    rating = review["rating"]
    sentiment = "positive" if rating >= 4 else "neutral" if rating == 3 else "negative"
    action = "escalate" if rating <= 2 and re.search(ESCALATE, text) else "reply"
    return sentiment, topic, action


def _reply(review: dict, topic: str, facts: dict) -> str:
    if not re.search(r"[à-ỹđ]", review["text"].lower()):
        return "Thank you for your review! We're glad you like it and hope to see you again."
    if topic == "delivery" and "order" in facts:
        note = facts["order"].get("note") or f"trạng thái đơn: {facts['order']['status']}"
        return f"Shop xin lỗi bạn vì đơn hàng đến chậm. Shop đã kiểm tra: {note}. Cảm ơn bạn đã kiên nhẫn chờ."
    if topic == "size_fit" and "stock" in facts:
        avail = [k for k, v in facts["stock"]["stock"].items() if v > 0]
        return f"Shop cảm ơn bạn đã góp ý về size. Hiện shop còn các size: {', '.join(avail)}. Bạn nhắn shop để được tư vấn nhé."
    if review["rating"] >= 4:
        return "Shop cảm ơn bạn đã tin tưởng và ủng hộ. Hẹn gặp lại bạn ở những đơn sau nhé!"
    return "Shop xin lỗi vì trải nghiệm chưa tốt của bạn. Shop đã ghi nhận góp ý để cải thiện."


class KeywordBaseline:
    name = "keyword-baseline"

    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        results = {}  # tool name -> parsed result
        for m in messages:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls", []):
                    results.setdefault(tc["function"]["name"], None)
            if m["role"] == "tool":
                name = next(tc["function"]["name"] for a in messages if a["role"] == "assistant"
                            for tc in a.get("tool_calls", []) if tc["id"] == m["tool_call_id"])
                try:
                    results[name] = json.loads(m["content"])
                except ValueError:
                    results[name] = {"error": m["content"]}

        def do(name: str, args: dict) -> LLMResponse:
            n = sum(1 for m in messages if m["role"] == "tool")
            return LLMResponse(tool_calls=[ToolCall(f"call_{n}", name, json.dumps(args, ensure_ascii=False))])

        if "get_review" not in results:
            rid = re.search(r"R\d+", messages[1]["content"]).group(0)
            return do("get_review", {"review_id": rid})
        review = results["get_review"]
        sentiment, topic, action = _classify(review)
        if topic in ("delivery", "wrong_item") and review["rating"] <= 3 and "get_order" not in results:
            return do("get_order", {"order_id": review["order_id"]})
        if topic == "size_fit" and "get_inventory" not in results:
            return do("get_inventory", {"sku": review["sku"]})
        if "post_reply" not in results and "escalate_to_human" not in results:
            if action == "escalate":
                return do("escalate_to_human", {"review_id": review["review_id"],
                                                "reason": f"{topic}: {review['text'][:120]}", "priority": "normal"})
            facts = {"order": results.get("get_order"), "stock": results.get("get_inventory")}
            facts = {k: v for k, v in facts.items() if v}
            return do("post_reply", {"review_id": review["review_id"], "text": _reply(review, topic, facts)})
        return do("submit_triage", {"sentiment": sentiment, "topic": topic, "action": action,
                                    "summary": f"{topic} ({sentiment})"})

"""Harness behaviour with a scripted model: schema errors, guards, confirmation, budgets."""

import json

from shopops.agent import build_agent, reply_guard, triage
from shopops.harness.llm import LLMResponse, ScriptedLLM, call
from shopops.harness.loop import Agent, Decision, DeferToQueue, ToolSpec


def say(*calls) -> LLMResponse:
    return LLMResponse(tool_calls=list(calls))


SUBMIT = {"sentiment": "negative", "topic": "delivery", "action": "reply", "summary": "late"}


async def test_happy_path_posts_reply(mcp, store):
    llm = ScriptedLLM([
        say(call("get_review", {"review_id": "R003"})),
        say(call("get_order", {"order_id": "O1003"})),
        say(call("post_reply", {"review_id": "R003", "text": "Shop xin lỗi, đơn trễ do kho trung chuyển quá tải."})),
        say(call("submit_triage", SUBMIT)),
    ])
    run = await triage(await build_agent(llm, mcp), "R003")
    assert run.stop_reason == "final" and run.final == SUBMIT
    assert store.reviews["R003"]["status"] == "replied"


async def test_bad_json_and_schema_errors_go_back_to_model(mcp):
    llm = ScriptedLLM([
        say(call("get_order", "{not json")),
        say(call("escalate_to_human", {"review_id": "R002", "reason": "broken", "priority": "urgent"})),
        say(call("submit_triage", SUBMIT)),
    ])
    run = await triage(await build_agent(llm, mcp), "R002")
    assert run.calls("invalid") == ["get_order", "escalate_to_human"]
    tool_msgs = [m["content"] for m in run.messages if m["role"] == "tool"]
    assert "invalid arguments" in tool_msgs[0] and "'urgent' is not one of" in tool_msgs[1]


async def test_unknown_tool_and_marketplace_error(mcp):
    llm = ScriptedLLM([
        say(call("delete_review", {"review_id": "R001"}), call("get_order", {"order_id": "NOPE"})),
        say(call("submit_triage", SUBMIT)),
    ])
    run = await triage(await build_agent(llm, mcp), "R001")
    assert [e.outcome for e in run.events[:2]] == ["unknown_tool", "error"]
    assert "404" in run.events[1].result


async def test_guard_blocks_refund_promise(mcp, store):
    llm = ScriptedLLM([
        say(call("post_reply", {"review_id": "R031", "text": "Shop sẽ hoàn tiền 100% và tặng voucher 500k!"})),
        say(call("post_reply", {"review_id": "R031", "text": "Cảm ơn bạn đã ủng hộ shop."})),
        say(call("submit_triage", {**SUBMIT, "topic": "other"})),
    ])
    run = await triage(await build_agent(llm, mcp), "R031")
    assert run.calls("blocked") == ["post_reply"] and run.calls("ok")[0] == "post_reply"
    assert store.reviews["R031"]["reply"]["text"] == "Cảm ơn bạn đã ủng hộ shop."


def test_guard_rules():
    assert reply_guard("post_reply", {"text": "Shop tặng bạn mã giảm giá"})
    assert reply_guard("post_reply", {"text": "We will refund you"})
    assert reply_guard("post_reply", {"text": "Shop đền 200k"})
    assert reply_guard("post_reply", {"text": "Áo 100% cotton, size L còn 5 cái."}) is None
    assert reply_guard("get_order", {"text": "hoàn tiền"}) is None


async def test_confirmation_reject_edit_defer(mcp, store):
    reply = call("post_reply", {"review_id": "R001", "text": "draft"})

    async def reject(action):
        return Decision("rejected", note="too short")

    llm = ScriptedLLM([say(reply), say(call("submit_triage", SUBMIT))])
    run = await triage(await build_agent(llm, mcp, confirmer=reject), "R001")
    assert run.calls("rejected") == ["post_reply"] and store.reviews["R001"]["status"] == "new"

    async def edit(action):
        return Decision("approved", arguments={**action.arguments, "text": "Cảm ơn bạn nhiều!"})

    llm = ScriptedLLM([say(reply), say(call("submit_triage", SUBMIT))])
    await triage(await build_agent(llm, mcp, confirmer=edit), "R001")
    assert store.reviews["R001"]["reply"]["text"] == "Cảm ơn bạn nhiều!"

    queue = DeferToQueue()
    llm = ScriptedLLM([say(call("escalate_to_human", {"review_id": "R002", "reason": "x", "priority": "normal"})),
                       say(call("submit_triage", SUBMIT))])
    run = await triage(await build_agent(llm, mcp, confirmer=queue), "R002")
    assert run.calls("deferred") == ["escalate_to_human"] and len(queue.queued) == 1
    assert store.reviews["R002"]["status"] == "new"  # nothing happened until a human approves


async def test_read_only_tools_skip_confirmation(mcp):
    asked = []

    async def confirmer(action):
        asked.append(action.tool)
        return Decision("approved")

    llm = ScriptedLLM([say(call("get_review", {"review_id": "R001"}), call("get_inventory", {"sku": "P01"})),
                       say(call("submit_triage", SUBMIT))])
    await triage(await build_agent(llm, mcp, confirmer=confirmer), "R001")
    assert asked == []


async def test_loop_detection_and_max_steps():
    async def noop(args):
        return "{}"

    tools = [ToolSpec("ping", "", {"type": "object"}, noop)]
    same = say(call("ping", {}))
    run = await Agent(ScriptedLLM([same] * 10), tools, "sys", max_repeats=2).run("go")
    assert run.stop_reason == "loop_detected" and run.steps == 3

    steps = [say(call("ping", {"i": i})) for i in range(10)]
    run = await Agent(ScriptedLLM(steps), tools, "sys", max_steps=4).run("go")
    assert run.stop_reason == "max_steps" and run.steps == 4


async def test_llm_retry_then_nudge():
    attempts = []

    def flaky(messages):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("boom")
        return LLMResponse(content="Done, it is fine.")

    async def submit(args):
        return "recorded"

    tools = [ToolSpec("finish", "", {"type": "object"}, submit, final=True)]
    llm = ScriptedLLM([flaky, flaky, say(call("finish", {"ok": True}))])
    agent = Agent(llm, tools, "sys", llm_retries=1, retry_backoff=0)
    run = await agent.run("go")
    assert run.stop_reason == "final" and run.final == {"ok": True}
    assert any(m["content"].startswith("Finish by calling") for m in run.messages if m["role"] == "user")


async def test_tracer_records_steps(mcp, tmp_path):
    from shopops.harness.trace import Tracer

    tracer = Tracer(tmp_path / "t.jsonl", run_id="R001")
    llm = ScriptedLLM([say(call("get_review", {"review_id": "R001"})), say(call("submit_triage", SUBMIT))])
    await triage(await build_agent(llm, mcp, tracer=tracer), "R001")
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and "tool" in kinds

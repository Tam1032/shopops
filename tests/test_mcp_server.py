import json


async def test_tools_and_annotations(mcp):
    tools = {t.name: t for t in (await mcp.list_tools()).tools}
    assert {"get_review", "get_order", "get_inventory", "post_reply", "escalate_to_human"} <= set(tools)
    assert tools["get_order"].annotations.read_only_hint is True
    assert tools["post_reply"].annotations.read_only_hint is False
    assert tools["escalate_to_human"].input_schema["properties"]["priority"]["enum"] == ["normal", "high"]


async def test_call_and_error(mcp):
    ok = await mcp.call_tool("get_inventory", {"sku": "P02"})
    assert json.loads(ok.content[0].text)["stock"]["29"] == 0
    bad = await mcp.call_tool("get_order", {"order_id": "NOPE"})
    assert bad.is_error and "404" in bad.content[0].text


async def test_resources(mcp):
    res = await mcp.read_resource("shop://policies")
    assert "escalate_to_human" in res.contents[0].text

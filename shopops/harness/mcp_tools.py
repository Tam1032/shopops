"""Turn the tools of a connected MCP server into harness ToolSpecs."""

import json

from mcp import Client

from shopops.harness.loop import ToolSpec


def _text(result) -> str:
    parts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    return "\n".join(parts) if parts else json.dumps(result.structured_content or {}, ensure_ascii=False)


async def load_mcp_tools(client: Client, allow: set[str] | None = None) -> list[ToolSpec]:
    """Any tool not marked read-only by the server needs confirmation before it runs."""
    listed = await client.list_tools()
    specs = []
    for tool in listed.tools:
        if allow is not None and tool.name not in allow:
            continue
        ann = tool.annotations
        read_only = bool(ann and ann.read_only_hint)

        async def handler(args: dict, _name=tool.name) -> str:
            result = await client.call_tool(_name, args)
            text = _text(result)
            if result.is_error:
                raise RuntimeError(text)
            return text

        specs.append(ToolSpec(
            name=tool.name,
            description=tool.description or "",
            parameters=tool.input_schema,
            handler=handler,
            requires_confirmation=not read_only,
        ))
    return specs

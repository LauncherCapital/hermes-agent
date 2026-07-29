"""Trusted caller-context injection for Ringo MCP tools."""

from __future__ import annotations

from typing import Any

from tools.registry import registry


_RINGO_CALLER_TOOLSETS = {"mcp-ringo_ie", "mcp-ringo_admin"}


def _transform_tool_args(
    *,
    tool_name: str,
    args: dict[str, Any],
    trusted_runtime_metadata: dict[str, str] | None = None,
    **_: object,
) -> dict[str, Any] | None:
    if registry.get_toolset_for_tool(tool_name) not in _RINGO_CALLER_TOOLSETS:
        return None

    schema = registry.get_schema(tool_name) or {}
    properties = (schema.get("parameters") or {}).get("properties") or {}
    if "caller_token" not in properties:
        return None

    transformed = dict(args)
    transformed["caller_token"] = str(
        (trusted_runtime_metadata or {}).get("slack_caller_token") or ""
    ).strip()
    return transformed


def register(ctx) -> None:
    ctx.register_hook("transform_tool_args", _transform_tool_args)

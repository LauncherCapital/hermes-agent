"""Trusted caller-context injection for Ringo MCP tools."""

from __future__ import annotations

import json
import logging
from typing import Any

from tools.registry import registry


logger = logging.getLogger(__name__)

_RINGO_CALLER_TOOLSETS = {"mcp-ringo_ie", "mcp-ringo_admin"}
_RINGO_MCP_SERVERS = {"ringo_ie", "ringo_admin"}


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
    metadata = registry.get_metadata(tool_name)
    hidden_arguments = metadata.get("mcp_hidden_arguments") or ()
    if (
        "caller_token" not in properties
        and "caller_token" not in hidden_arguments
    ):
        return None

    transformed = dict(args)
    transformed["caller_token"] = str(
        (trusted_runtime_metadata or {}).get("slack_caller_token") or ""
    ).strip()
    return transformed


def _build_mcp_call_meta(
    *,
    server_name: str,
    remote_tool_name: str = "",
    trusted_runtime_metadata: dict[str, str] | None = None,
    **_: object,
) -> dict[str, str] | None:
    if server_name not in _RINGO_MCP_SERVERS:
        return None
    caller_grant = str(
        (trusted_runtime_metadata or {}).get("slack_caller_token") or ""
    ).strip()
    if not caller_grant:
        return None
    logger.info(
        "Ringo MCP caller metadata attached server=%s tool=%s",
        server_name,
        str(remote_tool_name or ""),
    )
    return {"ringo/caller_grant": caller_grant}


def _caller_policy(
    trusted_runtime_metadata: dict[str, str] | None,
) -> tuple[str, set[str], bool]:
    runtime = trusted_runtime_metadata or {}
    explicit_capabilities = "caller_capabilities" in runtime
    legacy_has_caller = bool(
        str(runtime.get("slack_caller_token") or "").strip()
    )
    mode = str(
        runtime.get("caller_mode")
        or ("interactive" if legacy_has_caller else "system")
    ).strip()
    raw = str(runtime.get("caller_capabilities") or "[]")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = []
    capabilities = {
        str(value).strip()
        for value in parsed
        if isinstance(value, str) and str(value).strip()
    } if isinstance(parsed, list) else set()
    if legacy_has_caller:
        capabilities.add("caller")
    return mode, capabilities, explicit_capabilities


def _filter_tool_names(
    *,
    tool_names: tuple[str, ...],
    trusted_runtime_metadata: dict[str, str] | None = None,
    **_: object,
) -> list[str]:
    mode, capabilities, explicit_capabilities = _caller_policy(
        trusted_runtime_metadata
    )
    allowed: list[str] = []
    for tool_name in tool_names:
        if registry.get_toolset_for_tool(tool_name) not in _RINGO_CALLER_TOOLSETS:
            allowed.append(tool_name)
            continue
        metadata = registry.get_metadata(tool_name)
        access = (
            (metadata.get("mcp_meta") or {}).get("ringo/access") or {}
        )
        if not isinstance(access, dict) or not access:
            # Migration default: unannotated Ringo tools remain project-visible.
            allowed.append(tool_name)
            continue
        if access.get("caller") == "required" and "caller" not in capabilities:
            continue
        modes = access.get("modes")
        if isinstance(modes, list) and modes and mode not in modes:
            continue
        capability = str(access.get("capability") or "").strip()
        if (
            capability
            and capability not in capabilities
            and explicit_capabilities
        ):
            continue
        allowed.append(tool_name)
    return allowed


def register(ctx) -> None:
    ctx.register_hook("transform_tool_args", _transform_tool_args)
    ctx.register_hook("build_mcp_call_meta", _build_mcp_call_meta)
    ctx.register_hook("filter_tool_names", _filter_tool_names)

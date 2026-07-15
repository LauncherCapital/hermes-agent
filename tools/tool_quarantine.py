"""Process-global registry of tools the provider rejected at the schema level.

When a provider's tool-schema validator rejects ONE tool's ``input_schema``
(e.g. Anthropic: ``tools.N.custom.input_schema: JSON schema is invalid``), it
400s the ENTIRE request — so every tool turn fails, not just calls to the bad
tool. The conversation loop recovers by dropping that one tool and retrying
(see ``FailoverReason.invalid_tool_schema``). This module remembers what was
dropped so it can be surfaced to the user instead of silently disappearing:

* per run — the loop also stashes the drops on the agent for ``result_json``
  (the dashboard activity feed reads that), and
* persistently — this registry backs ``GET /api/tools/quarantined`` (the
  dashboard Integrations page shows a badge on the offending MCP server).

Keyed by tool name (already server-prefixed for MCP tools), so re-registering
a fixed tool that then validates clean simply stops being reported once
``clear()`` is called on reconnect.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

_lock = threading.Lock()
# tool_name -> {"tool", "server", "reason", "first_seen", "last_seen", "count"}
_quarantined: dict[str, dict] = {}


def quarantine(tool_name: str, *, server: Optional[str], reason: str) -> None:
    """Record (or refresh) a tool the provider rejected as schema-invalid."""
    now = time.time()
    reason = (reason or "").strip()[:300]
    with _lock:
        entry = _quarantined.get(tool_name)
        if entry is None:
            _quarantined[tool_name] = {
                "tool": tool_name,
                "server": server or None,
                "reason": reason,
                "first_seen": now,
                "last_seen": now,
                "count": 1,
            }
        else:
            entry["last_seen"] = now
            entry["count"] += 1
            if reason:
                entry["reason"] = reason
            if server:
                entry["server"] = server


def clear(tool_name: str) -> None:
    """Drop a tool's quarantine record (e.g. when its MCP server reconnects
    with a fresh, valid tool list)."""
    with _lock:
        _quarantined.pop(tool_name, None)


def clear_server(server: str) -> None:
    """Drop all quarantine records for a server (reconnect / reconfigure)."""
    with _lock:
        for name in [k for k, v in _quarantined.items() if v.get("server") == server]:
            _quarantined.pop(name, None)


def list_quarantined() -> list[dict]:
    """Snapshot of all currently quarantined tools, newest-seen first."""
    with _lock:
        items = [dict(v) for v in _quarantined.values()]
    items.sort(key=lambda v: v.get("last_seen") or 0, reverse=True)
    return items

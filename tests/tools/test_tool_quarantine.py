"""Tests for tools/tool_quarantine.py — the schema-rejected tool registry."""

from __future__ import annotations

import importlib


def _fresh():
    # Process-global module state — reimport clean per test.
    import tools.tool_quarantine as q
    importlib.reload(q)
    return q


def test_quarantine_records_and_lists():
    q = _fresh()
    q.quarantine("github__search_issues", server="github", reason="bad schema")
    items = q.list_quarantined()
    assert len(items) == 1
    it = items[0]
    assert it["tool"] == "github__search_issues"
    assert it["server"] == "github"
    assert it["reason"] == "bad schema"
    assert it["count"] == 1
    assert it["first_seen"] <= it["last_seen"]


def test_quarantine_dedupes_and_counts():
    q = _fresh()
    q.quarantine("t", server="s", reason="r1")
    q.quarantine("t", server="s", reason="r2")
    items = q.list_quarantined()
    assert len(items) == 1
    assert items[0]["count"] == 2
    assert items[0]["reason"] == "r2"  # latest reason wins


def test_clear_and_clear_server():
    q = _fresh()
    q.quarantine("a", server="s1", reason="r")
    q.quarantine("b", server="s2", reason="r")
    q.clear("a")
    assert {i["tool"] for i in q.list_quarantined()} == {"b"}
    q.quarantine("c", server="s2", reason="r")
    q.clear_server("s2")
    assert q.list_quarantined() == []


def test_builtin_tool_has_null_server():
    q = _fresh()
    q.quarantine("some_builtin", server=None, reason="r")
    assert q.list_quarantined()[0]["server"] is None

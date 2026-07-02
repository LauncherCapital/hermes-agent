"""Keepalive-driven MCP tool drift detection.

Stateless streamable-HTTP MCP servers cannot push
``notifications/tools/list_changed`` (no server→client channel; a redeployed
server has no memory of old sessions). The keepalive already fetches
``list_tools`` every cycle — these tests cover the diff that turns that
result into the same refresh path the notification would have triggered.
"""

from types import SimpleNamespace
from unittest.mock import patch

from tools.mcp_tool import MCPServerTask, _tools_signature


def _tool(name, description="", schema=None):
    return SimpleNamespace(name=name, description=description, inputSchema=schema or {})


class TestToolsSignature:
    def test_order_insensitive(self):
        a = [_tool("x"), _tool("y")]
        b = [_tool("y"), _tool("x")]
        assert _tools_signature(a) == _tools_signature(b)

    def test_detects_added_tool(self):
        assert _tools_signature([_tool("x")]) != _tools_signature([_tool("x"), _tool("y")])

    def test_detects_description_change(self):
        # Description edits steer the model as much as new tools — must diff.
        assert _tools_signature([_tool("x", "old")]) != _tools_signature([_tool("x", "new")])

    def test_detects_schema_change(self):
        old = [_tool("x", schema={"type": "object", "properties": {"q": {}}})]
        new = [_tool("x", schema={"type": "object", "properties": {"q": {}, "limit": {}}})]
        assert _tools_signature(old) != _tools_signature(new)

    def test_empty_and_none_equal(self):
        assert _tools_signature([]) == _tools_signature(None)


class TestKeepaliveToolsCheck:
    def _task(self, tools):
        # MCPServerTask uses __slots__ — patch the refresh hook on the class.
        task = MCPServerTask("srv")
        task._tools = tools
        return task

    def test_no_change_no_refresh(self):
        task = self._task([_tool("x", "d")])
        result = SimpleNamespace(tools=[_tool("x", "d")])
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as refresh:
            assert task._keepalive_tools_check(result) is False
            refresh.assert_not_called()

    def test_change_schedules_refresh(self):
        task = self._task([_tool("x")])
        result = SimpleNamespace(tools=[_tool("x"), _tool("y")])
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as refresh:
            assert task._keepalive_tools_check(result) is True
            refresh.assert_called_once()

    def test_result_without_tools_attr_treated_as_empty(self):
        task = self._task([])
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as refresh:
            assert task._keepalive_tools_check(object()) is False
            refresh.assert_not_called()

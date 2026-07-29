"""Tests for dispatch-only tool argument transformation."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import hermes_cli.plugins as plugins_mod
import model_tools
from agent.agent_runtime_helpers import invoke_tool


def test_transformed_args_reach_dispatch_but_not_observers(monkeypatch):
    from tools.registry import registry

    dispatched = {}
    observed = {}

    def _dispatch(name, args, **_):
        dispatched.update(args)
        return '{"ok": true}'

    def _hook(hook_name, **kwargs):
        if hook_name == "transform_tool_args":
            return [{**kwargs["args"], "caller_token": "trusted-token"}]
        if hook_name == "post_tool_call":
            observed.update(kwargs["args"])
        return []

    monkeypatch.setattr(registry, "dispatch", _dispatch)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _hook)

    result = model_tools.handle_function_call(
        "dummy_tool",
        {"caller_token": "model-token", "value": 1},
        task_id="task",
        session_id="session",
        tool_call_id="call",
        skip_pre_tool_call_hook=True,
        trusted_runtime_metadata={"slack_caller_token": "trusted-token"},
    )

    assert result == '{"ok": true}'
    assert dispatched == {"caller_token": "trusted-token", "value": 1}
    assert observed == {"caller_token": "model-token", "value": 1}


def test_transform_failure_dispatches_original_copy(monkeypatch):
    from tools.registry import registry

    dispatched = {}
    monkeypatch.setattr(
        registry,
        "dispatch",
        lambda _name, args, **_: dispatched.update(args) or '{"ok": true}',
    )
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    model_tools.handle_function_call(
        "dummy_tool",
        {"value": 1},
        skip_pre_tool_call_hook=True,
    )

    assert dispatched == {"value": 1}


def test_no_hook_returns_distinct_copy():
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    original = {"nested": {"value": 1}}

    transformed = plugins_mod.get_transformed_tool_args("dummy", original)

    assert transformed == original
    assert transformed is not original


def test_concurrent_agent_path_forwards_trusted_runtime_metadata():
    runtime = {"slack_caller_token": "trusted-token"}
    dispatch = Mock(return_value='{"ok": true}')
    agent = SimpleNamespace(
        _memory_manager=None,
        _trusted_runtime_metadata=runtime,
        _current_turn_id="turn",
        _current_api_request_id="request",
        session_id="session",
        valid_tool_names=set(),
        enabled_toolsets=None,
        disabled_toolsets=None,
    )

    with patch(
        "agent.agent_runtime_helpers._ra",
        return_value=SimpleNamespace(handle_function_call=dispatch),
    ):
        result = invoke_tool(
            agent,
            "dummy_tool",
            {"value": 1},
            "task",
            tool_call_id="call",
            pre_tool_block_checked=True,
        )

    assert result == '{"ok": true}'
    assert dispatch.call_args.kwargs["trusted_runtime_metadata"] is runtime

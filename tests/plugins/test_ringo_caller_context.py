import importlib.util
import sys
from pathlib import Path

from tools.registry import registry


PLUGIN_DIR = Path(__file__).parents[2] / "plugins" / "ringo-caller-context"


def _load_plugin():
    package_name = "test_ringo_caller_context_plugin"
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _register_tool(
    name: str,
    *,
    toolset: str,
    caller_token: bool,
    access: dict | None = None,
    remote_tool: str | None = None,
) -> None:
    properties = {"value": {"type": "string"}}
    if caller_token:
        properties["caller_token"] = {"type": "string"}
    registry.register(
        name=name,
        toolset=toolset,
        schema={
            "name": name,
            "parameters": {"type": "object", "properties": properties},
        },
        handler=lambda **_: "{}",
        metadata={
            "mcp_meta": {"ringo/access": access},
            "mcp_hidden_arguments": (
                ["caller_token"] if caller_token else []
            ),
            "mcp_remote_tool": remote_tool or "",
        } if access or caller_token or remote_tool else None,
    )


def test_injects_and_overwrites_caller_token_for_ringo_tools():
    plugin = _load_plugin()
    for name, toolset in (
        ("mcp_ringo_ie_artifact_create", "mcp-ringo_ie"),
        ("mcp_ringo_admin_admin_list_projects", "mcp-ringo_admin"),
    ):
        _register_tool(name, toolset=toolset, caller_token=True)

        result = plugin._transform_tool_args(
            tool_name=name,
            args={"value": "x", "caller_token": "model-token"},
            trusted_runtime_metadata={"slack_caller_token": "trusted-token"},
        )

        assert result == {"value": "x", "caller_token": "trusted-token"}


def test_does_not_inject_into_other_tools_and_blanks_untrusted_token():
    plugin = _load_plugin()
    other = "mcp_other_write"
    ringo_without_caller = "mcp_ringo_ie_memory_search"
    _register_tool(other, toolset="mcp-other", caller_token=True)
    _register_tool(
        ringo_without_caller,
        toolset="mcp-ringo_ie",
        caller_token=False,
    )

    assert plugin._transform_tool_args(
        tool_name=other,
        args={"caller_token": "model-token"},
        trusted_runtime_metadata={"slack_caller_token": "trusted-token"},
    ) is None
    assert plugin._transform_tool_args(
        tool_name=ringo_without_caller,
        args={"value": "x"},
        trusted_runtime_metadata={"slack_caller_token": "trusted-token"},
    ) is None
    assert plugin._transform_tool_args(
        tool_name="mcp_ringo_ie_artifact_create",
        args={"caller_token": "model-token"},
        trusted_runtime_metadata={},
    ) == {"caller_token": ""}


def test_injects_hidden_legacy_argument_not_visible_in_provider_schema():
    plugin = _load_plugin()
    name = "mcp_ringo_ie_profile_get"
    registry.register(
        name=name,
        toolset="mcp-ringo_ie",
        schema={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
        handler=lambda **_: "{}",
        metadata={"mcp_hidden_arguments": ["caller_token"]},
    )

    assert plugin._transform_tool_args(
        tool_name=name,
        args={"value": "x"},
        trusted_runtime_metadata={"slack_caller_token": "trusted-token"},
    ) == {"value": "x", "caller_token": "trusted-token"}


def test_injects_current_slack_destination_when_model_omits_it():
    plugin = _load_plugin()
    for remote_tool in ("send_message", "slack_upload_file"):
        name = f"mcp_ringo_ie_{remote_tool}"
        _register_tool(
            name,
            toolset="mcp-ringo_ie",
            caller_token=False,
            remote_tool=remote_tool,
        )

        assert plugin._transform_tool_args(
            tool_name=name,
            args={"value": "x"},
            trusted_runtime_metadata={
                "channel_id": "D_CURRENT",
                "reply_target_ts": "123.456",
            },
        ) == {
            "value": "x",
            "channel": "D_CURRENT",
            "thread_ts": "123.456",
        }


def test_explicit_slack_destination_is_never_overwritten_or_threaded():
    plugin = _load_plugin()
    name = "mcp_ringo_ie_send_message"
    _register_tool(
        name,
        toolset="mcp-ringo_ie",
        caller_token=False,
        remote_tool="send_message",
    )

    assert plugin._transform_tool_args(
        tool_name=name,
        args={"channel": "C_OTHER", "value": "x"},
        trusted_runtime_metadata={
            "channel_id": "D_CURRENT",
            "reply_target_ts": "123.456",
        },
    ) is None


def test_injects_current_conversation_for_history_without_forcing_thread():
    plugin = _load_plugin()
    for remote_tool, argument in (
        ("slack_fetch_history", "channel"),
        ("message_fetch_history", "conversation_id"),
    ):
        name = f"mcp_ringo_ie_{remote_tool}"
        _register_tool(
            name,
            toolset="mcp-ringo_ie",
            caller_token=False,
            remote_tool=remote_tool,
        )

        assert plugin._transform_tool_args(
            tool_name=name,
            args={},
            trusted_runtime_metadata={
                "channel_id": "D_CURRENT",
                "reply_target_ts": "123.456",
            },
        ) == {argument: "D_CURRENT"}


def test_watch_injects_current_channel_but_never_current_dm():
    plugin = _load_plugin()
    name = "mcp_ringo_ie_slack_watch_channel"
    _register_tool(
        name,
        toolset="mcp-ringo_ie",
        caller_token=False,
        remote_tool="slack_watch_channel",
    )

    assert plugin._transform_tool_args(
        tool_name=name,
        args={},
        trusted_runtime_metadata={
            "channel_id": "C_CURRENT",
            "channel_type": "channel",
        },
    ) == {"channel_id": "C_CURRENT"}
    assert plugin._transform_tool_args(
        tool_name=name,
        args={},
        trusted_runtime_metadata={
            "channel_id": "D_CURRENT",
            "channel_type": "im",
        },
    ) is None


def test_nudge_injects_current_conversation_and_reply_target():
    plugin = _load_plugin()
    name = "mcp_ringo_ie_nudge_create"
    _register_tool(
        name,
        toolset="mcp-ringo_ie",
        caller_token=False,
        remote_tool="nudge_create",
    )

    assert plugin._transform_tool_args(
        tool_name=name,
        args={"name": "follow up"},
        trusted_runtime_metadata={
            "channel_id": "D_CURRENT",
            "reply_target_ts": "123.456",
        },
    ) == {
        "name": "follow up",
        "channel": "D_CURRENT",
        "thread_ts": "123.456",
    }


def test_explicit_other_nudge_channel_does_not_inherit_current_thread():
    plugin = _load_plugin()
    name = "mcp_ringo_ie_nudge_create_explicit"
    _register_tool(
        name,
        toolset="mcp-ringo_ie",
        caller_token=False,
        remote_tool="nudge_create",
    )

    assert plugin._transform_tool_args(
        tool_name=name,
        args={"channel": "C_OTHER"},
        trusted_runtime_metadata={
            "channel_id": "D_CURRENT",
            "reply_target_ts": "123.456",
        },
    ) is None


def test_person_nudge_does_not_inherit_current_conversation():
    plugin = _load_plugin()
    name = "mcp_ringo_ie_nudge_create_person"
    _register_tool(
        name,
        toolset="mcp-ringo_ie",
        caller_token=False,
        remote_tool="nudge_create",
    )

    assert plugin._transform_tool_args(
        tool_name=name,
        args={"target": "slack:U_OTHER"},
        trusted_runtime_metadata={
            "channel_id": "D_CURRENT",
            "reply_target_ts": "123.456",
        },
    ) is None


def test_missing_trusted_destination_fails_closed_without_forging_arguments():
    plugin = _load_plugin()
    name = "mcp_ringo_ie_slack_upload_file"
    _register_tool(
        name,
        toolset="mcp-ringo_ie",
        caller_token=False,
        remote_tool="slack_upload_file",
    )

    assert plugin._transform_tool_args(
        tool_name=name,
        args={"filename": "report.md"},
        trusted_runtime_metadata={},
    ) is None


def test_builds_out_of_band_meta_only_for_ringo_servers():
    plugin = _load_plugin()

    assert plugin._build_mcp_call_meta(
        server_name="ringo_ie",
        trusted_runtime_metadata={"slack_caller_token": "trusted-token"},
    ) == {"ringo/caller_grant": "trusted-token"}
    assert plugin._build_mcp_call_meta(
        server_name="ringo_admin",
        trusted_runtime_metadata={"slack_caller_token": "trusted-token"},
    ) == {"ringo/caller_grant": "trusted-token"}
    assert plugin._build_mcp_call_meta(
        server_name="github",
        trusted_runtime_metadata={"slack_caller_token": "trusted-token"},
    ) is None
    assert plugin._build_mcp_call_meta(
        server_name="ringo_ie",
        trusted_runtime_metadata={},
    ) is None


def test_filters_ringo_tools_by_caller_mode_and_capability():
    plugin = _load_plugin()
    public = "mcp_ringo_ie_public"
    profile = "mcp_ringo_ie_profile_get"
    admin = "mcp_ringo_admin_admin_credit_usage"
    system_only = "mcp_ringo_ie_nudge_list"
    other = "mcp_github_search"
    _register_tool(public, toolset="mcp-ringo_ie", caller_token=False)
    _register_tool(
        profile,
        toolset="mcp-ringo_ie",
        caller_token=True,
        access={
            "version": 1,
            "caller": "required",
            "capability": "caller",
        },
    )
    _register_tool(
        admin,
        toolset="mcp-ringo_admin",
        caller_token=True,
        access={
            "version": 1,
            "caller": "required",
            "modes": ["interactive"],
            "capability": "admin:admin_credit_usage",
        },
    )
    _register_tool(other, toolset="mcp-github", caller_token=False)
    _register_tool(
        system_only,
        toolset="mcp-ringo_ie",
        caller_token=False,
        access={
            "version": 1,
            "caller": "forbidden",
            "modes": ["scheduled", "system"],
        },
    )

    scheduled = plugin._filter_tool_names(
        tool_names=(public, profile, admin, system_only, other),
        trusted_runtime_metadata={
            "caller_mode": "scheduled",
            "caller_capabilities": "[]",
        },
    )
    assert set(scheduled) == {public, system_only, other}

    member = plugin._filter_tool_names(
        tool_names=(public, profile, admin, system_only, other),
        trusted_runtime_metadata={
            "caller_mode": "interactive",
            "caller_capabilities": '["caller"]',
        },
    )
    assert set(member) == {public, profile, other}

    admin_member = plugin._filter_tool_names(
        tool_names=(public, profile, admin, system_only, other),
        trusted_runtime_metadata={
            "caller_mode": "interactive",
            "caller_capabilities": (
                '["caller", "admin:admin_credit_usage"]'
            ),
        },
    )
    assert set(admin_member) == {public, profile, admin, other}


def test_legacy_runtime_keeps_capability_tools_visible_for_mixed_deploy():
    plugin = _load_plugin()
    admin = "mcp_ringo_admin_admin_credit_usage"
    _register_tool(
        admin,
        toolset="mcp-ringo_admin",
        caller_token=True,
        access={
            "version": 1,
            "caller": "required",
            "modes": ["interactive"],
            "capability": "admin:admin_credit_usage",
        },
    )

    assert plugin._filter_tool_names(
        tool_names=(admin,),
        trusted_runtime_metadata={
            "slack_caller_token": "trusted-token",
        },
    ) == [admin]

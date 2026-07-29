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
        } if access or caller_token else None,
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

    scheduled = plugin._filter_tool_names(
        tool_names=(public, profile, admin, other),
        trusted_runtime_metadata={
            "caller_mode": "scheduled",
            "caller_capabilities": "[]",
        },
    )
    assert set(scheduled) == {public, other}

    member = plugin._filter_tool_names(
        tool_names=(public, profile, admin, other),
        trusted_runtime_metadata={
            "caller_mode": "interactive",
            "caller_capabilities": '["caller"]',
        },
    )
    assert set(member) == {public, profile, other}

    admin_member = plugin._filter_tool_names(
        tool_names=(public, profile, admin, other),
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

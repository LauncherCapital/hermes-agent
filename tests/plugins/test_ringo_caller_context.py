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


def _register_tool(name: str, *, toolset: str, caller_token: bool) -> None:
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

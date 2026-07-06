"""Tests for the MCP admin surface on the API server adapter.

Covers:
- GET /v1/mcp/servers — installed (config.yaml) ⨝ live (registry) join,
  including installed-but-disconnected entries
- POST /admin/reload-mcp — full reconnect summary
- POST /admin/config — declarative apply of {model, mcp_servers, toolsets,
  web, env}; pure MCP additions use incremental discovery while removals
  force a full reload; ringo_ie removal is refused
- Auth enforcement (401 when API_SERVER_KEY is set)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_MCP_MOD = "tools.mcp_tool"


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/mcp/servers", adapter._handle_mcp_servers)
    app.router.add_post("/admin/reload-mcp", adapter._handle_admin_reload_mcp)
    app.router.add_post("/admin/config", adapter._handle_admin_config)
    return app


def _fake_live_server(tool_names):
    return SimpleNamespace(
        _tools=[SimpleNamespace(name=n) for n in tool_names],
        _is_http=lambda: True,
    )


def _save_servers(servers: dict):
    from hermes_cli.config import load_config, save_config

    config = load_config()
    config["mcp_servers"] = servers
    save_config(config)


@pytest.fixture
def adapter():
    return _make_adapter()


class TestAuth:
    @pytest.mark.asyncio
    async def test_endpoints_require_key(self):
        app = _create_app(_make_adapter(api_key="sk-secret"))
        async with TestClient(TestServer(app)) as cli:
            assert (await cli.get("/v1/mcp/servers")).status == 401
            assert (await cli.post("/admin/reload-mcp")).status == 401
            assert (await cli.post("/admin/config", json={})).status == 401

    @pytest.mark.asyncio
    async def test_valid_key_accepted(self):
        app = _create_app(_make_adapter(api_key="sk-secret"))
        headers = {"Authorization": "Bearer sk-secret"}
        async with TestClient(TestServer(app)) as cli:
            with patch.dict(f"{_MCP_MOD}._servers", {}, clear=True):
                assert (await cli.get("/v1/mcp/servers", headers=headers)).status == 200


class TestMcpInventory:
    @pytest.mark.asyncio
    async def test_join_installed_and_live(self, adapter):
        """Installed-but-down and connected servers are both reported."""
        _save_servers({
            "ringo_ie": {"url": "https://ie.example/mcp/", "headers": {"Authorization": "Bearer k"}},
            "broken": {"url": "https://down.example/mcp/"},
        })
        live = {"ringo_ie": _fake_live_server(["memory_search", "fact_record"])}
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.dict(f"{_MCP_MOD}._servers", live, clear=True):
                resp = await cli.get("/v1/mcp/servers")
                assert resp.status == 200
                data = {e["name"]: e for e in (await resp.json())["data"]}

        assert data["ringo_ie"]["connected"] is True
        assert data["ringo_ie"]["installed"] is True
        assert data["ringo_ie"]["tools"] == ["fact_record", "memory_search"]
        assert data["ringo_ie"]["tool_count"] == 2
        assert data["broken"]["connected"] is False
        assert data["broken"]["installed"] is True
        assert data["broken"]["tools"] == []

    @pytest.mark.asyncio
    async def test_headers_never_returned(self, adapter):
        _save_servers({"srv": {"url": "https://x/mcp/", "headers": {"Authorization": "Bearer secret"}}})
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.dict(f"{_MCP_MOD}._servers", {}, clear=True):
                body = await (await cli.get("/v1/mcp/servers")).text()
        assert "secret" not in body


class TestReloadMcp:
    @pytest.mark.asyncio
    async def test_reload_summary(self, adapter):
        app = _create_app(adapter)
        shutdown = MagicMock()
        discover = MagicMock(return_value=["mcp_srv_a", "mcp_srv_b"])
        async with TestClient(TestServer(app)) as cli:
            with patch.dict(f"{_MCP_MOD}._servers", {"srv": _fake_live_server(["a"])}, clear=True), \
                 patch(f"{_MCP_MOD}.shutdown_mcp_servers", shutdown), \
                 patch(f"{_MCP_MOD}.discover_mcp_tools", discover):
                resp = await cli.post("/admin/reload-mcp")
                assert resp.status == 200
                data = await resp.json()
        shutdown.assert_called_once()
        discover.assert_called_once()
        assert data["tool_count"] == 2
        assert data["reconnected"] == ["srv"]


class TestAdminConfig:
    @pytest.mark.asyncio
    async def test_unknown_key_rejected(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/admin/config", json={"nope": 1})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_invalid_json_rejected(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/admin/config", data=b"not json")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_model_apply(self, adapter):
        from hermes_cli.config import load_config

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/admin/config",
                json={"model": {"default": "moonshotai/kimi-k2.6", "provider": "openrouter"}},
            )
            assert resp.status == 200
            assert (await resp.json())["applied"]["model"] == "moonshotai/kimi-k2.6"

        model_cfg = load_config()["model"]
        assert model_cfg["default"] == "moonshotai/kimi-k2.6"
        assert model_cfg["provider"] == "openrouter"
        assert "model" not in model_cfg  # stale flat key must not shadow 'default'

    @pytest.mark.asyncio
    async def test_reasoning_effort_apply_clear_and_validate(self, adapter):
        from hermes_cli.config import load_config

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/admin/config", json={"agent": {"reasoning_effort": "high"}})
            assert resp.status == 200
            assert (await resp.json())["applied"]["reasoning_effort"] == "high"
            assert load_config()["agent"]["reasoning_effort"] == "high"

            # "none" is a valid level (disables thinking), not a clear.
            resp = await cli.post("/admin/config", json={"agent": {"reasoning_effort": "none"}})
            assert resp.status == 200
            assert load_config()["agent"]["reasoning_effort"] == "none"

            resp = await cli.post("/admin/config", json={"agent": {"reasoning_effort": "bogus"}})
            assert resp.status == 400

            # null clears back to the default.
            resp = await cli.post("/admin/config", json={"agent": {"reasoning_effort": None}})
            assert resp.status == 200
            assert "reasoning_effort" not in (load_config().get("agent") or {})

    @pytest.mark.asyncio
    async def test_mcp_addition_uses_incremental_discovery(self, adapter):
        """Adding a server must not drop live connections (no full shutdown)."""
        from hermes_cli.config import load_config

        app = _create_app(adapter)
        shutdown = MagicMock()
        discover = MagicMock(return_value=["mcp_notion_search"])
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MCP_MOD}.shutdown_mcp_servers", shutdown), \
                 patch(f"{_MCP_MOD}.discover_mcp_tools", discover):
                resp = await cli.post(
                    "/admin/config",
                    json={"mcp_servers": {"notion": {
                        "url": "https://mcp.notion.example/",
                        "headers": {"Authorization": "Bearer tok"},
                    }}},
                )
                assert resp.status == 200
                applied = (await resp.json())["applied"]

        shutdown.assert_not_called()
        discover.assert_called_once()
        assert applied["mcp"]["added"] == ["notion"]
        entry = load_config()["mcp_servers"]["notion"]
        assert entry["url"] == "https://mcp.notion.example/"
        assert entry["headers"]["Authorization"] == "Bearer tok"

    @pytest.mark.asyncio
    async def test_mcp_removal_triggers_full_reload(self, adapter):
        from hermes_cli.config import load_config

        _save_servers({"notion": {"url": "https://mcp.notion.example/"}})
        app = _create_app(adapter)
        shutdown = MagicMock()
        discover = MagicMock(return_value=[])
        async with TestClient(TestServer(app)) as cli:
            with patch.dict(f"{_MCP_MOD}._servers", {}, clear=True), \
                 patch(f"{_MCP_MOD}.shutdown_mcp_servers", shutdown), \
                 patch(f"{_MCP_MOD}.discover_mcp_tools", discover):
                resp = await cli.post("/admin/config", json={"mcp_servers": {"notion": None}})
                assert resp.status == 200
                applied = (await resp.json())["applied"]

        shutdown.assert_called_once()
        assert applied["mcp_removed"] == ["notion"]
        assert "notion" not in (load_config().get("mcp_servers") or {})

    @pytest.mark.asyncio
    async def test_ringo_ie_removal_refused(self, adapter):
        from hermes_cli.config import load_config

        _save_servers({"ringo_ie": {"url": "https://ie.example/mcp/"}})
        app = _create_app(adapter)
        shutdown = MagicMock()
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MCP_MOD}.shutdown_mcp_servers", shutdown):
                resp = await cli.post("/admin/config", json={"mcp_servers": {"ringo_ie": None}})
                assert resp.status == 200
                applied = (await resp.json())["applied"]

        shutdown.assert_not_called()
        assert "mcp_removed" not in applied
        assert "ringo_ie" in load_config()["mcp_servers"]

    @pytest.mark.asyncio
    async def test_toolsets_apply(self, adapter):
        from hermes_cli.config import load_config

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/admin/config", json={"toolsets": ["web", "skills"]})
            assert resp.status == 200
        assert load_config()["platform_toolsets"]["api_server"] == ["web", "skills"]

    @pytest.mark.asyncio
    async def test_env_apply_writes_and_reloads(self, adapter):
        app = _create_app(adapter)
        save_env = MagicMock()
        reload_env = MagicMock()
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_cli.config.save_env_value", save_env), \
                 patch("hermes_cli.config.reload_env", reload_env):
                resp = await cli.post(
                    "/admin/config",
                    json={"env": {"TAVILY_API_KEY": "tk", "OLD_KEY": None}},
                )
                assert resp.status == 200

        save_env.assert_any_call("TAVILY_API_KEY", "tk")
        save_env.assert_any_call("OLD_KEY", "")  # blank, not delete
        reload_env.assert_called_once()

    @pytest.mark.asyncio
    async def test_env_denylist_maps_to_400(self, adapter):
        """save_env_value's ValueError (denylisted name) surfaces as 400."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/admin/config", json={"env": {"PATH": "/evil"}})
            assert resp.status == 400

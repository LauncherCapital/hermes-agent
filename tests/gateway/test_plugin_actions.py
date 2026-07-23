import uuid
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.event_ingress import write_project_marker
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.plugins import PluginActionUnavailable


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-api-key-long-enough"})
    )


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/plugin-actions", adapter._handle_plugin_action)
    return app


def _body(project_id: str) -> dict:
    return {
        "action": "fixture.reconcile",
        "project_id": project_id,
        "request_id": str(uuid.uuid4()),
        "payload": {"revision": 7},
    }


@pytest.mark.asyncio
async def test_plugin_action_requires_auth_and_exact_claimed_project(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    dispatch = AsyncMock(return_value={"status": "ready"})

    with patch("hermes_cli.plugins.invoke_plugin_action", dispatch):
        async with TestClient(TestServer(_app(_adapter()))) as client:
            unauthorized = await client.post(
                "/v1/plugin-actions",
                json=_body(project_id),
            )
            wrong_project = await client.post(
                "/v1/plugin-actions",
                json=_body(str(uuid.uuid4())),
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )

    assert unauthorized.status == 401
    assert wrong_project.status == 403
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_action_dispatches_canonical_request(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    body = _body(project_id)
    dispatch = AsyncMock(return_value={"status": "ready", "revision": 7})

    with patch("hermes_cli.plugins.invoke_plugin_action", dispatch):
        async with TestClient(TestServer(_app(_adapter()))) as client:
            response = await client.post(
                "/v1/plugin-actions",
                json=body,
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            payload = await response.json()

    assert response.status == 200
    assert payload == {
        "ok": True,
        "action": body["action"],
        "request_id": body["request_id"],
        "result": {"status": "ready", "revision": 7},
    }
    dispatch.assert_awaited_once_with(body["action"], request=body)


@pytest.mark.asyncio
async def test_plugin_action_returns_explicit_unavailable_and_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    body = _body(project_id)

    async with TestClient(TestServer(_app(_adapter()))) as client:
        with patch(
            "hermes_cli.plugins.invoke_plugin_action",
            AsyncMock(side_effect=PluginActionUnavailable(body["action"])),
        ):
            unavailable = await client.post(
                "/v1/plugin-actions",
                json=body,
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            unavailable_payload = await unavailable.json()
        with patch(
            "hermes_cli.plugins.invoke_plugin_action",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            failed = await client.post(
                "/v1/plugin-actions",
                json=body,
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            failed_payload = await failed.json()

    assert unavailable.status == 503
    assert unavailable_payload["error"]["code"] == "plugin_action_unavailable"
    assert failed.status == 500
    assert failed_payload["error"]["code"] == "plugin_action_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"action": "Fixture Reconcile"}, "invalid_action"),
        ({"action": "reconcile"}, "invalid_action"),
        ({"request_id": ""}, "invalid_request_id"),
        ({"request_id": 7}, "invalid_request_id"),
        ({"payload": []}, "invalid_payload"),
    ],
)
async def test_plugin_action_rejects_invalid_contract(
    tmp_path, monkeypatch, change, code
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    body = {**_body(project_id), **change}

    async with TestClient(TestServer(_app(_adapter()))) as client:
        response = await client.post(
            "/v1/plugin-actions",
            json=body,
            headers={"Authorization": "Bearer test-api-key-long-enough"},
        )
        payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == code

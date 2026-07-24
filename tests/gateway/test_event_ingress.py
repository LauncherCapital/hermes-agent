import base64
import hashlib
import json
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from gateway.config import PlatformConfig
from gateway.event_ingress import (
    EventReplayGuard,
    EventIngressError,
    canonical_request,
    read_project_marker,
    write_project_marker,
)
from gateway.platforms.api_server import APIServerAdapter


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-api-key-long-enough"})
    )


def _app(adapter: APIServerAdapter, *, admin: bool = False) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/events", adapter._handle_events)
    app.router.add_post(
        "/v1/message-store/query", adapter._handle_message_store_query
    )
    if admin:
        app.router.add_post("/admin/config", adapter._handle_admin_config)
    return app


@pytest.mark.asyncio
async def test_message_store_query_requires_api_auth_and_returns_plugin_result():
    adapter = _adapter()
    hook = AsyncMock(
        return_value=[
            {
                "coverage_complete": True,
                "covered_since": "2026-07-20T00:00:00+00:00",
                "messages": [],
            }
        ]
    )
    body = {
        "operation": "recent_activity",
        "start": "2026-07-20T00:00:00+00:00",
        "end": "2026-07-21T00:00:00+00:00",
        "allowed_source_ids": [],
    }

    with patch("hermes_cli.plugins.invoke_hook_async", hook):
        async with TestClient(TestServer(_app(adapter))) as client:
            unauthorized = await client.post("/v1/message-store/query", json=body)
            response = await client.post(
                "/v1/message-store/query",
                json=body,
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            payload = await response.json()

    assert unauthorized.status == 401
    assert response.status == 200
    assert payload["coverage_complete"] is True
    hook.assert_awaited_once_with("message_store_query", request=body)


@pytest.mark.asyncio
async def test_message_store_query_fails_closed_without_plugin_handler():
    adapter = _adapter()
    with patch(
        "hermes_cli.plugins.invoke_hook_async", AsyncMock(return_value=[])
    ):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/message-store/query",
                json={"operation": "recent_activity"},
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "message_store_unavailable"


def _signed_request(private_key, key_id, project_id, envelope, *, timestamp=None):
    body = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(body).hexdigest()
    ts = str(timestamp if timestamp is not None else int(time.time()))
    signature = private_key.sign(
        canonical_request(timestamp=ts, project_id=project_id, body_sha256=digest)
    )
    return body, {
        "Content-Type": "application/json",
        "X-Ringo-Key-Id": key_id,
        "X-Ringo-Timestamp": ts,
        "X-Ringo-Content-SHA256": digest,
        "X-Ringo-Signature": base64.b64encode(signature).decode(),
    }


def test_project_marker_refuses_volume_reassignment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first_project = str(uuid.uuid4())
    write_project_marker(first_project)

    with pytest.raises(EventIngressError) as exc_info:
        write_project_marker(str(uuid.uuid4()))

    assert exc_info.value.status == 409
    assert exc_info.value.code == "project_already_claimed"
    assert read_project_marker()["project_id"] == first_project


@pytest.mark.asyncio
async def test_signed_event_dispatch_and_identical_replay_are_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    key_id = "ie-signing-v1"
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    write_project_marker(project_id, event_verifiers={key_id: public_pem})
    envelope = {
        "schema_version": 1,
        "delivery_id": str(uuid.uuid4()),
        "sequence": 1,
        "project_id": project_id,
        "provider": "fixture",
        "workspace_id": "W1",
        "event_type": "message.created",
    }
    body, headers = _signed_request(
        private_key, key_id, project_id, envelope
    )
    hook = AsyncMock(return_value=[{"status": "accepted", "sequence": 1}])

    with patch("hermes_cli.plugins.invoke_hook_async", hook):
        async with TestClient(TestServer(_app(_adapter()))) as client:
            first = await client.post("/v1/events", data=body, headers=headers)
            replay = await client.post("/v1/events", data=body, headers=headers)
            first_payload = await first.json()
            replay_payload = await replay.json()

    assert first.status == 200
    assert first_payload["ok"] is True
    assert replay.status == 200
    assert replay_payload["replayed"] is True
    hook.assert_awaited_once()


def test_replay_guard_drops_uncommitted_reservation_after_restart(tmp_path):
    path = tmp_path / "event_replays.db"
    delivery_id = str(uuid.uuid4())
    digest = "a" * 64

    first = EventReplayGuard(path)
    assert first.reserve(delivery_id, digest) == "new"
    assert first.reserve(delivery_id, digest) == "pending"

    restarted = EventReplayGuard(path)
    assert restarted.reserve(delivery_id, digest) == "new"


def test_replay_guard_keeps_committed_delivery_after_restart(tmp_path):
    path = tmp_path / "event_replays.db"
    delivery_id = str(uuid.uuid4())
    digest = "b" * 64

    first = EventReplayGuard(path)
    assert first.reserve(delivery_id, digest) == "new"
    first.commit(delivery_id, digest)

    restarted = EventReplayGuard(path)
    assert restarted.reserve(delivery_id, digest) == "duplicate"


@pytest.mark.asyncio
async def test_signed_event_rejects_stale_timestamp_and_wrong_project(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    key_id = "ie-signing-v1"
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    write_project_marker(project_id, event_verifiers={key_id: public_pem})

    stale_envelope = {
        "delivery_id": "stale",
        "project_id": project_id,
        "sequence": 1,
    }
    stale_body, stale_headers = _signed_request(
        private_key,
        key_id,
        project_id,
        stale_envelope,
        timestamp=int(time.time()) - 1000,
    )
    other_project = str(uuid.uuid4())
    wrong_envelope = {
        "delivery_id": "wrong",
        "project_id": other_project,
        "sequence": 1,
    }
    wrong_body, wrong_headers = _signed_request(
        private_key, key_id, other_project, wrong_envelope
    )

    async with TestClient(TestServer(_app(_adapter()))) as client:
        stale = await client.post("/v1/events", data=stale_body, headers=stale_headers)
        wrong = await client.post("/v1/events", data=wrong_body, headers=wrong_headers)
        stale_payload = await stale.json()
        wrong_payload = await wrong.json()

    assert stale.status == 401
    assert stale_payload["error"]["code"] == "stale_timestamp"
    assert wrong.status == 403
    assert wrong_payload["error"]["code"] == "project_mismatch"


@pytest.mark.asyncio
async def test_admin_claim_writes_marker_and_notifies_service(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    hook = AsyncMock(return_value=[{"status": "ready"}])

    with patch("hermes_cli.plugins.invoke_hook_async", hook):
        async with TestClient(TestServer(_app(_adapter(), admin=True))) as client:
            response = await client.post(
                "/admin/config",
                json={"project": {"id": project_id}},
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )

    assert response.status == 200
    assert read_project_marker()["project_id"] == project_id
    hook.assert_awaited_once_with(
        "project_claimed",
        project_id=project_id,
        active_key_version=1,
    )


@pytest.mark.asyncio
async def test_gap_response_releases_replay_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    key_id = "ie-signing-v1"
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    write_project_marker(project_id, event_verifiers={key_id: public_pem})
    envelope = {
        "schema_version": 1,
        "delivery_id": str(uuid.uuid4()),
        "sequence": 2,
        "project_id": project_id,
        "provider": "fixture",
        "workspace_id": "W1",
    }
    body, headers = _signed_request(private_key, key_id, project_id, envelope)
    hook = AsyncMock(
        return_value=[{"status": "gap_detected", "expected_sequence": 1}]
    )

    with patch("hermes_cli.plugins.invoke_hook_async", hook):
        async with TestClient(TestServer(_app(_adapter()))) as client:
            first = await client.post("/v1/events", data=body, headers=headers)
            replay = await client.post("/v1/events", data=body, headers=headers)
            first_payload = await first.json()

    assert first.status == 409
    assert replay.status == 409
    assert first_payload["error"]["code"] == "delivery_gap"
    assert first_payload["error"]["expected_sequence"] == 1
    assert hook.await_count == 2


@pytest.mark.asyncio
async def test_conflict_response_releases_replay_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    key_id = "ie-signing-v1"
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    write_project_marker(project_id, event_verifiers={key_id: public_pem})
    envelope = {
        "schema_version": 1,
        "delivery_id": str(uuid.uuid4()),
        "sequence": 1,
        "project_id": project_id,
        "provider": "fixture",
        "workspace_id": "W1",
    }
    body, headers = _signed_request(private_key, key_id, project_id, envelope)
    hook = AsyncMock(
        return_value=[{"status": "conflict", "code": "delivery_conflict"}]
    )

    with patch("hermes_cli.plugins.invoke_hook_async", hook):
        async with TestClient(TestServer(_app(_adapter()))) as client:
            first = await client.post("/v1/events", data=body, headers=headers)
            replay = await client.post("/v1/events", data=body, headers=headers)
            first_payload = await first.json()

    assert first.status == 409
    assert replay.status == 409
    assert first_payload["error"]["code"] == "delivery_conflict"
    assert hook.await_count == 2

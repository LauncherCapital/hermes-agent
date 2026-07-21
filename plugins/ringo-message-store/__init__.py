"""Project-local message store service plugin."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.request

from gateway.event_ingress import read_project_marker

from .crypto import build_recovery_registration, decrypt_delivery_envelope
from .store import MessageStore


_lock = threading.Lock()
_store_instance: MessageStore | None = None
_registration_health: dict = {"status": "pending"}


def _store() -> MessageStore | None:
    global _store_instance
    marker = read_project_marker()
    if marker is None:
        return None
    project_id = marker["project_id"]
    key_version = int(marker.get("active_key_version") or 1)
    with _lock:
        if (
            _store_instance is None
            or _store_instance.project_id != project_id
            or _store_instance.key_version != key_version
        ):
            _store_instance = MessageStore(project_id, key_version=key_version)
        return _store_instance


async def _on_ingress_event(
    *,
    event: dict,
    body_sha256: str,
    **_: object,
) -> dict:
    store = _store()
    if store is None:
        return {"status": "unclaimed"}
    if isinstance(event.get("encryption"), dict):
        try:
            event = await asyncio.to_thread(decrypt_delivery_envelope, event)
        except ValueError:
            return {"status": "conflict", "code": "invalid_ciphertext"}
    try:
        return await asyncio.to_thread(store.record_envelope, event, body_sha256)
    except ValueError:
        return {"status": "conflict", "code": "delivery_conflict"}


async def _on_project_claimed(*, project_id: str, **_: object) -> dict:
    store = await asyncio.to_thread(_store)
    if store is None or store.project_id != project_id:
        raise RuntimeError("project claim did not initialize message store")
    await asyncio.to_thread(_register_project_key, project_id)
    return {"status": "ready", "key_registration": dict(_registration_health)}


def _register_project_key(project_id: str) -> dict:
    global _registration_health
    registration = build_recovery_registration(project_id)
    if registration is None:
        _registration_health = {"status": "pending", "reason": "recovery_key_unavailable"}
        return _registration_health
    base_url = (os.environ.get("RINGO_IE_MCP_URL") or "").strip().rstrip("/")
    api_key = (os.environ.get("RINGO_IE_MCP_KEY") or "").strip()
    if not base_url or not api_key:
        _registration_health = {"status": "pending", "reason": "control_channel_unavailable"}
        return _registration_health
    if base_url.endswith("/mcp"):
        base_url = base_url[:-4]
    request = urllib.request.Request(
        base_url + "/api/v1/message-store/keys/register",
        data=json.dumps(registration, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read())
        _registration_health = {
            "status": "registered",
            "version": int(payload.get("version") or registration["version"]),
            "public_key_sha256": payload.get("public_key_sha256"),
        }
    except Exception as exc:
        _registration_health = {
            "status": "error",
            "error": type(exc).__name__,
        }
    return _registration_health


def _health_report(**_: object) -> dict:
    store = _store()
    if store is None:
        return {"name": "ringo_message_store", "status": "unclaimed"}
    return {**store.health(), "key_registration": dict(_registration_health)}


async def _on_message_store_query(*, request: dict, **_: object) -> dict:
    store = _store()
    if store is None:
        return {"coverage_complete": False, "reason": "unclaimed", "messages": []}
    if str(request.get("project_id") or "") != store.project_id:
        raise ValueError("message store project mismatch")
    result = await asyncio.to_thread(store.query, request)
    result["project_id"] = store.project_id
    result["acl_version"] = str(request.get("acl_version") or "") or None
    return result


def register(ctx) -> None:
    ctx.register_hook("ingress_event", _on_ingress_event)
    ctx.register_hook("project_claimed", _on_project_claimed)
    ctx.register_hook("health_report", _health_report)
    ctx.register_hook("message_store_query", _on_message_store_query)
    # A cold-provisioned instance already has its immutable project marker by
    # the time bundled plugins are loaded.  Initialising here ensures the
    # project key exists immediately after claim, while an unclaimed warm-pool
    # instance still remains completely keyless.
    store = _store()
    if store is not None:
        _register_project_key(store.project_id)

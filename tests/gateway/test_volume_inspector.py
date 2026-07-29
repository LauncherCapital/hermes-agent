import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.event_ingress import write_project_marker
from gateway.platforms.api_server import APIServerAdapter
from gateway.volume_inspector import (
    VolumeInspectorError,
    inspect_volume,
    read_preview_file,
    validate_preview_target,
)


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-api-key-long-enough"})
    )


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/volume/inspect", adapter._handle_volume_inspect)
    app.router.add_post("/v1/volume/preview", adapter._handle_volume_preview)
    return app


def _identity(project_id: str) -> dict[str, str]:
    return {"project_id": project_id, "agent_id": str(uuid.uuid4())}


def test_tree_classifies_real_canonical_and_legacy_paths_only(tmp_path):
    canonical = tmp_path / "skills/channels/C1/SKILL.md"
    legacy = tmp_path / "slack/T1/channel/C1/MEMORY.md"
    canonical.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    canonical.write_text("encrypted envelope")
    legacy.write_text("legacy")

    result = inspect_volume(tmp_path)
    by_path = {node["path"]: node for node in result["nodes"]}

    assert result["root"] == "HERMES_HOME"
    assert by_path["skills/channels/C1/SKILL.md"]["provenance"] == "canonical"
    assert by_path["skills/channels/C1/SKILL.md"]["previewable"] is True
    assert by_path["slack/T1/channel/C1/MEMORY.md"]["provenance"] == "legacy"

    no_legacy_root = tmp_path / "without-legacy"
    (no_legacy_root / "skills").mkdir(parents=True)
    paths = {node["path"] for node in inspect_volume(no_legacy_root)["nodes"]}
    assert not any(path == "slack" or path.startswith("slack/") for path in paths)


def test_sensitive_directories_are_opaque_but_session_files_are_browseable(tmp_path):
    secret = tmp_path / "credentials/private/service-key.json"
    secret.parent.mkdir(parents=True)
    secret.write_text("never enumerate me")
    session = tmp_path / "sessions/raw.json"
    session.parent.mkdir(parents=True)
    session.write_text('{"messages": []}')

    result = inspect_volume(tmp_path)
    by_path = {node["path"]: node for node in result["nodes"]}

    assert by_path["credentials"]["status"] == "locked"
    assert not any(path.startswith("credentials/") for path in by_path)
    assert by_path["sessions/raw.json"]["previewable"] is True


def test_depth_and_node_caps_are_enforced(tmp_path):
    deep = tmp_path
    for part in ("a", "b", "c", "d"):
        deep = deep / part
        deep.mkdir()
    for index in range(10):
        (tmp_path / f"file-{index}.txt").write_text(str(index))

    depth_limited = inspect_volume(tmp_path, max_depth=2)
    node_limited = inspect_volume(tmp_path, max_nodes=3)

    assert depth_limited["truncated"] is True
    assert all(node["depth"] <= 2 for node in depth_limited["nodes"])
    assert node_limited["truncated"] is True
    assert len(node_limited["nodes"]) == 3


@pytest.mark.parametrize(
    "path",
    [
        "../.env",
        "skills/../.env",
        "/opt/data/.env",
        ".env",
        "credentials/project.json",
        "state/session-token.json",
        "data.sqlite",
        "artifacts/image.png",
    ],
)
def test_preview_denies_traversal_sensitive_and_non_text_files(
    tmp_path,
    path,
):
    with pytest.raises(VolumeInspectorError):
        validate_preview_target(path, tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "state/project.json",
        "sessions/raw.json",
        "skills/users/U1/SKILL.md",
        "slack/T1/channel/C1/MEMORY.md",
        "cron/output/latest.jsonl",
    ],
)
def test_preview_accepts_safe_text_files(tmp_path, path):
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("readable")

    resolved, channel_id = validate_preview_target(path, tmp_path)

    assert resolved == target
    assert channel_id is None


def test_read_preview_file_returns_bounded_plain_text(tmp_path):
    target = tmp_path / "cron/output/latest.log"
    target.parent.mkdir(parents=True)
    target.write_text("가" * 20_000)

    result = read_preview_file(target, "cron/output/latest.log")

    assert result["path"] == "cron/output/latest.log"
    assert result["encoding"] == "utf-8"
    assert len(result["content"].encode("utf-8")) <= 32_000
    assert result["truncated"] is True


def test_preview_denies_symlink_in_any_component(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("outside")
    channels = tmp_path / "skills/channels"
    channels.mkdir(parents=True)
    (channels / "C1").symlink_to(outside, target_is_directory=True)

    with pytest.raises(VolumeInspectorError, match="symlink"):
        validate_preview_target("skills/channels/C1/SKILL.md", tmp_path)


@pytest.mark.asyncio
async def test_runtime_volume_endpoints_require_control_key_and_claimed_project(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    canonical = tmp_path / "skills/channels/C1/SKILL.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("encrypted envelope")
    identity = _identity(project_id)
    body = {**identity, "path": "skills/channels/C1/SKILL.md"}

    preview = AsyncMock(
        return_value={
            "path": body["path"],
            "content": "safe plaintext",
            "encoding": "utf-8",
            "truncated": False,
        }
    )
    with patch("hermes_cli.plugins.invoke_plugin_action", preview):
        async with TestClient(TestServer(_app(_adapter()))) as client:
            unauthorized = await client.post(
                "/v1/volume/inspect",
                json=identity,
            )
            wrong_project = await client.post(
                "/v1/volume/inspect",
                json={**identity, "project_id": str(uuid.uuid4())},
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            caller_root = await client.post(
                "/v1/volume/inspect",
                json={**_identity(project_id), "root": "/tmp/other"},
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            allowed = await client.post(
                "/v1/volume/preview",
                json=body,
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            payload = await allowed.json()

    assert unauthorized.status == 401
    assert wrong_project.status == 403
    assert caller_root.status == 400
    assert allowed.status == 200
    assert payload["content"] == "safe plaintext"
    request = preview.await_args.kwargs["request"]
    assert request["project_id"] == project_id
    assert request["payload"]["agent_id"] == body["agent_id"]


@pytest.mark.asyncio
async def test_runtime_preview_rejects_locked_path_before_plugin_dispatch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    preview = AsyncMock()

    with patch("hermes_cli.plugins.invoke_plugin_action", preview):
        async with TestClient(TestServer(_app(_adapter()))) as client:
            response = await client.post(
                "/v1/volume/preview",
                json={**_identity(project_id), "path": ".env"},
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )

    assert response.status == 403
    preview.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_preview_reads_safe_text_without_plugin_dispatch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    output = tmp_path / "cron/output/latest.log"
    output.parent.mkdir(parents=True)
    output.write_text("completed")
    preview = AsyncMock()

    with patch("hermes_cli.plugins.invoke_plugin_action", preview):
        async with TestClient(TestServer(_app(_adapter()))) as client:
            response = await client.post(
                "/v1/volume/preview",
                json={
                    **_identity(project_id),
                    "path": "cron/output/latest.log",
                },
                headers={"Authorization": "Bearer test-api-key-long-enough"},
            )
            payload = await response.json()

    assert response.status == 200
    assert payload["content"] == "completed"
    preview.assert_not_awaited()

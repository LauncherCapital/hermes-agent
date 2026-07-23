import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


PLUGIN_DIR = Path(__file__).parents[2] / "plugins" / "ringo-channel-memory"
PROJECT_ID = "11111111-1111-1111-1111-111111111111"
AGENT_ID = "22222222-2222-2222-2222-222222222222"
SESSION_ID = "ringo_slack_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _load_sync_module():
    package_name = "test_ringo_channel_memory_plugin"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    assert package_spec.loader is not None
    package_spec.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.sync")


def _snapshot(content="# Project\n\nStable API contract.\n", revision="r1"):
    return {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "agent_id": AGENT_ID,
        "revision": revision,
        "profile": {
            "provider": "slack",
            "workspace_id": "T1",
            "channel_id": "C1",
            "index_markdown": "# Channel memory index — #general\n",
            "documents": [
                {
                    "name": "project-context.md",
                    "type": "project",
                    "description": "Project context",
                    "summary": "Stable API contract",
                    "content": content,
                    "metadata": {"source": "slack"},
                }
            ],
        },
    }


def _request():
    return {
        "project_id": PROJECT_ID,
        "payload": {
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "channel_id": "C1",
            "session_id": SESSION_ID,
        },
    }


def test_prepare_materializes_and_binds_exact_channel(tmp_path):
    sync = _load_sync_module()
    service = sync.ChannelMemorySyncService(tmp_path)
    service.fetch_snapshot = lambda **kwargs: _snapshot()

    result = service.prepare(request=_request())
    root = tmp_path / "slack" / "T1" / "channel" / "C1"
    manifest = json.loads(service.manifest_path.read_text(encoding="utf-8"))

    assert result["status"] == "ready"
    assert (root / "project-context.md").read_text() == _snapshot()["profile"][
        "documents"
    ][0]["content"]
    assert (root / "MEMORY.md").read_text().startswith("# Channel memory index")
    assert manifest["bindings"][SESSION_ID] == "T1:C1"


def test_tool_observer_requires_bound_session_and_exact_detail_path(tmp_path):
    sync = _load_sync_module()
    service = sync.ChannelMemorySyncService(tmp_path)
    service.fetch_snapshot = lambda **kwargs: _snapshot()
    service.prepare(request=_request())
    queued = []
    service.queue_sync = queued.append
    root = tmp_path / "slack" / "T1" / "channel" / "C1"

    for session_id, path in [
        ("other-session", root / "project-context.md"),
        (SESSION_ID, root / "MEMORY.md"),
        (SESSION_ID, tmp_path / "slack" / "T1" / "channel" / "C2" / "notes.md"),
        (SESSION_ID, root / "project-context.md"),
    ]:
        service.observe_tool(
            session_id=session_id,
            tool_name="patch",
            args={"path": str(path)},
            result={"success": True},
            status="ok",
        )

    assert queued == ["T1:C1"]


def test_bound_curator_session_is_confined_to_exact_channel_files(tmp_path):
    sync = _load_sync_module()
    service = sync.ChannelMemorySyncService(tmp_path)
    service.fetch_snapshot = lambda **kwargs: _snapshot()
    service.prepare(request=_request())
    root = tmp_path / "slack" / "T1" / "channel" / "C1"

    assert (
        service.authorize_tool(
            session_id=SESSION_ID,
            tool_name="read_file",
            args={"path": str(root / "MEMORY.md")},
        )
        is None
    )
    assert (
        service.authorize_tool(
            session_id=SESSION_ID,
            tool_name="read_file",
            args={"path": str(root / "project-context.md")},
        )
        is None
    )
    assert (
        service.authorize_tool(
            session_id=SESSION_ID,
            tool_name="patch",
            args={
                "mode": "replace",
                "path": str(root / "project-context.md"),
            },
        )
        is None
    )
    for tool_name, args in [
        ("send_message", {}),
        ("terminal", {"command": "true"}),
        ("write_file", {"path": str(root / "MEMORY.md")}),
        ("read_file", {"path": str(tmp_path / "other.md")}),
        ("patch", {"mode": "patch", "patch": "*** Begin Patch"}),
    ]:
        result = service.authorize_tool(
            session_id=SESSION_ID,
            tool_name=tool_name,
            args=args,
        )
        assert result and result["action"] == "block"

    assert (
        service.authorize_tool(
            session_id="unbound-session",
            tool_name="terminal",
            args={"command": "true"},
        )
        is None
    )


def test_changed_detail_pushes_complete_document_set_without_model_json(
    tmp_path,
    monkeypatch,
):
    sync = _load_sync_module()
    monkeypatch.setenv("RINGO_IE_MCP_URL", "https://ie.example.com/mcp")
    monkeypatch.setenv("RINGO_IE_MCP_KEY", "secret")
    service = sync.ChannelMemorySyncService(tmp_path)
    service.fetch_snapshot = lambda **kwargs: _snapshot()
    service.prepare(request=_request())
    path = tmp_path / "slack" / "T1" / "channel" / "C1" / "project-context.md"
    path.write_text("# Project\n\nThe API contract is v2.\n", encoding="utf-8")
    sent = {}

    def accept(url, *, method="GET", body=None):
        sent.update(body)
        return {
            "status": "accepted",
            "revision": "r2",
            "profile": {
                **_snapshot()["profile"],
                "index_markdown": "# Updated index\n",
            },
        }

    service._request_json = accept
    result = service.push_local("T1:C1")

    assert result["revision"] == "r2"
    assert sent["base_revision"] == "r1"
    assert sent["documents"][0]["name"] == "project-context.md"
    assert sent["documents"][0]["content"].strip().endswith(
        "The API contract is v2."
    )
    assert sent["documents"][0]["type"] == "project"


def test_restart_requeues_only_dirty_bound_channels(tmp_path):
    sync = _load_sync_module()
    service = sync.ChannelMemorySyncService(tmp_path)
    service.fetch_snapshot = lambda **kwargs: _snapshot()
    service.prepare(request=_request())
    root = tmp_path / "slack" / "T1" / "channel" / "C1"
    (root / "project-context.md").write_text("# Changed\n", encoding="utf-8")
    queued = []
    service.queue_sync = queued.append

    assert service.resume_dirty() == 1
    assert queued == ["T1:C1"]


def test_prepare_rejects_parent_path_components(tmp_path):
    sync = _load_sync_module()
    service = sync.ChannelMemorySyncService(tmp_path)
    request = _request()
    request["payload"]["workspace_id"] = ".."

    try:
        service.prepare(request=request)
    except sync.ChannelMemorySyncError as exc:
        assert "invalid workspace_id" in str(exc)
    else:
        raise AssertionError("parent path component was accepted")


def test_channel_root_symlink_cannot_escape_volume(tmp_path):
    sync = _load_sync_module()
    outside = tmp_path / "outside"
    outside.mkdir()
    slack = tmp_path / "slack"
    slack.mkdir()
    try:
        os.symlink(outside, slack / "T1")
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    service = sync.ChannelMemorySyncService(tmp_path)
    service.fetch_snapshot = lambda **kwargs: _snapshot()

    try:
        service.prepare(request=_request())
    except sync.ChannelMemorySyncError as exc:
        assert "escapes volume" in str(exc)
    else:
        raise AssertionError("symlinked channel root escaped the volume")

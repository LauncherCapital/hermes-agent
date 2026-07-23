import importlib.util
import json
import sys
from pathlib import Path

import pytest


PLUGIN_DIR = Path(__file__).parents[2] / "plugins" / "ringo-profile-sync"
PROJECT_ID = "11111111-1111-1111-1111-111111111111"
AGENT_ID = "22222222-2222-2222-2222-222222222222"
PRINCIPAL_ID = "33333333-3333-3333-3333-333333333333"


def _load_sync_module():
    package_name = "test_ringo_profile_sync_plugin"
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


def _document(
    *,
    document_type="person_character",
    content="# Person\n\n## Preference\n- concise\n\n## Working style\n- None\n",
    revision="r1",
):
    if document_type.startswith("person_"):
        suffix = "profile.md" if document_type == "person_profile" else "CHARACTER.md"
        name = "profile" if document_type == "person_profile" else "character"
        target = {"principal_id": PRINCIPAL_ID}
        path = f"profiles/{PRINCIPAL_ID}/{suffix}"
        document_id = f"person:{PRINCIPAL_ID}:{name}"
    else:
        suffix = "profile.md" if document_type == "organization_profile" else "ORGANIZATION.md"
        name = "profile" if document_type == "organization_profile" else "character"
        target = {"provider": "slack", "workspace_id": "T123"}
        path = f"organizations/slack/T123/{suffix}"
        document_id = f"organization:slack:T123:{name}"
    entries = (
        [{"kind": "preference", "value": "concise"}]
        if document_type == "person_character"
        else []
    )
    return {
        "id": document_id,
        "type": document_type,
        "path": path,
        "editable": document_type.endswith("_character"),
        "revision": revision,
        "content": content,
        "target": target,
        "entries": entries,
    }


def _snapshot(*documents, revision="snapshot-1"):
    return {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "agent_id": AGENT_ID,
        "revision": revision,
        "documents": list(documents),
    }


def test_snapshot_writes_skips_and_removes_only_managed_documents(tmp_path):
    sync = _load_sync_module()
    service = sync.ProfileSyncService(tmp_path)
    editable = _document()
    readonly = _document(
        document_type="person_profile",
        content="# Generated profile\n",
    )

    first = service.apply_snapshot(_snapshot(editable, readonly))
    character = tmp_path / editable["path"]
    first_mtime = character.stat().st_mtime_ns
    notes = character.parent / "notes.md"
    notes.write_text("local notes", encoding="utf-8")
    second = service.apply_snapshot(_snapshot(editable, readonly))
    second_mtime = character.stat().st_mtime_ns
    removed = service.apply_snapshot(_snapshot(revision="snapshot-2"))

    assert first["changed_files"] == 2
    assert second["changed_files"] == 0
    assert second_mtime == first_mtime
    assert removed["removed_files"] == 2
    assert notes.read_text(encoding="utf-8") == "local notes"


def test_editable_document_pushes_structured_entries_and_updates_revision(
    tmp_path,
    monkeypatch,
):
    sync = _load_sync_module()
    monkeypatch.setenv("RINGO_IE_MCP_URL", "https://ie.example.com/mcp")
    monkeypatch.setenv("RINGO_IE_MCP_KEY", "secret")
    service = sync.ProfileSyncService(tmp_path)
    document = _document()
    service.apply_snapshot(_snapshot(document))
    path = tmp_path / document["path"]
    path.write_text(
        path.read_text(encoding="utf-8").replace("concise", "direct answers"),
        encoding="utf-8",
    )
    sent = {}

    def accept(url, *, method="GET", body=None):
        sent.update(body)
        return {"status": "accepted", "revision": "r2"}

    service._request_json = accept
    result = service.push_local(document["id"])
    manifest = json.loads(service.manifest_path.read_text(encoding="utf-8"))

    assert result["revision"] == "r2"
    assert sent["principal_id"] == PRINCIPAL_ID
    assert sent["entries"] == [
        {"kind": "preference", "value": "direct answers"}
    ]
    assert manifest["documents"][document["id"]]["revision"] == "r2"


def test_conflict_keeps_local_document_during_following_snapshot(tmp_path):
    sync = _load_sync_module()

    class FixtureService(sync.ProfileSyncService):
        def push_local(self, document_id):
            return {"status": "conflict"}

        def fetch_snapshot(self, agent_id):
            return _snapshot(
                _document(content="# Remote replacement\n", revision="r2"),
                revision="snapshot-2",
            )

    service = FixtureService(tmp_path)
    document = _document()
    service.apply_snapshot(_snapshot(document))
    path = tmp_path / document["path"]
    path.write_text("# Local edit\n", encoding="utf-8")

    result = service.reconcile(
        request={
            "project_id": PROJECT_ID,
            "payload": {"agent_id": AGENT_ID},
        }
    )

    assert result["conflicts"] == [document["id"]]
    assert path.read_text(encoding="utf-8") == "# Local edit\n"


def test_edit_during_push_is_not_overwritten_by_following_snapshot(
    tmp_path,
    monkeypatch,
):
    sync = _load_sync_module()
    monkeypatch.setenv("RINGO_IE_MCP_URL", "https://ie.example.com/mcp")
    monkeypatch.setenv("RINGO_IE_MCP_KEY", "secret")

    class FixtureService(sync.ProfileSyncService):
        def fetch_snapshot(self, agent_id):
            return _snapshot(
                _document(content="# Remote canonical\n", revision="r2"),
                revision="snapshot-2",
            )

    service = FixtureService(tmp_path)
    document = _document()
    service.apply_snapshot(_snapshot(document))
    path = tmp_path / document["path"]
    path.write_text("# First local edit\n", encoding="utf-8")
    service.queue_local_sync = lambda document_id: None

    def accept(url, *, method="GET", body=None):
        path.write_text("# Newer local edit\n", encoding="utf-8")
        return {"status": "accepted", "revision": "r2"}

    service._request_json = accept
    result = service.reconcile(
        request={
            "project_id": PROJECT_ID,
            "payload": {"agent_id": AGENT_ID},
        }
    )

    assert result["deferred"] == [document["id"]]
    assert path.read_text(encoding="utf-8") == "# Newer local edit\n"


def test_tool_observer_queues_only_known_editable_profile_path(tmp_path):
    sync = _load_sync_module()
    service = sync.ProfileSyncService(tmp_path)
    editable = _document()
    readonly = _document(document_type="person_profile", content="# Read only\n")
    service.apply_snapshot(_snapshot(editable, readonly))
    queued = []
    service.queue_local_sync = queued.append

    for document in (editable, readonly):
        service.observe_tool(
            tool_name="patch",
            args={"path": str(tmp_path / document["path"])},
            result=json.dumps({"success": True}),
            status="ok",
        )

    assert queued == [editable["id"]]


def test_manifest_binding_and_corruption_fail_closed(tmp_path):
    sync = _load_sync_module()
    service = sync.ProfileSyncService(tmp_path)
    service.apply_snapshot(_snapshot(_document()))

    with pytest.raises(sync.ProfileSyncError, match="project"):
        service.apply_snapshot(
            {
                **_snapshot(_document()),
                "project_id": "44444444-4444-4444-4444-444444444444",
            }
        )

    service.manifest_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(sync.ProfileSyncError, match="malformed"):
        service.apply_snapshot(_snapshot(_document()))

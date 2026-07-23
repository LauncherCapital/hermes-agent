import importlib.util
import json
import sys
from pathlib import Path

import pytest


PLUGIN_DIR = Path(__file__).parents[2] / "plugins" / "ringo-skill-sync"


def _load_sync_module():
    package_name = "test_ringo_skill_sync_plugin"
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


def _snapshot(*skills):
    return {
        "schema_version": 1,
        "project_id": "11111111-1111-1111-1111-111111111111",
        "agent_id": "22222222-2222-2222-2222-222222222222",
        "revision": "snapshot-1",
        "skills": list(skills),
    }


def _skill(name, *, body="Body", revision="r1", origin="user", files=None):
    return {
        "name": name,
        "description": f"{name} description",
        "body": body,
        "files": files or {},
        "origin": origin,
        "editable": origin == "agent",
        "revision": revision,
    }


def test_reconcile_writes_atomically_and_skips_identical_files(tmp_path):
    sync = _load_sync_module()
    service = sync.SkillSyncService(tmp_path)
    payload = _snapshot(
        _skill(
            "daily-brief",
            files={"scripts/render.py": "print('ok')\n"},
        )
    )

    first = service.apply_snapshot(payload)
    skill_md = tmp_path / "skills/daily-brief/SKILL.md"
    supporting = tmp_path / "skills/daily-brief/scripts/render.py"
    first_mtime = skill_md.stat().st_mtime_ns
    second = service.apply_snapshot(payload)

    assert first["changed_files"] == 2
    assert second["changed_files"] == 0
    assert skill_md.stat().st_mtime_ns == first_mtime
    assert "daily-brief description" in skill_md.read_text(encoding="utf-8")
    assert supporting.read_text(encoding="utf-8") == "print('ok')\n"
    assert not list(tmp_path.rglob("*.ringo-tmp"))


def test_reconcile_removes_only_paths_recorded_as_plugin_managed(tmp_path):
    sync = _load_sync_module()
    service = sync.SkillSyncService(tmp_path)
    service.apply_snapshot(
        _snapshot(
            _skill(
                "research",
                files={
                    "scripts/old.py": "old",
                    "references/keep.md": "managed",
                },
            )
        )
    )
    local = tmp_path / "skills/research/references/local.md"
    local.write_text("local", encoding="utf-8")

    result = service.apply_snapshot(
        {
            **_snapshot(
                _skill(
                    "research",
                    revision="r2",
                    files={"references/keep.md": "updated"},
                )
            ),
            "revision": "snapshot-2",
        }
    )

    assert result["removed_files"] == 1
    assert not (tmp_path / "skills/research/scripts/old.py").exists()
    assert local.read_text(encoding="utf-8") == "local"

    service.apply_snapshot(
        {
            **_snapshot(),
            "revision": "snapshot-3",
        }
    )
    assert not (tmp_path / "skills/research/SKILL.md").exists()
    assert not (tmp_path / "skills/research/references/keep.md").exists()
    assert local.exists()


def test_dirty_agent_owned_skill_is_pushed_before_remote_apply(tmp_path):
    sync = _load_sync_module()

    class FixtureService(sync.SkillSyncService):
        def __init__(self, home):
            super().__init__(home)
            self.remote = _snapshot(
                _skill("learned", body="Remote old", origin="agent")
            )
            self.pushes = []

        def fetch_snapshot(self, agent_id):
            return self.remote

        def push_local(self, name, *, deleted=False):
            self.pushes.append((name, deleted))
            local = self.read_local_skill(name)
            self.remote = {
                **self.remote,
                "revision": "snapshot-2",
                "skills": [
                    _skill(
                        "learned",
                        body=local["body"],
                        revision="r2",
                        origin="agent",
                        files=local["files"],
                    )
                ],
            }
            self.update_local_revision(name, "r2")
            return {"status": "accepted", "revision": "r2"}

    service = FixtureService(tmp_path)
    service.apply_snapshot(service.remote)
    skill_md = tmp_path / "skills/learned/SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8").replace("Remote old", "Local edit"),
        encoding="utf-8",
    )

    result = service.reconcile(
        request={
            "project_id": service.remote["project_id"],
            "payload": {"agent_id": service.remote["agent_id"]},
        }
    )

    assert service.pushes == [("learned", False)]
    assert result["status"] == "ready"
    assert "Local edit" in skill_md.read_text(encoding="utf-8")


def test_manifest_is_project_and_agent_bound(tmp_path):
    sync = _load_sync_module()
    service = sync.SkillSyncService(tmp_path)
    service.apply_snapshot(_snapshot(_skill("one")))

    manifest = json.loads(
        (tmp_path / "state/ringo-skill-sync.json").read_text(encoding="utf-8")
    )
    assert manifest["project_id"] == "11111111-1111-1111-1111-111111111111"
    assert manifest["agent_id"] == "22222222-2222-2222-2222-222222222222"

    with pytest.raises(sync.SkillSyncError, match="project"):
        service.apply_snapshot(
            {
                **_snapshot(_skill("one")),
                "project_id": "33333333-3333-3333-3333-333333333333",
            }
        )


def test_malformed_manifest_fails_closed_instead_of_rebinding(tmp_path):
    sync = _load_sync_module()
    service = sync.SkillSyncService(tmp_path)
    service.manifest_path.parent.mkdir(parents=True)
    service.manifest_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(sync.SkillSyncError, match="malformed"):
        service.apply_snapshot(_snapshot(_skill("one")))


def test_successful_supported_tool_edits_are_queued_only(tmp_path):
    sync = _load_sync_module()
    service = sync.SkillSyncService(tmp_path)
    queued = []
    service.queue_local_sync = lambda name, deleted=False: queued.append(
        (name, deleted)
    )

    service.observe_tool(
        tool_name="skill_manage",
        args={"action": "edit", "name": "one"},
        result=json.dumps({"success": True}),
        status="ok",
    )
    service.observe_tool(
        tool_name="skill_manage",
        args={"action": "delete", "name": "two"},
        result=json.dumps({"success": True}),
        status="ok",
    )
    service.observe_tool(
        tool_name="skill_manage",
        args={"action": "edit", "name": "three"},
        result=json.dumps({"success": False, "error": "no"}),
        status="ok",
    )
    service.observe_tool(
        tool_name="read_file",
        args={"path": str(tmp_path / "skills/four/SKILL.md")},
        result="ok",
        status="ok",
    )

    assert queued == [("one", False), ("two", True)]


def test_edit_during_correction_request_remains_dirty_for_next_push(
    tmp_path,
    monkeypatch,
):
    sync = _load_sync_module()
    monkeypatch.setenv("RINGO_IE_MCP_URL", "https://ie.example.com/mcp")
    monkeypatch.setenv("RINGO_IE_MCP_KEY", "secret")
    service = sync.SkillSyncService(tmp_path)
    service.apply_snapshot(
        _snapshot(_skill("learned", body="Remote", origin="agent"))
    )
    skill_md = tmp_path / "skills/learned/SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8").replace("Remote", "First edit"),
        encoding="utf-8",
    )

    def accept_then_edit(url, *, method="GET", body=None):
        assert method == "POST"
        assert body["body"].strip() == "First edit"
        skill_md.write_text(
            skill_md.read_text(encoding="utf-8").replace("First edit", "Second edit"),
            encoding="utf-8",
        )
        return {"status": "accepted", "revision": "r2"}

    service._request_json = accept_then_edit
    service.push_local("learned")

    assert service._dirty_agent_skills() == ["learned"]


def test_reconcile_uses_ie_snapshot_and_correction_http_contract(tmp_path, monkeypatch):
    sync = _load_sync_module()
    monkeypatch.setenv("RINGO_IE_MCP_URL", "https://ie.example.com/mcp")
    monkeypatch.setenv("RINGO_IE_MCP_KEY", "secret")
    service = sync.SkillSyncService(tmp_path)
    remote = _snapshot(_skill("learned", body="Remote", origin="agent"))
    service.apply_snapshot(remote)
    skill_md = tmp_path / "skills/learned/SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8").replace("Remote", "Local"),
        encoding="utf-8",
    )
    requests = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def urlopen(request, timeout):
        assert timeout == 10.0
        requests.append(request)
        assert request.get_header("Authorization") == "Bearer secret"
        if request.method == "POST":
            correction = json.loads(request.data)
            assert request.full_url.endswith("/api/v1/agent/skills/corrections")
            assert correction["base_revision"] == "r1"
            assert correction["body"].strip() == "Local"
            remote["revision"] = "snapshot-2"
            remote["skills"] = [
                _skill("learned", body="Local", revision="r2", origin="agent")
            ]
            return Response({"status": "accepted", "revision": "r2"})
        assert request.method == "GET"
        assert "/api/v1/agent/skills/snapshot?agent_id=" in request.full_url
        return Response(remote)

    monkeypatch.setattr(sync.urllib.request, "urlopen", urlopen)
    result = service.reconcile(
        request={
            "project_id": remote["project_id"],
            "payload": {"agent_id": remote["agent_id"]},
        }
    )

    assert [request.method for request in requests] == ["POST", "GET"]
    assert result["revision"] == "snapshot-2"
    assert "Local" in skill_md.read_text(encoding="utf-8")

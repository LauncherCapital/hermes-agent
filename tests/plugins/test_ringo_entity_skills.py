import importlib.util
import json
import sys
from pathlib import Path

import pytest


PLUGIN_DIR = Path(__file__).parents[2] / "plugins" / "ringo-entity-skills"
PROJECT_ID = "11111111-1111-1111-1111-111111111111"
AGENT_ID = "22222222-2222-2222-2222-222222222222"
SESSION_ID = "ringo_slack_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TURN_ID = "33333333-3333-3333-3333-333333333333"


def _load_service_module():
    package_name = "test_ringo_entity_skills_plugin"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    assert package_spec.loader is not None
    package_spec.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.service")


def _request(**payload):
    return {
        "project_id": PROJECT_ID,
        "payload": {
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "user_id": "U1",
            "channel_id": "C1",
            "channel_type": "channel",
            "include_organization": True,
            "session_id": SESSION_ID,
            "turn_id": TURN_ID,
            **payload,
        },
    }


@pytest.fixture(autouse=True)
def _encryption_keyring(monkeypatch):
    monkeypatch.setenv(
        "RINGO_MESSAGE_STORE_DB_KEYS",
        json.dumps({"1": "11" * 32}),
    )
    monkeypatch.setenv("RINGO_MESSAGE_STORE_DB_KEY_VERSION", "1")


def _service(service_mod, tmp_path):
    def check(payload):
        return {
            "authorized": True,
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": payload["workspace_id"],
            "channel_id": payload["channel_id"],
            "session_id": payload["session_id"],
            "operation": payload["operation"],
            "visibility": "public",
        }

    return service_mod.EntitySkillService(tmp_path, access_checker=check)


def _skill(path: Path, body: str, *, language: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    language_line = f"language_preference: {language}\n" if language else ""
    path.write_text(
        f"---\nname: context\n{language_line}---\n\n{body}\n",
        encoding="utf-8",
    )


def test_prepare_binds_only_exact_runtime_entities(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)

    result = service.prepare(
        request=_request(public_channel_ids=["C2", "C1"])
    )

    assert result["status"] == "ready"
    assert {
        (item["kind"], item["id"]) for item in result["entities"]
    } == {
        ("organizations", "T1"),
        ("users", "U1"),
        ("channels", "C1"),
        ("channels", "C2"),
    }
    assert all(item["path"].endswith("/SKILL.md") for item in result["entities"])
    assert not any(path.exists() for path in map(
        lambda item: Path(item["path"]),
        result["entities"],
    ))


def test_channel_access_uses_rest_root_not_mcp_path(tmp_path, monkeypatch):
    service_mod = _load_service_module()
    monkeypatch.setenv("RINGO_IE_MCP_URL", "https://ie.example.com/mcp/")
    monkeypatch.setenv("RINGO_IE_MCP_KEY", "control-key")
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "authorized": True,
                "project_id": PROJECT_ID,
                "agent_id": AGENT_ID,
                "workspace_id": "T1",
                "channel_id": "C1",
                "session_id": SESSION_ID,
                "operation": "materialize",
                "visibility": "public",
            }

    def post(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs["headers"]
        return Response()

    monkeypatch.setattr(service_mod.httpx, "post", post)
    service = service_mod.EntitySkillService(tmp_path)

    service._check_channel_access(
        project_id=PROJECT_ID,
        agent_id=AGENT_ID,
        workspace_id="T1",
        channel_id="C1",
        channel_type="channel",
        principal_id="",
        slack_user_id="U1",
        session_id=SESSION_ID,
        operation="materialize",
    )

    assert seen["url"] == (
        "https://ie.example.com/api/v1/agent/entity-skills/channel-access"
    )
    assert seen["headers"]["Authorization"] == "Bearer control-key"


def test_team_requires_explicit_verified_membership(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)

    try:
        service.prepare(
            request=_request(
                team_slug="product",
                team_verified=True,
                team_member_ids=["U2"],
            )
        )
    except service_mod.EntitySkillError as exc:
        assert "membership" in str(exc)
    else:
        raise AssertionError("unverified team membership was accepted")


def test_review_lease_serializes_shared_entity_and_turn_is_idempotent(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    assert service.prepare(request=_request())["status"] == "ready"

    busy = service.prepare(
        request=_request(
            session_id="ringo_slack_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            turn_id="44444444-4444-4444-4444-444444444444",
            user_id="U2",
            channel_id="C2",
        )
    )
    assert busy["status"] == "busy"  # organization T1 is shared

    result = service.finish(
        request={
            "project_id": PROJECT_ID,
            "payload": {
                "session_id": SESSION_ID,
                "turn_id": TURN_ID,
                "success": True,
            },
        }
    )
    assert result["status"] == "no_change"
    assert service.prepare(request=_request())["status"] == "duplicate"


def test_bound_review_can_only_read_or_edit_exact_skill_files(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    prepared = service.prepare(request=_request())
    user_path = next(
        item["path"]
        for item in prepared["entities"]
        if item["kind"] == "users"
    )

    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="read_file",
        args={"path": user_path},
    )["action"] == "handled"
    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="write_file",
        args={"path": user_path, "content": "durable"},
    )["action"] == "handled"
    blocked = service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="read_file",
        args={"path": tmp_path / "skills/users/U2/SKILL.md"},
    )
    assert blocked["action"] == "block"
    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="terminal",
        args={},
    )["action"] == "block"


def test_unbound_turn_cannot_scan_another_user_skill(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    blocked = service.authorize_tool(
        session_id="ordinary-session",
        tool_name="read_file",
        args={"path": tmp_path / "skills/users/U2/SKILL.md"},
    )
    assert blocked["action"] == "block"


def test_pre_llm_loads_only_current_ids_and_language(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    _skill(
        tmp_path / "skills/users/U1/SKILL.md",
        "Use Korean for this person.",
        language="ko",
    )
    _skill(
        tmp_path / "skills/users/U2/SKILL.md",
        "CONFIDENTIAL_U2_CONTEXT",
    )
    _skill(
        tmp_path / "skills/channels/C1/SKILL.md",
        "Stable public-channel terminology.",
    )
    _skill(
        tmp_path / "skills/organizations/T1/SKILL.md",
        "The organization uses short decision records.",
    )
    injected = service.inject_context(
        session_id="ordinary-session",
        trusted_runtime_metadata={
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "user_id": "U1",
            "channel_id": "C1",
            "channel_type": "channel",
        },
    )
    assert "Use Korean for this person." in injected["context"]
    assert "Stable public-channel terminology." in injected["context"]
    assert "short decision records" in injected["context"]
    assert "CONFIDENTIAL_U2_CONTEXT" not in injected["context"]

    context = service.context(
        request={
            "project_id": PROJECT_ID,
            "payload": {
                "agent_id": AGENT_ID,
                "workspace_id": "T1",
                "user_id": "U1",
                "session_id": "context-session",
            },
        }
    )
    assert context["language_preference"] == "ko"


def test_finish_reports_only_actual_file_changes(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    prepared = service.prepare(request=_request())
    user_path = Path(next(
        item["path"]
        for item in prepared["entities"]
        if item["kind"] == "users"
    ))
    content = (
        "---\nname: context\nlanguage_preference: ko\n---\n\n"
        "Explicit preference: concise Korean.\n"
    )
    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="write_file",
        args={"path": str(user_path), "content": content},
    )["action"] == "handled"

    result = service.finish(
        request={
            "project_id": PROJECT_ID,
            "payload": {
                "session_id": SESSION_ID,
                "turn_id": TURN_ID,
                "success": True,
            },
        }
    )
    assert result["status"] == "applied"
    assert result["changed"] == [str(user_path)]
    assert user_path.exists()
    encrypted = user_path.read_text()
    assert '"algorithm":"AES-256-GCM"' in encrypted
    assert "Explicit preference" not in encrypted


def test_context_encrypts_canonical_skills_without_deleting_legacy(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    principal_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    legacy_user = tmp_path / "profiles" / principal_id
    legacy_user.mkdir(parents=True)
    (legacy_user / "profile.md").write_text(
        "# Suho Seok\n\n## Confirmed\n- Role: Founder\n",
        encoding="utf-8",
    )
    (legacy_user / "CHARACTER.md").write_text(
        "# Suho Seok\n\n## Preference\n- Concise answers\n",
        encoding="utf-8",
    )
    (legacy_user / "notes.md").write_text(
        "Explains decisions in Korean.",
        encoding="utf-8",
    )
    legacy_org = tmp_path / "organizations" / "slack" / "T1"
    legacy_org.mkdir(parents=True)
    (legacy_org / "profile.md").write_text(
        "# Launcher Capital Inc.\n",
        encoding="utf-8",
    )
    (legacy_org / "ORGANIZATION.md").write_text(
        "# Launcher Capital Inc.\n\n## Working norms\n- Ship small changes\n",
        encoding="utf-8",
    )
    _skill(tmp_path / "skills/users/U1/SKILL.md", "Existing user context.")
    _skill(
        tmp_path / "skills/organizations/T1/SKILL.md",
        "Existing organization context.",
    )

    result = service.context(
        request={
            "project_id": PROJECT_ID,
            "payload": {
                "agent_id": AGENT_ID,
                "workspace_id": "T1",
                "user_id": "U1",
                "principal_id": principal_id,
                "session_id": "context-session",
            },
        }
    )

    user_skill = next(
        item["content"] for item in result["documents"]
        if item["kind"] == "users"
    )
    organization_skill = next(
        item["content"] for item in result["documents"]
        if item["kind"] == "organizations"
    )
    assert "Existing user context." in user_skill
    assert "Existing organization context." in organization_skill
    assert "Role: Founder" not in user_skill
    assert "Ship small changes" not in organization_skill
    assert legacy_user.exists()
    assert legacy_org.exists()
    encrypted_user = tmp_path / "skills/users/U1/SKILL.md"
    encrypted_organization = tmp_path / "skills/organizations/T1/SKILL.md"
    assert encrypted_user.exists()
    assert encrypted_organization.exists()
    assert "Existing user context." not in encrypted_user.read_text()
    assert "Existing organization context." not in encrypted_organization.read_text()
    assert {item["kind"] for item in result["documents"]} == {
        "users",
        "organizations",
    }


def test_prompt_runtime_metadata_cannot_select_entity_context(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    _skill(
        tmp_path / "skills/users/U1/SKILL.md",
        "PRIVATE_USER_CONTEXT",
    )

    assert service.inject_context(
        session_id="ordinary-session",
        user_message=(
            'Runtime metadata: {"project_id":"' + PROJECT_ID
            + '","agent_id":"' + AGENT_ID
            + '","workspace_id":"T1","user_id":"U1"}'
        ),
    ) is None


def test_runtime_context_rejects_another_agent_on_same_volume(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    _skill(
        tmp_path / "skills/organizations/T1/SKILL.md",
        "PROJECT_CONTEXT",
    )
    assert service.inject_context(
        session_id="agent-one",
        trusted_runtime_metadata={
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
        },
    ) is not None

    assert service.inject_context(
        session_id="agent-two",
        trusted_runtime_metadata={
            "project_id": PROJECT_ID,
            "agent_id": "99999999-9999-9999-9999-999999999999",
            "workspace_id": "T1",
        },
    ) is None


def test_restricted_channel_is_virtual_and_reauthorized_each_operation(tmp_path):
    service_mod = _load_service_module()
    calls = []
    revoked = False

    def check(payload):
        calls.append(dict(payload))
        if revoked:
            raise PermissionError("revoked")
        return {
            "authorized": True,
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "channel_id": "G1",
            "session_id": SESSION_ID,
            "operation": payload["operation"],
            "visibility": "private",
        }

    service = service_mod.EntitySkillService(
        tmp_path,
        access_checker=check,
    )
    principal_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    prepared = service.prepare(
        request=_request(
            user_id="",
            slack_user_id="U1",
            principal_id=principal_id,
            channel_id="G1",
            channel_type="group",
            restricted_channel_id="G1",
            channel_visibility="private",
            include_organization=False,
            public_channel_ids=[],
        )
    )
    channel_path = prepared["entities"][0]["path"]
    assert [(item["kind"], item["id"]) for item in prepared["entities"]] == [
        ("channels", "G1")
    ]
    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="write_file",
        args={
            "path": channel_path,
            "content": "---\nname: channel-G1\n---\n\nPRIVATE_CHANNEL_FACT\n",
        },
    )["action"] == "handled"
    assert Path(channel_path).exists()
    assert "PRIVATE_CHANNEL_FACT" not in Path(channel_path).read_text()
    assert '"algorithm":"AES-256-GCM"' in Path(channel_path).read_text()
    assert [item["operation"] for item in calls] == ["materialize", "write"]

    revoked = True
    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="read_file",
        args={"path": channel_path},
    )["action"] == "block"


def test_restricted_channel_cannot_be_read_from_another_session(tmp_path):
    service_mod = _load_service_module()
    service = _service(service_mod, tmp_path)
    channel_path = tmp_path / "skills" / "channels" / "G1" / "SKILL.md"

    blocked = service.authorize_tool(
        session_id="another-session",
        tool_name="read_file",
        args={"path": channel_path},
    )

    assert blocked["action"] == "block"


def test_dm_loads_live_authorized_restricted_channel_skill(tmp_path):
    service_mod = _load_service_module()
    revoked = False
    calls = []

    def check(payload):
        calls.append(dict(payload))
        if revoked or payload["channel_id"] != "G1":
            raise PermissionError("denied")
        return {
            "authorized": True,
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "channel_id": "G1",
            "session_id": payload["session_id"],
            "operation": payload["operation"],
            "visibility": "private",
        }

    service = service_mod.EntitySkillService(
        tmp_path,
        access_checker=check,
    )
    principal_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    prepared = service.prepare(
        request=_request(
            user_id="",
            slack_user_id="U1",
            principal_id=principal_id,
            channel_id="G1",
            channel_type="group",
            restricted_channel_id="G1",
            channel_visibility="private",
            include_organization=False,
            public_channel_ids=[],
        )
    )
    channel_path = prepared["entities"][0]["path"]
    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="write_file",
        args={
            "path": channel_path,
            "content": (
                "---\nname: channel-G1\n---\n\n"
                "PRIVATE_CHANNEL_CONTEXT\n"
            ),
        },
    )["action"] == "handled"
    service.finish(
        request={
            "project_id": PROJECT_ID,
            "payload": {
                "session_id": SESSION_ID,
                "turn_id": TURN_ID,
                "success": True,
            },
        }
    )

    injected = service.inject_context(
        session_id="dm-session",
        trusted_runtime_metadata={
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "user_id": "U1",
            "principal_id": principal_id,
            "channel_id": "D1",
            "channel_type": "im",
        },
    )

    assert "PRIVATE_CHANNEL_CONTEXT" in injected["context"]
    assert calls[-1]["operation"] == "read"
    assert calls[-1]["session_id"] == "dm-session"

    revoked = True
    assert service.inject_context(
        session_id="revoked-dm-session",
        trusted_runtime_metadata={
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "user_id": "U1",
            "principal_id": principal_id,
            "channel_id": "D1",
            "channel_type": "im",
        },
    ) is None


def test_multi_user_channel_never_loads_another_private_channel(tmp_path):
    service_mod = _load_service_module()

    def check(payload):
        visibility = (
            "private" if payload["channel_id"] == "G1" else "public"
        )
        return {
            "authorized": True,
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "channel_id": payload["channel_id"],
            "session_id": payload["session_id"],
            "operation": payload["operation"],
            "visibility": visibility,
        }

    service = service_mod.EntitySkillService(
        tmp_path,
        access_checker=check,
    )
    principal_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    prepared = service.prepare(
        request=_request(
            user_id="",
            slack_user_id="U1",
            principal_id=principal_id,
            channel_id="G1",
            channel_type="group",
            restricted_channel_id="G1",
            channel_visibility="private",
            include_organization=False,
            public_channel_ids=[],
        )
    )
    channel_path = prepared["entities"][0]["path"]
    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="write_file",
        args={
            "path": channel_path,
            "content": "---\nname: private\n---\n\nPRIVATE_OTHER_CHANNEL\n",
        },
    )["action"] == "handled"
    service.finish(
        request={
            "project_id": PROJECT_ID,
            "payload": {
                "session_id": SESSION_ID,
                "turn_id": TURN_ID,
                "success": True,
            },
        }
    )

    injected = service.inject_context(
        session_id="public-session",
        trusted_runtime_metadata={
            "project_id": PROJECT_ID,
            "agent_id": AGENT_ID,
            "workspace_id": "T1",
            "user_id": "U1",
            "principal_id": principal_id,
            "channel_id": "C_PUBLIC",
            "channel_type": "channel",
        },
    )

    assert injected is None or "PRIVATE_OTHER_CHANNEL" not in injected["context"]

import importlib.util
import sys
from pathlib import Path


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


def _skill(path: Path, body: str, *, language: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    language_line = f"language_preference: {language}\n" if language else ""
    path.write_text(
        f"---\nname: context\n{language_line}---\n\n{body}\n",
        encoding="utf-8",
    )


def test_prepare_binds_only_exact_runtime_entities(tmp_path):
    service_mod = _load_service_module()
    service = service_mod.EntitySkillService(tmp_path)

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


def test_team_requires_explicit_verified_membership(tmp_path):
    service_mod = _load_service_module()
    service = service_mod.EntitySkillService(tmp_path)

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
    service = service_mod.EntitySkillService(tmp_path)
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
    service = service_mod.EntitySkillService(tmp_path)
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
    ) is None
    assert service.authorize_tool(
        session_id=SESSION_ID,
        tool_name="write_file",
        args={"path": user_path, "content": "durable"},
    ) is None
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
    service = service_mod.EntitySkillService(tmp_path)
    blocked = service.authorize_tool(
        session_id="ordinary-session",
        tool_name="read_file",
        args={"path": tmp_path / "skills/users/U2/SKILL.md"},
    )
    assert blocked["action"] == "block"


def test_pre_llm_loads_only_current_ids_and_language(tmp_path):
    service_mod = _load_service_module()
    service = service_mod.EntitySkillService(tmp_path)
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
    user_message = (
        "<slack_user_message>hello</slack_user_message>\n\n"
        'Runtime metadata:\n{"workspace_id":"T1","user_id":"U1",'
        '"channel_id":"C1","channel_type":"channel"}\n'
    )

    injected = service.inject_context(user_message=user_message)
    assert "Use Korean for this person." in injected["context"]
    assert "Stable public-channel terminology." in injected["context"]
    assert "short decision records" in injected["context"]
    assert "CONFIDENTIAL_U2_CONTEXT" not in injected["context"]

    context = service.context(
        request={
            "project_id": PROJECT_ID,
            "payload": {"workspace_id": "T1", "user_id": "U1"},
        }
    )
    assert context["language_preference"] == "ko"


def test_finish_reports_only_actual_file_changes(tmp_path):
    service_mod = _load_service_module()
    service = service_mod.EntitySkillService(tmp_path)
    prepared = service.prepare(request=_request())
    user_path = Path(next(
        item["path"]
        for item in prepared["entities"]
        if item["kind"] == "users"
    ))
    _skill(user_path, "Explicit preference: concise Korean.", language="ko")

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


def test_context_migrates_legacy_profile_files_into_entity_skills(tmp_path):
    service_mod = _load_service_module()
    service = service_mod.EntitySkillService(tmp_path)
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
                "workspace_id": "T1",
                "user_id": "U1",
                "principal_id": principal_id,
            },
        }
    )

    user_skill = (tmp_path / "skills/users/U1/SKILL.md").read_text(
        encoding="utf-8"
    )
    organization_skill = (
        tmp_path / "skills/organizations/T1/SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Existing user context." in user_skill
    assert "Role: Founder" in user_skill
    assert "Concise answers" in user_skill
    assert "Explains decisions in Korean." in user_skill
    assert "Existing organization context." in organization_skill
    assert "Ship small changes" in organization_skill
    assert not legacy_user.exists()
    assert not legacy_org.exists()
    assert {item["kind"] for item in result["documents"]} == {
        "users",
        "organizations",
    }

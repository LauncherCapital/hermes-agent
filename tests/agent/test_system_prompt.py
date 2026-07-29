"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _host_managed_prompt=False,
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(cwd=None, skip_soul=False):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def test_host_managed_prompt_keeps_soul_context_and_compact_skills():
    agent = _make_agent(
        _host_managed_prompt=True,
        valid_tool_names={"skill_view"},
        platform="api_server",
        model="openai/gpt-5",
    )

    with (
        patch("run_agent.load_soul_md", return_value="SOUL"),
        patch("run_agent.build_nous_subscription_prompt", return_value="NOUS"),
        patch("run_agent.build_environment_hints", return_value="ENV"),
        patch("run_agent.build_context_files_prompt", return_value="AGENTS"),
        patch(
            "run_agent.build_skills_system_prompt",
            return_value="<available_skills>INDEX</available_skills>",
        ) as skills,
    ):
        parts = build_system_prompt_parts(agent)

    assert "SOUL" in parts["stable"]
    assert "AGENTS" in parts["context"]
    assert "<available_skills>INDEX</available_skills>" in parts["stable"]
    assert "NOUS" not in parts["stable"]
    assert "ENV" not in parts["stable"]
    assert "Do not use markdown" not in parts["stable"]
    assert "Conversation started:" not in parts["volatile"]
    skills.assert_called_once()
    assert skills.call_args.kwargs["compact"] is True

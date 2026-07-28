"""Tests for tools/tool_result_storage.py -- 3-layer tool result persistence."""

import pytest
from unittest.mock import MagicMock, patch

from tools.budget_config import (
    CANARY_CONTEXT_BUDGET_CHARS,
    DEFAULT_CONTEXT_BUDGET_CHARS,
    DEFAULT_RESULT_SIZE_CHARS,
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
)
from tools.tool_result_storage import (
    HEREDOC_MARKER,
    PERSISTED_OUTPUT_TAG,
    PERSISTED_OUTPUT_CLOSING_TAG,
    STORAGE_DIR,
    ToolResultContextBudget,
    _build_persisted_message,
    _heredoc_marker,
    _resolve_storage_dir,
    _write_to_sandbox,
    enforce_turn_budget,
    generate_preview,
    generate_head_tail_preview,
    maybe_persist_tool_result,
    tool_result_context_metrics,
)


# ── generate_preview ──────────────────────────────────────────────────

class TestGeneratePreview:
    def test_short_content_unchanged(self):
        text = "short result"
        preview, has_more = generate_preview(text)
        assert preview == text
        assert has_more is False

    def test_long_content_truncated(self):
        text = "x" * 5000
        preview, has_more = generate_preview(text, max_chars=2000)
        assert len(preview) <= 2000
        assert has_more is True

    def test_truncates_at_newline_boundary(self):
        # 1500 chars + newline + 600 chars  (past halfway)
        text = "a" * 1500 + "\n" + "b" * 600
        preview, has_more = generate_preview(text, max_chars=2000)
        assert preview == "a" * 1500 + "\n"
        assert has_more is True

    def test_ignores_early_newline(self):
        # Newline at position 100, well before halfway of 2000
        text = "a" * 100 + "\n" + "b" * 3000
        preview, has_more = generate_preview(text, max_chars=2000)
        assert len(preview) == 2000
        assert has_more is True

    def test_empty_content(self):
        preview, has_more = generate_preview("")
        assert preview == ""
        assert has_more is False

    def test_exact_boundary(self):
        text = "x" * DEFAULT_PREVIEW_SIZE_CHARS
        preview, has_more = generate_preview(text)
        assert preview == text
        assert has_more is False


class TestGenerateHeadTailPreview:
    def test_tail_evidence_outside_head_is_preserved_deterministically(self):
        evidence = "TAIL_EVIDENCE_FINAL_PRICE"
        text = "HEAD\n" + ("x" * 5_000) + evidence

        preview, has_more = generate_head_tail_preview(text, max_chars=1_500)

        assert preview.startswith("HEAD")
        assert evidence in preview
        assert "[middle omitted]" in preview
        assert len(preview) <= 1_500
        assert has_more is True

    def test_tiny_budget_never_exceeds_requested_size(self):
        marker_size = len("\n...[middle omitted]...\n")

        preview, has_more = generate_head_tail_preview(
            "x" * 100,
            max_chars=marker_size + 1,
        )

        assert len(preview) <= marker_size + 1
        assert has_more is True


# ── _heredoc_marker ───────────────────────────────────────────────────

class TestHeredocMarker:
    def test_default_marker_when_no_collision(self):
        assert _heredoc_marker("normal content") == HEREDOC_MARKER

    def test_uuid_marker_on_collision(self):
        content = f"some text with {HEREDOC_MARKER} embedded"
        marker = _heredoc_marker(content)
        assert marker != HEREDOC_MARKER
        assert marker.startswith("HERMES_PERSIST_")
        assert marker not in content


# ── _write_to_sandbox ─────────────────────────────────────────────────

class TestWriteToSandbox:
    def test_success(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        result = _write_to_sandbox("hello world", "/tmp/hermes-results/abc.txt", env)
        assert result is True
        env.execute.assert_called_once()
        cmd = env.execute.call_args[0][0]
        assert "mkdir -p" in cmd
        # Content travels through stdin, NOT inside the command string —
        # otherwise large content would hit Linux's 128 KB MAX_ARG_STRLEN
        # ceiling on `bash -c <cmd>` (#22906).
        assert "hello world" not in cmd
        assert env.execute.call_args[1]["stdin_data"] == "hello world"

    def test_failure_returns_false(self):
        env = MagicMock()
        env.execute.return_value = {"output": "error", "returncode": 1}
        result = _write_to_sandbox("content", "/tmp/hermes-results/abc.txt", env)
        assert result is False

    def test_large_content_via_stdin(self):
        """Regression: 200 KB content exceeds Linux MAX_ARG_STRLEN (128 KB).
        It must travel via stdin, never inside the command string."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        big = "x" * 200_000
        _write_to_sandbox(big, "/tmp/hermes-results/big.txt", env)
        cmd = env.execute.call_args[0][0]
        assert len(cmd) < 1_000  # cmd is just `mkdir -p X && cat > Y`
        assert env.execute.call_args[1]["stdin_data"] == big

    def test_timeout_passed(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        _write_to_sandbox("content", "/tmp/hermes-results/abc.txt", env)
        assert env.execute.call_args[1]["timeout"] == 30

    def test_uses_parent_dir_of_remote_path(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        remote_path = "/data/data/com.termux/files/usr/tmp/hermes-results/abc.txt"
        _write_to_sandbox("content", remote_path, env)
        cmd = env.execute.call_args[0][0]
        assert "mkdir -p /data/data/com.termux/files/usr/tmp/hermes-results" in cmd

    def test_path_with_spaces_is_quoted(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        remote_path = "/tmp/hermes results/abc file.txt"
        _write_to_sandbox("content", remote_path, env)
        cmd = env.execute.call_args[0][0]
        assert "'/tmp/hermes results'" in cmd
        assert "'/tmp/hermes results/abc file.txt'" in cmd

    def test_shell_metacharacters_neutralized(self):
        """Paths with shell metacharacters must be quoted to prevent injection."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        malicious_path = "/tmp/hermes-results/$(whoami).txt"
        _write_to_sandbox("content", malicious_path, env)
        cmd = env.execute.call_args[0][0]
        # The $() must not appear unquoted — shlex.quote wraps it
        assert "'/tmp/hermes-results/$(whoami).txt'" in cmd

    def test_semicolon_injection_neutralized(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        malicious_path = "/tmp/x; rm -rf /; echo .txt"
        _write_to_sandbox("content", malicious_path, env)
        cmd = env.execute.call_args[0][0]
        # The semicolons must be inside quotes, not acting as command separators
        assert "'/tmp/x; rm -rf /; echo .txt'" in cmd


class TestResolveStorageDir:
    def test_defaults_to_storage_dir_without_env(self):
        assert _resolve_storage_dir(None) == STORAGE_DIR

    def test_uses_env_temp_dir_when_available(self):
        env = MagicMock()
        env.get_temp_dir.return_value = "/data/data/com.termux/files/usr/tmp"
        assert _resolve_storage_dir(env) == "/data/data/com.termux/files/usr/tmp/hermes-results"


# ── _build_persisted_message ──────────────────────────────────────────

class TestBuildPersistedMessage:
    def test_structure(self):
        msg = _build_persisted_message(
            preview="first 100 chars...",
            has_more=True,
            original_size=50_000,
            file_path="/tmp/hermes-results/test123.txt",
        )
        assert msg.startswith(PERSISTED_OUTPUT_TAG)
        assert msg.endswith(PERSISTED_OUTPUT_CLOSING_TAG)
        assert "50,000 characters" in msg
        assert "/tmp/hermes-results/test123.txt" in msg
        assert "read_file" in msg
        assert "first 100 chars..." in msg
        assert "..." in msg  # has_more indicator

    def test_no_ellipsis_when_complete(self):
        msg = _build_persisted_message(
            preview="complete content",
            has_more=False,
            original_size=16,
            file_path="/tmp/hermes-results/x.txt",
        )
        # Should not have the trailing "..." indicator before closing tag
        lines = msg.strip().split("\n")
        assert lines[-2] != "..."

    def test_large_size_shows_mb(self):
        msg = _build_persisted_message(
            preview="x",
            has_more=True,
            original_size=2_000_000,
            file_path="/tmp/hermes-results/big.txt",
        )
        assert "MB" in msg


# ── maybe_persist_tool_result ─────────────────────────────────────────

class TestMaybePersistToolResult:
    def test_below_threshold_returns_unchanged(self):
        content = "small result"
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_123",
            env=None,
            threshold=50_000,
        )
        assert result == content

    def test_above_threshold_with_env_persists(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_456",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        assert "tc_456.txt" in result
        assert len(result) < len(content)
        env.execute.assert_called_once()

    def test_persists_full_content_as_is(self):
        """Content is persisted verbatim — no JSON extraction."""
        import json
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        raw = "line1\nline2\n" * 5_000
        content = json.dumps({"output": raw, "exit_code": 0, "error": None})
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_json",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        # Content is delivered through stdin (no longer embedded in the
        # command string — see test_large_content_via_stdin for why).
        assert env.execute.call_args[1]["stdin_data"] == content

    def test_above_threshold_no_env_truncates_inline(self):
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_789",
            env=None,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG not in result
        assert "Truncated" in result
        assert len(result) < len(content)

    def test_env_write_failure_falls_back_to_truncation(self):
        env = MagicMock()
        env.execute.return_value = {"output": "disk full", "returncode": 1}
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_fail",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG not in result
        assert "Truncated" in result

    def test_env_execute_exception_falls_back(self):
        env = MagicMock()
        env.execute.side_effect = RuntimeError("connection lost")
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_exc",
            env=env,
            threshold=30_000,
        )
        assert "Truncated" in result

    def test_read_file_never_persisted(self):
        """read_file has threshold=inf, should never be persisted."""
        env = MagicMock()
        content = "x" * 200_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="read_file",
            tool_use_id="tc_rf",
            env=env,
            threshold=float("inf"),
        )
        assert result == content
        env.execute.assert_not_called()

    def test_uses_registry_threshold_when_not_provided(self):
        """When threshold=None, looks up from registry."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "x" * 60_000

        mock_registry = MagicMock()
        mock_registry.get_max_result_size.return_value = 30_000

        with patch("tools.registry.registry", mock_registry):
            result = maybe_persist_tool_result(
                content=content,
                tool_name="terminal",
                tool_use_id="tc_reg",
                env=env,
                threshold=None,
            )
        # Should have persisted since 60K > 30K
        assert PERSISTED_OUTPUT_TAG in result or "Truncated" in result

    def test_unicode_content_survives(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "日本語テスト " * 10_000  # ~60K chars of unicode
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_uni",
            env=env,
            threshold=30_000,
        )
        assert PERSISTED_OUTPUT_TAG in result
        # Preview should contain unicode
        assert "日本語テスト" in result

    def test_empty_content_returns_unchanged(self):
        result = maybe_persist_tool_result(
            content="",
            tool_name="terminal",
            tool_use_id="tc_empty",
            env=None,
            threshold=30_000,
        )
        assert result == ""

    def test_whitespace_only_below_threshold(self):
        content = " " * 100
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_ws",
            env=None,
            threshold=30_000,
        )
        assert result == content

    def test_file_path_uses_tool_use_id(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="unique_id_abc",
            env=env,
            threshold=30_000,
        )
        assert "unique_id_abc.txt" in result

    def test_preview_included_in_persisted_output(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        # Create content with a distinctive start
        content = "DISTINCTIVE_START_MARKER" + "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_prev",
            env=env,
            threshold=30_000,
        )
        assert "DISTINCTIVE_START_MARKER" in result

    def test_env_temp_dir_changes_persisted_path(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        env.get_temp_dir.return_value = "/data/data/com.termux/files/usr/tmp"
        content = "x" * 60_000
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_termux",
            env=env,
            threshold=30_000,
        )
        assert "/data/data/com.termux/files/usr/tmp/hermes-results/tc_termux.txt" in result
        cmd = env.execute.call_args[0][0]
        assert "mkdir -p /data/data/com.termux/files/usr/tmp/hermes-results" in cmd

    def test_threshold_zero_forces_persist(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        content = "even short content"
        result = maybe_persist_tool_result(
            content=content,
            tool_name="terminal",
            tool_use_id="tc_zero",
            env=env,
            threshold=0,
        )
        # Any non-empty content with threshold=0 should be persisted
        assert PERSISTED_OUTPUT_TAG in result


# ── enforce_turn_budget ───────────────────────────────────────────────

class TestEnforceTurnBudget:
    def test_under_budget_no_changes(self):
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "small"},
            {"role": "tool", "tool_call_id": "t2", "content": "also small"},
        ]
        result = enforce_turn_budget(msgs, env=None, config=BudgetConfig(turn_budget=200_000))
        assert result[0]["content"] == "small"
        assert result[1]["content"] == "also small"

    def test_over_budget_largest_persisted_first(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "a" * 80_000},
            {"role": "tool", "tool_call_id": "t2", "content": "b" * 130_000},
        ]
        # Total 210K > 200K budget
        enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
        # The larger one (130K) should be persisted first
        assert PERSISTED_OUTPUT_TAG in msgs[1]["content"]

    def test_already_persisted_results_skipped(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": "t1",
             "content": f"{PERSISTED_OUTPUT_TAG}\nalready persisted\n{PERSISTED_OUTPUT_CLOSING_TAG}"},
            {"role": "tool", "tool_call_id": "t2", "content": "x" * 250_000},
        ]
        enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
        # t1 should be untouched (already persisted)
        assert msgs[0]["content"].startswith(PERSISTED_OUTPUT_TAG)
        # t2 should be persisted
        assert PERSISTED_OUTPUT_TAG in msgs[1]["content"]

    def test_medium_result_regression(self):
        """6 results of 42K chars each (252K total) — each under 100K default
        threshold but aggregate exceeds 200K budget. L3 should persist."""
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        msgs = [
            {"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 42_000}
            for i in range(6)
        ]
        enforce_turn_budget(msgs, env=env, config=BudgetConfig(turn_budget=200_000))
        # At least some results should be persisted to get under 200K
        persisted_count = sum(
            1 for m in msgs if PERSISTED_OUTPUT_TAG in m["content"]
        )
        assert persisted_count >= 2  # Need to shed at least ~52K

    def test_no_env_falls_back_to_truncation(self):
        msgs = [
            {"role": "tool", "tool_call_id": "t1", "content": "x" * 250_000},
        ]
        enforce_turn_budget(msgs, env=None, config=BudgetConfig(turn_budget=200_000))
        # Should be truncated (no sandbox available)
        assert "Truncated" in msgs[0]["content"] or PERSISTED_OUTPUT_TAG in msgs[0]["content"]

    def test_returns_same_list(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]
        result = enforce_turn_budget(msgs, env=None, config=BudgetConfig(turn_budget=200_000))
        assert result is msgs

    def test_empty_messages(self):
        result = enforce_turn_budget([], env=None, config=BudgetConfig(turn_budget=200_000))
        assert result == []


# ── Cross-API result context budget ──────────────────────────────────

class TestToolResultContextBudget:
    # Content-free production fixture from execution
    # f215d59d-4e24-477c-aeb1-7f7452f02ce0. Each nested list is one tool
    # batch; values are result character counts in call order.
    PRODUCTION_BATCHES = [
        [1_159],
        [1_159, 1_159, 1_159],
        [1_405, 15_680, 1_404],
        [1_408, 1_825, 537],
        [26_158, 1_026, 6_376],
        [533, 533, 539],
        [386],
    ]

    def test_production_fixture_reproduces_replayed_result_context(self):
        cumulative_chars = 0
        replayed_tokens = 0
        for batch in self.PRODUCTION_BATCHES:
            cumulative_chars += sum(batch)
            replayed_tokens += (cumulative_chars + 3) // 4

        assert cumulative_chars == 62_446
        assert replayed_tokens == 60_196

    def test_production_fixture_spills_before_first_model_exposure(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        budget = ToolResultContextBudget(
            config=BudgetConfig(context_budget=CANARY_CONTEXT_BUDGET_CHARS),
        )
        inline_batches = []
        call_index = 0
        replayed_inline_tokens = 0

        for batch in self.PRODUCTION_BATCHES:
            inline_batch = []
            for size in batch:
                call_index += 1
                inline_batch.append(
                    budget.prepare(
                        content="x" * size,
                        tool_name=f"read_tool_{call_index}",
                        tool_use_id=f"call_{call_index}",
                        env=env,
                    )
                )
            inline_batches.append(inline_batch)
            replayed_inline_tokens += (budget.inline_chars + 3) // 4

        assert budget.raw_chars == 62_446
        assert budget.budget_chars == CANARY_CONTEXT_BUDGET_CHARS == 40_000
        assert budget.budget_spill_count >= 1
        assert budget.inline_chars < budget.raw_chars
        assert replayed_inline_tokens <= 43_000
        assert 60_196 - replayed_inline_tokens >= 17_000
        assert any(
            PERSISTED_OUTPUT_TAG in result
            for batch in inline_batches
            for result in batch
        )
        assert sum(
            "Result-context budget reached" in result
            for batch in inline_batches
            for result in batch
        ) == 1

    def test_budget_is_fail_open_without_recoverable_storage(self):
        budget = ToolResultContextBudget(
            config=BudgetConfig(context_budget=10),
        )
        content = "evidence that must remain available"

        result = budget.prepare(
            content=content,
            tool_name="read_tool",
            tool_use_id="call_1",
            env=None,
        )

        assert result == content
        assert budget.budget_spill_count == 0

    def test_default_budget_does_not_change_existing_inline_recall(
        self,
        caplog,
    ):
        env = MagicMock()
        content = "HEAD\n" + ("x" * 50_000) + "TAIL_EVIDENCE"
        budget = ToolResultContextBudget()

        with caplog.at_level("INFO", logger="tools.tool_result_storage"):
            result = budget.prepare(
                content=content,
                tool_name="read_tool",
                tool_use_id="call_1",
                env=env,
            )

        assert DEFAULT_CONTEXT_BUDGET_CHARS == float("inf")
        assert result == content
        assert "TAIL_EVIDENCE" in result
        assert "budget_chars=disabled" in caplog.text
        env.execute.assert_not_called()

    def test_opt_in_budget_keeps_tail_evidence_in_inline_preview_and_full_storage(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        evidence = "TAIL_EVIDENCE_FINAL_PRICE"
        content = "HEAD\n" + ("x" * 50_000) + evidence
        budget = ToolResultContextBudget(
            config=BudgetConfig(context_budget=10),
        )

        result = budget.prepare(
            content=content,
            tool_name="read_tool",
            tool_use_id="call_1",
            env=env,
        )

        assert PERSISTED_OUTPUT_TAG in result
        assert "Preview (head+tail " in result
        assert result.index("HEAD") < result.index("[middle omitted]")
        assert result.index("[middle omitted]") < result.index(evidence)
        assert env.execute.call_args.kwargs["stdin_data"] == content

    def test_budget_is_fail_open_when_storage_write_fails(self):
        env = MagicMock()
        env.execute.return_value = {"output": "disk full", "returncode": 1}
        budget = ToolResultContextBudget(
            config=BudgetConfig(context_budget=10),
        )
        content = "evidence that must remain available"

        result = budget.prepare(
            content=content,
            tool_name="read_tool",
            tool_use_id="call_1",
            env=env,
        )

        assert result == content
        assert budget.budget_spill_count == 0

    def test_read_file_recovery_remains_inline_past_budget(self):
        env = MagicMock()
        env.execute.return_value = {"output": "", "returncode": 0}
        budget = ToolResultContextBudget(
            config=BudgetConfig(context_budget=10),
        )
        content = "recovered evidence"

        result = budget.prepare(
            content=content,
            tool_name="read_file",
            tool_use_id="call_1",
            env=env,
        )

        assert result == content
        env.execute.assert_not_called()

    def test_request_metrics_include_only_tool_result_metadata(self):
        messages = [
            {"role": "user", "content": "private user text"},
            {
                "role": "assistant",
                "content": "private reply",
                "tool_calls": [
                    {"function": {"name": "private_tool", "arguments": "z" * 7}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "x" * 9},
            {"role": "tool", "tool_call_id": "b", "content": "y" * 8},
        ]

        assert tool_result_context_metrics(messages) == {
            "result_count": 2,
            "result_chars": 17,
            "approx_tokens": 5,
            "tool_call_count": 1,
            "tool_argument_chars": 7,
            "approx_argument_tokens": 2,
        }

    def test_turn_metrics_are_content_free_and_report_recovery(self):
        budget = ToolResultContextBudget(
            config=BudgetConfig(context_budget=10),
        )
        budget.prepare(
            content="recovered evidence",
            tool_name="read_file",
            tool_use_id="call_1",
            env=MagicMock(),
        )

        assert budget.to_metrics() == {
            "budget_chars": 10,
            "result_count": 1,
            "raw_chars": 18,
            "inline_chars": 18,
            "budget_spill_count": 0,
            "recovery_result_count": 1,
            "replayed_approx_tokens": 0,
        }


# ── Per-tool threshold integration ────────────────────────────────────

class TestPerToolThresholds:
    """Verify registry wiring for per-tool thresholds."""

    def test_registry_has_get_max_result_size(self):
        from tools.registry import registry
        assert hasattr(registry, "get_max_result_size")

    def test_default_threshold(self):
        from tools.registry import registry
        # Unknown tool should return the default
        val = registry.get_max_result_size("nonexistent_tool_xyz")
        assert val == DEFAULT_RESULT_SIZE_CHARS

    def test_terminal_threshold(self):
        from tools.registry import registry
        # Trigger import of terminal_tool to register the tool
        try:
            import tools.terminal_tool  # noqa: F401
            val = registry.get_max_result_size("terminal")
            assert val == 100_000
        except ImportError:
            pytest.skip("terminal_tool not importable in test env")

    def test_read_file_result_size_cap(self):
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("read_file")
            assert val == 100_000
        except ImportError:
            pytest.skip("file_tools not importable in test env")

    def test_read_file_registry_cap_is_100k(self):
        """Regression test: read_file must have a 100_000 char registry cap (Layer 2 safety net)."""
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("read_file")
            assert val == 100_000, (
                f"read_file registry cap must be 100_000, got {val!r}. "
                "float('inf') is not allowed — it disables the Layer 2 result-size guard."
            )
        except ImportError:
            pytest.skip("file_tools not importable in test env")

    def test_search_files_threshold(self):
        from tools.registry import registry
        try:
            import tools.file_tools  # noqa: F401
            val = registry.get_max_result_size("search_files")
            assert val == 100_000
        except ImportError:
            pytest.skip("file_tools not importable in test env")

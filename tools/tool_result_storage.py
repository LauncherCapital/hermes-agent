"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at four levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written INTO THE
   SANDBOX temp dir (for example /tmp/hermes-results/{tool_use_id}.txt on
   standard Linux, or $TMPDIR/hermes-results/{tool_use_id}.txt on Termux)
   via env.execute(). The in-context content is replaced with a preview +
   file path reference. The model can read_file to access the full output
   on any backend.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.

4. **Cross-API result context budget** (ToolResultContextBudget): Across all
   assistant/tool batches in one user turn, persist new results before their
   first model exposure once inline evidence reaches the configured budget.
   Existing request prefixes are never rewritten.
"""

import logging
import os
import shlex
import uuid
from dataclasses import dataclass, field
from typing import Any

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
    PINNED_THRESHOLDS,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_CONTEXT_BUDGET_GUIDANCE = (
    "\n\n[Result-context budget reached. Before making another tool call, decide "
    "whether the evidence already answers the request. If it does, stop exploring "
    "and answer. If it does not, use a materially different, narrower call or read "
    "only the relevant section of the saved result.]"
)


def _content_size(content: Any) -> int:
    """Return serialized content size without retaining or logging its text."""
    if isinstance(content, str):
        return len(content)
    try:
        import json

        return len(json.dumps(content, ensure_ascii=False, default=str))
    except Exception:
        return len(str(content))


def tool_result_context_metrics(messages: list[dict]) -> dict[str, int]:
    """Measure tool payload sizes; never return content, arguments, or results."""
    sizes = [
        _content_size(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    argument_sizes = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            argument_sizes.append(_content_size(function.get("arguments", "")))
    chars = sum(sizes)
    argument_chars = sum(argument_sizes)
    return {
        "result_count": len(sizes),
        "result_chars": chars,
        "approx_tokens": (chars + 3) // 4,
        "tool_call_count": len(argument_sizes),
        "tool_argument_chars": argument_chars,
        "approx_argument_tokens": (argument_chars + 3) // 4,
    }


@dataclass
class ToolResultContextBudget:
    """Bound new inline tool evidence across one user turn.

    Results are persisted before their first model exposure, so prior request
    prefixes are never rewritten and prompt caching remains stable. The budget
    is soft: tools required to recover persisted output remain inline, and a
    missing sandbox fails open rather than discarding evidence.
    """

    config: BudgetConfig = field(default_factory=lambda: DEFAULT_BUDGET)
    raw_chars: int = 0
    inline_chars: int = 0
    result_count: int = 0
    budget_spill_count: int = 0
    recovery_result_count: int = 0
    replayed_approx_tokens: int = 0
    _guidance_emitted: bool = False

    @property
    def budget_chars(self) -> int | float:
        return self.config.context_budget

    def prepare(
        self,
        *,
        content: str,
        tool_name: str,
        tool_use_id: str,
        env=None,
    ) -> str:
        """Apply existing per-result persistence, then the cross-API budget."""
        raw_size = len(content)
        self.raw_chars += raw_size
        self.result_count += 1

        result = maybe_persist_tool_result(
            content=content,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            env=env,
            config=self.config,
            preview_strategy=(
                "head_tail"
                if self.budget_chars != float("inf")
                else "head"
            ),
        )
        budget_spill = False
        would_exceed = self.inline_chars + len(result) > self.budget_chars
        recovery_is_pinned = PINNED_THRESHOLDS.get(tool_name) == float("inf")
        if recovery_is_pinned:
            self.recovery_result_count += 1

        if (
            result == content
            and would_exceed
            and not recovery_is_pinned
            and env is not None
        ):
            persisted_result = maybe_persist_tool_result(
                content=content,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                env=env,
                config=self.config,
                threshold=0,
                preview_strategy="head_tail",
            )
            # The ordinary per-result safety cap keeps its historical inline
            # truncation fallback. A cumulative-budget spill is different:
            # without a recoverable saved copy, preserve accuracy and fail open.
            if PERSISTED_OUTPUT_TAG in persisted_result:
                result = persisted_result
            budget_spill = result != content
            if budget_spill:
                self.budget_spill_count += 1
                if not self._guidance_emitted:
                    result += _CONTEXT_BUDGET_GUIDANCE
                    self._guidance_emitted = True

        self.inline_chars += len(result)
        budget_label = (
            "disabled"
            if self.budget_chars == float("inf")
            else str(self.budget_chars)
        )
        logger.info(
            "Tool result context: tool=%s raw_chars=%d inline_chars=%d "
            "turn_raw_chars=%d turn_inline_chars=%d approx_inline_tokens=%d "
            "budget_chars=%s budget_spill=%s",
            tool_name,
            raw_size,
            len(result),
            self.raw_chars,
            self.inline_chars,
            (self.inline_chars + 3) // 4,
            budget_label,
            budget_spill,
        )
        return result

    def to_metrics(self) -> dict[str, int]:
        """Return content-free turn metrics for canary evaluation."""
        return {
            "budget_chars": (
                0
                if self.budget_chars == float("inf")
                else int(self.budget_chars)
            ),
            "result_count": self.result_count,
            "raw_chars": self.raw_chars,
            "inline_chars": self.inline_chars,
            "budget_spill_count": self.budget_spill_count,
            "recovery_result_count": self.recovery_result_count,
            "replayed_approx_tokens": self.replayed_approx_tokens,
        }

    def note_api_request(self) -> None:
        """Account for the current turn's result context replay on one request."""
        self.replayed_approx_tokens += (self.inline_chars + 3) // 4


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    if env is not None:
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
            except Exception as exc:
                logger.debug("Could not resolve env temp dir: %s", exc)
            else:
                if temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    return f"{temp_dir}/hermes-results"
    return STORAGE_DIR


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def generate_head_tail_preview(
    content: str,
    max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS,
) -> tuple[str, bool]:
    """Return deterministic head+tail evidence within ``max_chars``."""
    if len(content) <= max_chars:
        return content, False
    marker = "\n...[middle omitted]...\n"
    if max_chars <= len(marker):
        return content[:max_chars], True
    available = max_chars - len(marker)
    head_chars = (available + 1) // 2
    tail_chars = available - head_chars
    tail = content[-tail_chars:] if tail_chars else ""
    return content[:head_chars] + marker + tail, True


def _heredoc_marker(content: str) -> str:
    """Return a heredoc delimiter that doesn't collide with content."""
    if HEREDOC_MARKER not in content:
        return HEREDOC_MARKER
    return f"HERMES_PERSIST_{uuid.uuid4().hex[:8]}"


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(). Returns True on success.

    Pushes ``content`` through stdin rather than embedding it in the command
    string. Linux's ``MAX_ARG_STRLEN`` caps any single argv element at 128 KB
    (32 * PAGE_SIZE), so the previous heredoc-in-the-command-string approach
    silently failed with ``OSError: [Errno 7] Argument list too long`` for any
    tool result over ~128 KB — exactly the case persistence exists to handle.
    Routing through stdin removes that ceiling on local + ssh (``_stdin_mode
    == "pipe"``); remote backends with ``_stdin_mode == "heredoc"`` keep their
    existing API-body sized limit, which is orders of magnitude larger than
    the exec-arg ceiling.
    """
    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
    preview_includes_tail: bool = False,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
    preview_kind = "head+tail" if preview_includes_tail else "first"
    msg += f"Preview ({preview_kind} {len(preview)} chars):\n"
    msg += preview
    if has_more and not preview_includes_tail:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
    preview_strategy: str = "head",
) -> str:
    """Layer 2: persist oversized result into the sandbox, return preview + path.

    Writes via env.execute() so the file is accessible from any backend
    (local, Docker, SSH, Modal, Daytona). Falls back to inline truncation
    if write fails or no env is available.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (used for threshold lookup).
        tool_use_id: Unique ID for this tool call (used as filename).
        env: The active BaseEnvironment instance, or None.
        config: BudgetConfig controlling thresholds and preview size.
        threshold: Explicit override; takes precedence over config resolution.

    Returns:
        Original content if small, or <persisted-output> replacement.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    storage_dir = _resolve_storage_dir(env)
    remote_path = f"{storage_dir}/{tool_use_id}.txt"
    preview_includes_tail = preview_strategy == "head_tail"
    if preview_includes_tail:
        preview, has_more = generate_head_tail_preview(
            content,
            max_chars=config.preview_size,
        )
    else:
        preview, has_more = generate_preview(content, max_chars=config.preview_size)

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(
                    preview,
                    has_more,
                    len(content),
                    remote_path,
                    preview_includes_tail=preview_includes_tail,
                )
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate budget across all tool results in a turn.

    If total chars exceed budget, persist the largest non-persisted results
    first (via sandbox write) until under budget. Already-persisted results
    are skipped.

    Mutates the list in-place and returns it.
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if PERSISTED_OUTPUT_TAG not in content:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        replacement = maybe_persist_tool_result(
            content=content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=0,
        )
        if replacement != content:
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            logger.info(
                "Budget enforcement: persisted tool result %s (%d chars)",
                tool_use_id, size,
            )

    return tool_messages

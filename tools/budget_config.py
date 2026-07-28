"""Configurable budget constants for tool result persistence.

Per-tool resolution: pinned > config overrides > registry > default.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

# Tools whose thresholds must never be overridden.
# read_file=inf prevents infinite persist->read->persist loops.
PINNED_THRESHOLDS: Dict[str, float] = {
    "read_file": float("inf"),
}

# Defaults matching the current hardcoded values in tool_result_storage.py.
# Kept here as the single source of truth; tool_result_storage.py imports these.
DEFAULT_RESULT_SIZE_CHARS: int = 100_000
DEFAULT_TURN_BUDGET_CHARS: int = 200_000
# Cross-API limiting is canary-only. The default preserves historical Hermes
# behavior until a tenant/runtime explicitly opts in through config or env.
DEFAULT_CONTEXT_BUDGET_CHARS: float = float("inf")
CANARY_CONTEXT_BUDGET_CHARS: int = 40_000
CONTEXT_BUDGET_ENV_VAR = "HERMES_TOOL_RESULT_CONTEXT_BUDGET_CHARS"
DEFAULT_PREVIEW_SIZE_CHARS: int = 1_500


@dataclass(frozen=True)
class BudgetConfig:
    """Immutable budget constants for the 3-layer tool result persistence system.

    Layer 2 (per-result): resolve_threshold(tool_name) -> threshold in chars.
    Layer 3 (per-turn):   turn_budget -> aggregate char budget across all tool
                          results in a single assistant turn.
    Cross-API:            context_budget -> inline tool-result chars accumulated
                          across one user turn.
    Preview:              preview_size -> inline snippet size after persistence.
    """

    default_result_size: int = DEFAULT_RESULT_SIZE_CHARS
    turn_budget: int = DEFAULT_TURN_BUDGET_CHARS
    context_budget: int | float = DEFAULT_CONTEXT_BUDGET_CHARS
    preview_size: int = DEFAULT_PREVIEW_SIZE_CHARS
    tool_overrides: Dict[str, int] = field(default_factory=dict)

    def resolve_threshold(self, tool_name: str) -> int | float:
        """Resolve the persistence threshold for a tool.

        Priority: pinned -> tool_overrides -> registry per-tool -> default.
        """
        if tool_name in PINNED_THRESHOLDS:
            return PINNED_THRESHOLDS[tool_name]
        if tool_name in self.tool_overrides:
            return self.tool_overrides[tool_name]
        from tools.registry import registry
        return registry.get_max_result_size(tool_name, default=self.default_result_size)


def _positive_budget(value: Any) -> int | float:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_BUDGET_CHARS
    return parsed if parsed > 0 else DEFAULT_CONTEXT_BUDGET_CHARS


def load_runtime_budget_config(
    config: Mapping[str, Any] | None = None,
) -> BudgetConfig:
    """Resolve the opt-in cross-API budget from config, then tenant env.

    ``tool_output.context_budget_chars`` follows Hermes's existing output-limit
    config surface. ``HERMES_TOOL_RESULT_CONTEXT_BUDGET_CHARS`` has precedence
    so one tenant/runtime can be canaried without rewriting its volume config.
    Missing or invalid values preserve historical behavior (disabled).
    """
    if config is None:
        try:
            from hermes_cli.config import load_config

            loaded = load_config()
            config = loaded if isinstance(loaded, Mapping) else {}
        except Exception:
            config = {}

    section = config.get("tool_output") if isinstance(config, Mapping) else None
    configured = (
        section.get("context_budget_chars")
        if isinstance(section, Mapping)
        else None
    )
    if CONTEXT_BUDGET_ENV_VAR in os.environ:
        configured = os.environ.get(CONTEXT_BUDGET_ENV_VAR)

    return BudgetConfig(context_budget=_positive_budget(configured))


# Default config -- matches current hardcoded behavior exactly.
DEFAULT_BUDGET = BudgetConfig()

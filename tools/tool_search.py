"""Progressive tool disclosure ("tool search") for Hermes Agent.

When enabled, MCP and non-core plugin tools are replaced in the model-visible
tools array by three bridge tools — ``tool_search``, ``tool_describe``,
``tool_call`` — and surfaced on demand. Core Hermes tools never defer.

Design constraints this module is built around (see ``openclaw-tool-search-report``
for the full rationale):

* Core tools defined in ``toolsets._HERMES_CORE_TOOLS`` are *never* deferred.
  Always-load means always-load. No exceptions.
* Tool search is disabled unless config explicitly enables the rollout. In
  ``auto`` mode, the threshold gate runs every assembly: when deferrable tools
  would consume less than both ``threshold_pct`` of the model's context window
  (default 10%) and ``threshold_tokens`` (default 20K), tool search is a no-op.
* The catalog is stateless across turns and tools-array assemblies. It is
  rebuilt from the current tool-defs list every time. This is the lesson
  from OpenClaw's cron regression (openclaw/openclaw#84141): a session-keyed
  catalog that drifts out of sync with the live tool registry produces
  silent tool dropouts.
* Bridge tools route through ``model_tools.handle_function_call`` exactly
  like a direct call, so guardrails, plugin pre/post hooks, approval flows,
  and tool-result truncation all fire identically.
* Display and trajectory unwrap is implemented here so the user (CLI activity
  feed, gateway, saved trajectories) always sees the underlying tool, not
  the bridge.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("tools.tool_search")


# Bridge tool names. These names are reserved and may not collide with a
# user/plugin/MCP tool — registration of any tool with these names is
# rejected by the registry's existing override-protection logic.
TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_CALL_NAME = "tool_call"

BRIDGE_TOOL_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_CALL_NAME})

# When estimating tokens from char count without a real tokenizer, this is
# the cheap rule of thumb that's stable across providers. Roughly 4 chars
# per token for English+JSON. Underestimating leads to false negatives
# (tool search not activated when it should); overestimating leads to false
# positives (activated when not needed). 4.0 errs slightly toward
# underestimating, which is the safer default.
CHARS_PER_TOKEN = 4.0
DEFAULT_THRESHOLD_TOKENS = 20_000


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchConfig:
    """Resolved, validated tool-search configuration for a single assembly."""

    enabled: str  # "auto" | "on" | "off"
    threshold_pct: float  # 0..100 — only used when enabled == "auto"
    threshold_tokens: int  # absolute schema-token budget used in "auto"
    search_default_limit: int
    max_search_limit: int
    always_visible: frozenset[str] = frozenset()

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolSearchConfig":
        """Build a config from a raw dict / bool / None.

        Accepts the legacy bool shape (``tools.tool_search: true``) and the
        dict shape (``tools.tool_search: {enabled: auto, ...}``). Validates
        and clamps every numeric field; unknown values fall back to safe
        defaults rather than raising, so a typo in user config does not
        break the agent.
        """
        if raw is True:
            return cls(enabled="auto", threshold_pct=10.0,
                       threshold_tokens=DEFAULT_THRESHOLD_TOKENS,
                       search_default_limit=5, max_search_limit=20)
        if raw is False:
            return cls(enabled="off", threshold_pct=10.0,
                       threshold_tokens=DEFAULT_THRESHOLD_TOKENS,
                       search_default_limit=5, max_search_limit=20)
        if not isinstance(raw, dict):
            return cls(enabled="off", threshold_pct=10.0,
                       threshold_tokens=DEFAULT_THRESHOLD_TOKENS,
                       search_default_limit=5, max_search_limit=20)

        enabled_raw = str(raw.get("enabled", "off")).strip().lower()
        if enabled_raw in ("true", "1", "yes"):
            enabled = "on"
        elif enabled_raw in ("false", "0", "no"):
            enabled = "off"
        elif enabled_raw in ("auto", "on", "off"):
            enabled = enabled_raw
        else:
            enabled = "off"

        threshold_pct = _safe_float(raw.get("threshold_pct"), 10.0)
        threshold_pct = max(0.0, min(100.0, threshold_pct))
        threshold_tokens = max(
            1,
            _safe_int(raw.get("threshold_tokens"), DEFAULT_THRESHOLD_TOKENS),
        )

        max_search_limit = max(1, min(50, _safe_int(raw.get("max_search_limit"), 20)))
        search_default_limit = max(1, min(max_search_limit,
                                          _safe_int(raw.get("search_default_limit"), 5)))
        raw_always_visible = raw.get("always_visible")
        always_visible = frozenset(
            str(name).strip()
            for name in raw_always_visible
            if isinstance(name, str) and name.strip()
        ) if isinstance(raw_always_visible, list) else frozenset()

        return cls(
            enabled=enabled,
            threshold_pct=threshold_pct,
            threshold_tokens=threshold_tokens,
            search_default_limit=search_default_limit,
            max_search_limit=max_search_limit,
            always_visible=always_visible,
        )


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def load_config() -> ToolSearchConfig:
    """Load tool-search config from the user config file."""
    try:
        from hermes_cli.config import load_config as _load
        cfg = _load() or {}
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        if not isinstance(tools_cfg, dict):
            tools_cfg = {}
        return ToolSearchConfig.from_raw(tools_cfg.get("tool_search"))
    except Exception as e:
        logger.debug("Failed to load tool-search config: %s", e)
        return ToolSearchConfig.from_raw(None)


# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------


def _core_tool_names() -> frozenset[str]:
    """Return the set of tool names that must NEVER be deferred.

    Imported lazily because ``toolsets`` imports from ``tools.registry``
    and we don't want a hard cycle.
    """
    try:
        from toolsets import _HERMES_CORE_TOOLS
        return frozenset(_HERMES_CORE_TOOLS)
    except Exception:
        return frozenset()


def is_deferrable_tool_name(name: str) -> bool:
    """Return True if a tool with this name is *eligible* for deferral.

    A tool is deferrable iff it is registered with an MCP toolset prefix
    OR it is not in ``_HERMES_CORE_TOOLS``. Core tools are never deferred
    even when their toolset is technically plugin-provided (this protects
    against accidental shadowing).
    """
    if name in BRIDGE_TOOL_NAMES:
        return False
    if name in _core_tool_names():
        return False
    # Check registry toolset for MCP prefix.
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is None:
            return False
        if entry.toolset.startswith("mcp-"):
            return True
        # Non-MCP, non-core → plugin tool, eligible.
        return True
    except Exception:
        return False


def classify_tools(
    tool_defs: List[Dict[str, Any]],
    *,
    always_visible: Iterable[str] = (),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a tool-defs list into (visible, deferrable).

    ``visible`` retains every tool that must stay in the model-facing array:
    every core tool, plus any tool we can't classify. ``deferrable`` is the
    candidate set for catalog entry.
    """
    pinned = frozenset(always_visible)
    visible: List[Dict[str, Any]] = []
    deferrable: List[Dict[str, Any]] = []
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if name in BRIDGE_TOOL_NAMES:
            # Should never happen — bridge tools are added after classification —
            # but be defensive.
            continue
        if name in pinned:
            visible.append(td)
            continue
        if is_deferrable_tool_name(name):
            deferrable.append(td)
        else:
            visible.append(td)
    return visible, deferrable


# ---------------------------------------------------------------------------
# Token estimation and threshold gate
# ---------------------------------------------------------------------------


def estimate_tokens_from_schemas(tool_defs: Iterable[Dict[str, Any]]) -> int:
    """Estimate the token cost of a tool-defs list via the chars/4 rule.

    Cheap and stable across providers. The number doesn't need to be exact —
    it gates the activate/skip decision, and a typical 200K context with a
    10% threshold means the decision flips around 20K tokens of schema.
    Order-of-magnitude precision is fine.
    """
    total_chars = 0
    for td in tool_defs:
        try:
            total_chars += len(json.dumps(td, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            total_chars += len(str(td))
    return int(math.ceil(total_chars / CHARS_PER_TOKEN))


def should_activate(
    config: ToolSearchConfig,
    deferrable_tokens: int,
    context_length: Optional[int],
) -> bool:
    """Decide whether tool search should activate for the current assembly.

    ``"off"`` skips unconditionally. ``"on"`` activates unconditionally
    (as long as there is at least one deferrable tool — there's no point
    swapping a no-op). ``"auto"`` activates when the deferrable schemas
    reach either the context-relative threshold or the absolute schema-token
    budget.
    """
    if config.enabled == "off":
        return False
    if deferrable_tokens <= 0:
        return False
    if config.enabled == "on":
        return True
    return deferrable_tokens >= _auto_activation_threshold_tokens(config, context_length)


def _auto_activation_threshold_tokens(
    config: ToolSearchConfig,
    context_length: Optional[int],
) -> int:
    """Return the first auto-mode threshold the schema surface can reach."""
    if not context_length or context_length <= 0:
        return config.threshold_tokens
    relative_threshold = int(context_length * (config.threshold_pct / 100.0))
    return min(relative_threshold, config.threshold_tokens)


# ---------------------------------------------------------------------------
# Catalog + fielded BM25 retrieval with reciprocal-rank fusion
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    """One deferrable tool, in a form the bridge tools can search and serve."""

    name: str
    description: str
    schema: Dict[str, Any]  # The full {"type":"function", "function": {...}} entry.
    source: str  # "mcp" | "plugin" | "other"
    source_name: str  # Toolset name, e.g. "mcp-github" or "kanban"

    # Pre-tokenized fields for BM25.
    _tokens: List[str] = field(default_factory=list)


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _entry_search_text(td: Dict[str, Any]) -> str:
    """Build the search-text blob for a deferrable tool.

    Includes the tool name (with underscores broken into words so BM25 can
    match against query terms), the description, and the names of the
    top-level parameters. Schema bodies are deliberately excluded —
    indexing them adds noise without improving recall in our measurement.
    """
    fn = td.get("function") or {}
    name = fn.get("name", "")
    desc = fn.get("description", "") or ""
    params = ((fn.get("parameters") or {}).get("properties") or {})
    param_names = " ".join(params.keys())
    # Break snake_case and dotted names into words for BM25.
    name_words = name.replace("_", " ").replace(".", " ").replace("-", " ").replace(":", " ")
    return f"{name_words} {desc} {param_names}"


def _classify_source(name: str) -> Tuple[str, str]:
    """Return (source_kind, source_name) for a registered tool name."""
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is None:
            return ("other", "")
        if entry.toolset.startswith("mcp-"):
            return ("mcp", entry.toolset)
        return ("plugin", entry.toolset)
    except Exception:
        return ("other", "")


def build_catalog(tool_defs: List[Dict[str, Any]]) -> List[CatalogEntry]:
    """Build the deferred-tool catalog from a tool-defs list.

    Caller is expected to pass only the deferrable subset (``classify_tools``
    returns it as the second element).
    """
    catalog: List[CatalogEntry] = []
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        desc = fn.get("description", "") or ""
        source, source_name = _classify_source(name)
        entry = CatalogEntry(
            name=name,
            description=desc,
            schema=td,
            source=source,
            source_name=source_name,
            _tokens=_tokenize(_entry_search_text(td)),
        )
        catalog.append(entry)
    return catalog


def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                doc_lengths: List[int], avg_dl: float,
                doc_freq: Dict[str, int], n_docs: int,
                k1: float = 1.5, b: float = 0.75) -> float:
    """Standard BM25 score for one query against one document.

    Inlined small implementation rather than adding a dependency. Performance
    is fine — the catalog is bounded by N (tools) typically < 500, and we
    score against the in-memory tokens list.
    """
    if not doc_tokens:
        return 0.0
    score = 0.0
    dl = len(doc_tokens)
    # Pre-count tokens in the doc.
    doc_tf: Dict[str, int] = {}
    for t in doc_tokens:
        doc_tf[t] = doc_tf.get(t, 0) + 1
    for q in query_tokens:
        df = doc_freq.get(q, 0)
        if df == 0:
            continue
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        tf = doc_tf.get(q, 0)
        if tf == 0:
            continue
        norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0)))
        score += idf * norm
    return score


def _field_tokens(entry: CatalogEntry) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return name, description, parameter, and combined token streams."""
    fn = entry.schema.get("function") or {}
    name = str(fn.get("name") or entry.name)
    name_words = (
        name.replace("_", " ")
        .replace(".", " ")
        .replace("-", " ")
        .replace(":", " ")
    )
    description = str(fn.get("description") or entry.description or "")
    properties = ((fn.get("parameters") or {}).get("properties") or {})
    parameter_names = " ".join(str(key) for key in properties)
    return (
        _tokenize(name_words),
        _tokenize(description),
        _tokenize(parameter_names),
        entry._tokens
        or _tokenize(f"{name_words} {description} {parameter_names}"),
    )


def _bm25_ranking(
    catalog: List[CatalogEntry],
    query_tokens: List[str],
    documents: List[List[str]],
) -> List[CatalogEntry]:
    """Rank one catalog field with BM25, excluding zero-score documents."""
    doc_lengths = [len(tokens) for tokens in documents]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq: Dict[str, int] = {}
    for tokens in documents:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    scored: List[Tuple[float, CatalogEntry]] = []
    for entry, tokens in zip(catalog, documents):
        score = _bm25_score(
            query_tokens,
            tokens,
            doc_lengths,
            avg_dl,
            doc_freq,
            len(catalog),
        )
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [entry for _, entry in scored]


def _reciprocal_rank_fusion(
    rankings: List[List[CatalogEntry]],
    *,
    rank_constant: int = 60,
) -> List[CatalogEntry]:
    """Fuse independent field rankings without comparing unlike BM25 scores."""
    scores: Dict[str, float] = {}
    best_rank: Dict[str, int] = {}
    entries: Dict[str, CatalogEntry] = {}
    for ranking in rankings:
        for rank, entry in enumerate(ranking, start=1):
            entries[entry.name] = entry
            scores[entry.name] = scores.get(entry.name, 0.0) + (
                1.0 / (rank_constant + rank)
            )
            best_rank[entry.name] = min(best_rank.get(entry.name, rank), rank)
    return sorted(
        entries.values(),
        key=lambda entry: (
            -scores[entry.name],
            best_rank[entry.name],
            entry.name,
        ),
    )


def search_catalog(catalog: List[CatalogEntry], query: str, limit: int = 5) -> List[CatalogEntry]:
    """Return top catalog entries using fielded BM25 and reciprocal-rank fusion.

    Tool names, descriptions, and combined schema text are ranked separately.
    Parameter names remain searchable through the combined text, but do not
    receive an independent RRF vote: a generic argument such as ``file_id``
    should not outrank a tool whose name and description match the intent.
    A stable name-substring fallback covers punctuation-only or unusual names.
    """
    if not catalog or limit <= 0:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    fields = [_field_tokens(entry) for entry in catalog]
    rankings = [
        _bm25_ranking(catalog, query_tokens, [field[index] for field in fields])
        for index in (0, 1, 3)
    ]
    fused = _reciprocal_rank_fusion(rankings)

    if not fused:
        # Substring fallback against the original tool name.
        ql = query.lower()
        fused = sorted(
            (entry for entry in catalog if ql in entry.name.lower()),
            key=lambda entry: entry.name,
        )

    return fused[:limit]


# ---------------------------------------------------------------------------
# Bridge tool schemas
# ---------------------------------------------------------------------------


def bridge_tool_schemas(deferred_count: int) -> List[Dict[str, Any]]:
    """Build the bridge tool schemas to inject in place of deferred tools.

    The schemas are intentionally short — every byte added here is a byte
    the user pays on every turn. Descriptions are tuned to be unambiguous
    about the call sequence the model should follow.
    """
    desc_search = (
        f"Search {deferred_count} additional tools that are loaded on demand. "
        "Returns up to ``limit`` matches with name and description. Follow "
        f"with `{TOOL_DESCRIBE_NAME}` to load a tool's full parameter schema, "
        f"then `{TOOL_CALL_NAME}` to invoke it. Tools listed at the top of this "
        "system prompt are already available and do not need to be searched."
    )
    desc_describe = (
        f"Load the full JSON schema for one tool returned by `{TOOL_SEARCH_NAME}`. "
        f"Required before `{TOOL_CALL_NAME}` if the tool's parameters are unknown."
    )
    desc_call = (
        "Invoke a deferred tool by name with the given arguments. Argument shape "
        f"matches the tool's schema (see `{TOOL_DESCRIBE_NAME}`). Policy, hooks, "
        "and approvals run exactly as for any directly-listed tool."
    )

    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_SEARCH_NAME,
                "description": desc_search,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords describing the capability you need (e.g. 'create github issue').",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return. Default 5.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_DESCRIBE_NAME,
                "description": desc_describe,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact tool name (as returned by tool_search).",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_CALL_NAME,
                "description": desc_call,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact tool name to invoke.",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments for the tool, matching its schema.",
                        },
                    },
                    "required": ["name", "arguments"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Public entry point: assemble tool-defs with optional tool search
# ---------------------------------------------------------------------------


@dataclass
class AssemblyResult:
    """Outcome of one assembly. Useful for tests and observability."""

    tool_defs: List[Dict[str, Any]]
    activated: bool
    deferred_count: int = 0
    deferred_tokens: int = 0
    threshold_tokens: int = 0


def assemble_tool_defs(
    tool_defs: List[Dict[str, Any]],
    *,
    context_length: Optional[int] = None,
    config: Optional[ToolSearchConfig] = None,
) -> AssemblyResult:
    """Return the tool-defs list the model should actually see.

    When tool search is inactive (off, no deferrable tools, or below
    threshold), this is a passthrough. When active, MCP and plugin tools
    are stripped from the visible list and replaced with the three bridge
    tools. Core tools are *never* deferred regardless of config.

    Idempotent: calling with bridge tools already in the input is a no-op
    (they classify as non-core/non-deferrable but their names are reserved,
    so they are filtered out of the deferrable set).
    """
    if config is None:
        config = load_config()

    # Defensive: strip any bridge tools that may already be in the list
    # (e.g. someone called assemble twice).
    incoming = [td for td in tool_defs
                if (td.get("function") or {}).get("name") not in BRIDGE_TOOL_NAMES]

    visible, deferrable = classify_tools(
        incoming,
        always_visible=config.always_visible,
    )
    if not deferrable:
        return AssemblyResult(tool_defs=incoming, activated=False)

    deferrable_tokens = estimate_tokens_from_schemas(deferrable)
    activated = should_activate(config, deferrable_tokens, context_length)
    threshold_tokens = (
        _auto_activation_threshold_tokens(config, context_length)
        if config.enabled == "auto"
        else 0
    )
    logger.info(
        "tool_search assembly: activated=%s visible=%d deferred=%d "
        "deferred_schema_tokens_estimate=%d activation_threshold_tokens=%d",
        activated,
        len(visible),
        len(deferrable),
        deferrable_tokens,
        threshold_tokens,
    )

    if not activated:
        return AssemblyResult(
            tool_defs=incoming,
            activated=False,
            deferred_count=len(deferrable),
            deferred_tokens=deferrable_tokens,
            threshold_tokens=threshold_tokens,
        )

    bridge = bridge_tool_schemas(len(deferrable))
    result = visible + bridge

    return AssemblyResult(
        tool_defs=result,
        activated=True,
        deferred_count=len(deferrable),
        deferred_tokens=deferrable_tokens,
        threshold_tokens=threshold_tokens,
    )


# ---------------------------------------------------------------------------
# Bridge tool dispatch
# ---------------------------------------------------------------------------


def is_bridge_tool(name: str) -> bool:
    return name in BRIDGE_TOOL_NAMES


def recover_deferred_direct_call(
    name: str,
    raw_args: Any,
    *,
    bridge_available: bool,
) -> Optional[Tuple[str, str]]:
    """Wrap a historical direct call to a deferred tool in ``tool_call``.

    A resumed session can contain direct MCP/plugin calls from a turn where
    progressive disclosure was inactive. If a later turn activates tool
    search, the model may reuse that exact tool name even though only the
    bridge is visible. The bridge still enforces the current session's scoped
    deferred catalog before dispatch.
    """
    if not bridge_available or not is_deferrable_tool_name(name):
        return None
    if raw_args is None or (
        isinstance(raw_args, str) and not raw_args.strip()
    ):
        arguments: Any = {}
    elif isinstance(raw_args, dict):
        arguments = raw_args
    elif isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
    else:
        return None
    if not isinstance(arguments, dict):
        return None
    return (
        TOOL_CALL_NAME,
        json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _format_search_hit(entry: CatalogEntry) -> Dict[str, Any]:
    return {
        "name": entry.name,
        "source": entry.source,
        "source_name": entry.source_name,
        # Cap description so a chatty MCP server doesn't blow up the result.
        "description": (entry.description or "")[:400],
    }


def dispatch_tool_search(args: Dict[str, Any],
                         *,
                         current_tool_defs: List[Dict[str, Any]],
                         config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_search`` bridge tool. Returns a JSON string."""
    if config is None:
        config = load_config()
    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    raw_limit = args.get("limit")
    if raw_limit is None:
        limit = config.search_default_limit
    else:
        limit = max(1, min(config.max_search_limit, _safe_int(raw_limit, config.search_default_limit)))

    _, deferrable = classify_tools(current_tool_defs)
    catalog = build_catalog(deferrable)
    hits = search_catalog(catalog, query, limit=limit)
    return json.dumps({
        "query": query,
        "total_available": len(catalog),
        "matches": [_format_search_hit(h) for h in hits],
    }, ensure_ascii=False)


def dispatch_tool_describe(args: Dict[str, Any],
                           *,
                           current_tool_defs: List[Dict[str, Any]]) -> str:
    """Execute the ``tool_describe`` bridge tool. Returns a JSON string."""
    name = str(args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name is required"}, ensure_ascii=False)
    if not is_deferrable_tool_name(name):
        return json.dumps({
            "error": (
                f"'{name}' is not a deferrable tool. If you see it in the tools list "
                "already, call it directly; otherwise check the spelling against tool_search."
            ),
        }, ensure_ascii=False)
    _, deferrable = classify_tools(current_tool_defs)
    for td in deferrable:
        fn = td.get("function") or {}
        if fn.get("name") == name:
            return json.dumps({
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }, ensure_ascii=False)
    return json.dumps({
        "error": f"'{name}' is not currently available. Re-run tool_search to refresh.",
    }, ensure_ascii=False)


def scoped_deferrable_names(tool_defs: List[Dict[str, Any]]) -> frozenset[str]:
    """Return the set of deferrable tool names present in ``tool_defs``.

    ``tool_defs`` is expected to be the *pre-assembly* tool list for the
    current session's toolset scope (i.e. what
    ``get_tool_definitions(skip_tool_search_assembly=True)`` returns for the
    session's enabled/disabled toolsets). The resulting set is the universe of
    tools the session may legitimately reach through ``tool_call``. Used as a
    scoping gate by both the ``model_tools`` bridge dispatch and the
    ``tool_executor`` unwrap so a restricted-toolset session can never invoke
    an out-of-scope tool via the bridge.
    """
    names: set[str] = set()
    for td in tool_defs:
        name = (td.get("function") or {}).get("name", "")
        if name and is_deferrable_tool_name(name):
            names.add(name)
    return frozenset(names)


def resolve_underlying_call(args: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Parse a ``tool_call`` invocation into (underlying_name, args, error_msg).

    Used by:
    * the dispatcher in ``model_tools.handle_function_call``,
    * the display layer (so the activity feed shows the underlying tool),
    * the trajectory recorder.

    On parse error, returns ``(None, {}, error_message)``.
    """
    name = str(args.get("name") or "").strip()
    if not name:
        return None, {}, "tool_call requires a 'name' argument"
    if name in BRIDGE_TOOL_NAMES:
        return None, {}, f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
    raw_args = args.get("arguments")
    if raw_args is None:
        raw_args = {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return None, {}, f"tool_call 'arguments' is not valid JSON: {e}"
    if not isinstance(raw_args, dict):
        return None, {}, "tool_call 'arguments' must be an object"
    if not is_deferrable_tool_name(name):
        return None, {}, (
            f"'{name}' is not a deferrable tool. If it appears in the model-facing tools "
            "list already, call it directly instead of via tool_call."
        )
    return name, raw_args, None


__all__ = [
    "TOOL_SEARCH_NAME",
    "TOOL_DESCRIBE_NAME",
    "TOOL_CALL_NAME",
    "BRIDGE_TOOL_NAMES",
    "ToolSearchConfig",
    "CatalogEntry",
    "AssemblyResult",
    "load_config",
    "is_deferrable_tool_name",
    "classify_tools",
    "estimate_tokens_from_schemas",
    "should_activate",
    "build_catalog",
    "search_catalog",
    "bridge_tool_schemas",
    "assemble_tool_defs",
    "is_bridge_tool",
    "recover_deferred_direct_call",
    "dispatch_tool_search",
    "dispatch_tool_describe",
    "resolve_underlying_call",
    "scoped_deferrable_names",
]

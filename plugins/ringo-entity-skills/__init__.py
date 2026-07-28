"""Exact-identity entity skill loading and review confinement."""

from __future__ import annotations

import json

from hermes_constants import get_hermes_home

from .service import EntitySkillService


_service = EntitySkillService(get_hermes_home())


def _prepare(*, request: dict, **_: object) -> dict:
    return _service.prepare(request=request)


def _finish(*, request: dict, **_: object) -> dict:
    return _service.finish(request=request)


def _context(*, request: dict, **_: object) -> dict:
    return _service.context(request=request)


def _pre_llm_call(**kwargs: object) -> dict | None:
    return _service.inject_context(**kwargs)


def _pre_tool_call(**kwargs: object) -> dict | None:
    return _service.authorize_tool(**kwargs)


def _post_tool_call(**kwargs: object) -> None:
    _service.observe_tool(**kwargs)


def _health_report(**_: object) -> dict:
    return _service.health()


def _fail_closed_tool(*_: object, **__: object) -> str:
    return json.dumps({"error": "channel_skill_access_denied"})


def register(ctx) -> None:
    ctx.register_action("ringo.entity_skills.prepare", _prepare)
    ctx.register_action("ringo.entity_skills.finish", _finish)
    ctx.register_action("ringo.entity_skills.context", _context)
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("health_report", _health_report)
    ctx.register_tool(
        name="channel_skill_search",
        toolset="ringo_entity_skills",
        schema={
            "description": (
                "Search concise metadata and summaries from private channel "
                "skills the signed current DM caller may read. Search before "
                "using channel_skill_read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short terms describing the durable context.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        handler=_fail_closed_tool,
        description="Search authorized private channel skill summaries.",
        emoji="🔎",
    )
    ctx.register_tool(
        name="channel_skill_read",
        toolset="ringo_entity_skills",
        schema={
            "description": (
                "Read one private channel skill selected from "
                "channel_skill_search. Live membership is rechecked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": (
                            "Exact channel_id returned by channel_skill_search."
                        ),
                    }
                },
                "required": ["channel_id"],
                "additionalProperties": False,
            },
        },
        handler=_fail_closed_tool,
        description="Read one selected authorized private channel skill.",
        emoji="📚",
    )

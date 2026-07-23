"""Deterministic IE skill synchronization service plugin."""

from __future__ import annotations

import os

from gateway.event_ingress import read_project_marker
from hermes_constants import get_hermes_home

from .sync import SkillSyncService


_service = SkillSyncService(get_hermes_home())


def _reconcile(*, request: dict, **_: object) -> dict:
    return _service.reconcile(request=request)


def _post_tool_call(**kwargs: object) -> None:
    _service.observe_tool(**kwargs)


def _project_claimed(*, project_id: str, **_: object) -> dict:
    agent_id = (os.environ.get("RINGO_AGENT_ID") or "").strip()
    if not agent_id:
        return {"status": "waiting_for_agent"}
    _service.queue_reconcile(project_id=project_id, agent_id=agent_id)
    return {"status": "queued"}


def _health_report(**_: object) -> dict:
    return _service.health()


def register(ctx) -> None:
    ctx.register_action("ringo.skill.reconcile", _reconcile)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("project_claimed", _project_claimed)
    ctx.register_hook("health_report", _health_report)

    marker = read_project_marker()
    agent_id = (os.environ.get("RINGO_AGENT_ID") or "").strip()
    if marker is not None and agent_id:
        _service.queue_reconcile(
            project_id=str(marker["project_id"]),
            agent_id=agent_id,
        )

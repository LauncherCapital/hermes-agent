"""Exact-session public-channel memory synchronization plugin."""

from __future__ import annotations

from hermes_constants import get_hermes_home

from .sync import ChannelMemorySyncService


_service = ChannelMemorySyncService(get_hermes_home())


def _prepare(*, request: dict, **_: object) -> dict:
    return _service.prepare(request=request)


def _pre_tool_call(**kwargs: object) -> dict | None:
    return _service.authorize_tool(**kwargs)


def _post_tool_call(**kwargs: object) -> None:
    _service.observe_tool(**kwargs)


def _health_report(**_: object) -> dict:
    return _service.health()


def register(ctx) -> None:
    ctx.register_action("ringo.channel_memory.prepare", _prepare)
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("health_report", _health_report)
    _service.resume_dirty()

import asyncio

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.mark.asyncio
async def test_async_plugin_hook_is_awaited_and_sync_hook_still_works():
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="test"), manager)

    async def async_hook(*, value, **_):
        await asyncio.sleep(0)
        return {"async": value}

    def sync_hook(*, value, **_):
        return {"sync": value}

    context.register_hook("ingress_event", async_hook)
    context.register_hook("ingress_event", sync_hook)

    assert await manager.invoke_hook_async("ingress_event", value=7) == [
        {"async": 7},
        {"sync": 7},
    ]

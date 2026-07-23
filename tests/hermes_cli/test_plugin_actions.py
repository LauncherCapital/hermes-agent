import asyncio
import threading

import pytest

from hermes_cli.plugins import (
    PluginActionUnavailable,
    PluginContext,
    PluginManager,
    PluginManifest,
)


@pytest.mark.asyncio
async def test_sync_plugin_action_runs_off_the_gateway_event_loop():
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="fixture"), manager)
    event_loop_thread = threading.get_ident()

    def reconcile(*, request):
        return {
            "thread": threading.get_ident(),
            "revision": request["payload"]["revision"],
        }

    context.register_action("fixture.reconcile", reconcile)

    result = await manager.invoke_action(
        "fixture.reconcile",
        request={"payload": {"revision": 7}},
    )

    assert result["revision"] == 7
    assert result["thread"] != event_loop_thread


@pytest.mark.asyncio
async def test_async_plugin_action_is_awaited():
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="fixture"), manager)

    async def reconcile(*, request):
        await asyncio.sleep(0)
        return {"revision": request["payload"]["revision"]}

    context.register_action("fixture.reconcile", reconcile)

    assert await manager.invoke_action(
        "fixture.reconcile",
        request={"payload": {"revision": 8}},
    ) == {"revision": 8}


@pytest.mark.asyncio
async def test_unknown_action_does_not_invoke_another_handler():
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="fixture"), manager)
    called = False

    def reconcile(*, request):
        nonlocal called
        called = True
        return request

    context.register_action("fixture.reconcile", reconcile)

    with pytest.raises(PluginActionUnavailable):
        await manager.invoke_action("fixture.unknown", request={"payload": {}})

    assert called is False


def test_duplicate_or_invalid_action_registration_is_rejected():
    manager = PluginManager()
    first = PluginContext(PluginManifest(name="first"), manager)
    second = PluginContext(PluginManifest(name="second"), manager)

    first.register_action("fixture.reconcile", lambda **_: {})

    with pytest.raises(ValueError, match="already registered"):
        second.register_action("fixture.reconcile", lambda **_: {})
    with pytest.raises(ValueError, match="invalid plugin action"):
        first.register_action("Fixture Reconcile", lambda **_: {})
    with pytest.raises(ValueError, match="invalid plugin action"):
        first.register_action("reconcile", lambda **_: {})
    with pytest.raises(ValueError, match="invalid plugin action"):
        first.register_action(7, lambda **_: {})


@pytest.mark.asyncio
async def test_bundled_service_plugin_registers_and_handles_action(
    tmp_path, monkeypatch
):
    bundled = tmp_path / "bundled"
    plugin_dir = bundled / "fixture-service"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "\n".join(
            [
                "name: fixture-service",
                "kind: service",
                "actions:",
                "  - fixture.reconcile",
            ]
        )
    )
    (plugin_dir / "__init__.py").write_text(
        "\n".join(
            [
                "def register(ctx):",
                "    def reconcile(*, request):",
                "        return {'revision': request['payload']['revision']}",
                "    ctx.register_action('fixture.reconcile', reconcile)",
            ]
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["fixture-service"]
    assert loaded.enabled is True
    assert loaded.actions_registered == ["fixture.reconcile"]
    assert await manager.invoke_action(
        "fixture.reconcile",
        request={"payload": {"revision": 9}},
    ) == {"revision": 9}

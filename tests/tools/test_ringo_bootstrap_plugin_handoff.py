from pathlib import Path


def test_stage2_hands_skill_materialization_to_plugin_when_agent_is_known():
    script = Path("docker/stage2-hook.sh").read_text(encoding="utf-8")

    assert 'RINGO_AGENT_ID' in script
    assert '"agent_id=${RINGO_AGENT_ID}"' in script
    assert '"skill_sync=plugin"' in script
    assert "/api/v1/agent/bootstrap" in script

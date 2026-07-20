# LauncherCapital fork delta

Catalog of every commit this fork carries on top of upstream
(NousResearch/hermes-agent) release **v2026.6.5** (`chore: release v0.16.0`).
Kept so upstream rebases know exactly what our delta is and why each piece
exists. Regenerate the raw list with:

```
git log --oneline v2026.6.5..main
```

When you land a new commit on `main`, append it to the matching section here
(or add a section). Upstream has since tagged v2026.7.1 / v2026.7.7 /
v2026.7.7.2 — when we rebase onto one of those, prune entries that upstream
absorbed and update the base version above.

| commit | change |
|---|---|
| `e8aa7f092` | **feat(deploy)**: Railway recipe image — CMD `gateway run` (exec-form), no Dockerfile `VOLUME`, seed Ringo ie-MCP into config.yaml from env on first boot; CI publishes an immutable SHA image to GHCR (~2-3min pull+boot vs ~7-8min source build), no floating `:latest` — upgrading = build → e2e → re-pin |
| `bdfbe7221` | **feat(ringo)**: ie integration — generic ie-bootstrap shim (ringo boot logic centralized in ie, fork stays thin) + env-overridable OpenRouter attribution headers (X-Title / HTTP-Referer) |
| `8d4e5703d` | **feat(cron-api)**: jobs HTTP surface — ie drives hermes jobs over /api/jobs: per-job model/provider, `enabled_toolsets`, `context_from`; manual run wakes the ticker ("run now" means now); prompt cap 5000→20000; push cron-run completions to ie instead of being polled (#2) |
| `ceb36aa7f` | **feat(admin)**: config live-sync — no-reboot reconfiguration from ie: keepalive tool-diff refresh + api_server MCP admin endpoints; /admin/config accepts `agent.reasoning_effort` and `mcp_servers` tools include/exclude |
| `42a09a080` | **fix(resilience)**: one bad tool schema can't 400 the whole agent — schema sanitizer passes JSON-Schema data keywords (`dependentRequired`/`const`/`default`) through untouched; drop-and-retry on provider schema rejection |
| `e39f7298d` | **fix(mcp)**: reconnect loop never permanently gives up; stable sessions reset the retry budget (remote MCP redeploys used to kill the connection for good until process restart) |
| (this commit) | **chore(fork)**: map suho@launcher.capital in AUTHOR_MAP + this catalog |

"""Local materialization and reverse synchronization for IE skills."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ALLOWED_SUBDIRS = {"assets", "references", "scripts", "templates"}
_MAX_FILE_COUNT = 20
_MAX_FILE_CHARS = 200_000
_MAX_FILES_TOTAL_CHARS = 1_000_000


class SkillSyncError(RuntimeError):
    """Raised when a snapshot cannot be applied safely."""


def _uuid(value: object, field: str) -> str:
    try:
        return str(uuid.UUID(str(value or "")))
    except (TypeError, ValueError) as exc:
        raise SkillSyncError(f"invalid {field}") from exc


def _safe_name(value: object) -> str:
    name = value.strip() if isinstance(value, str) else ""
    if _NAME_RE.fullmatch(name) is None:
        raise SkillSyncError("invalid skill name")
    return name


def _safe_rel(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SkillSyncError("invalid skill file path")
    path = Path(value)
    parts = path.parts
    if (
        path.is_absolute()
        or len(parts) < 2
        or parts[0] not in _ALLOWED_SUBDIRS
        or any(part in {"", ".", ".."} for part in parts)
        or path.as_posix() != value
    ):
        raise SkillSyncError("invalid skill file path")
    return value


def _skill_md(name: str, description: str, body: str) -> str:
    if not body.endswith("\n"):
        body += "\n"
    return "\n".join(
        [
            "---",
            f"name: {name}",
            "description: " + json.dumps(description, ensure_ascii=False),
            "metadata:",
            "  hermes:",
            "    source: ringo-ie",
            "---",
            "",
            body,
        ]
    )


def _content_hash(files: dict[str, str]) -> str:
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SkillSyncService:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.skills_dir = self.home / "skills"
        self.manifest_path = self.home / "state" / "ringo-skill-sync.json"
        self._lock = threading.RLock()
        self._queue: queue.Queue[tuple[str, bool] | tuple[None, dict]] = queue.Queue()
        self._queued: set[tuple[str, bool]] = set()
        self._worker: threading.Thread | None = None
        self._health: dict[str, Any] = {
            "name": "ringo_skill_sync",
            "status": "idle",
            "queue_depth": 0,
            "last_error": None,
            "conflicts": [],
        }

    def _load_manifest(self) -> dict[str, Any]:
        try:
            raw = self.manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"schema_version": SCHEMA_VERSION, "skills": {}}
        except OSError as exc:
            raise SkillSyncError("skill sync manifest is unreadable") from exc
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise SkillSyncError("skill sync manifest is malformed") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
            or not isinstance(payload.get("skills"), dict)
        ):
            raise SkillSyncError("unsupported skill sync manifest")
        return payload

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_name(
            self.manifest_path.name + ".ringo-tmp"
        )
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.manifest_path)
        os.chmod(self.manifest_path, 0o600)

    def _desired_files(self, skill: dict[str, Any]) -> tuple[str, dict[str, str]]:
        name = _safe_name(skill.get("name"))
        description = skill.get("description") or ""
        body = skill.get("body") or ""
        if not isinstance(description, str) or not isinstance(body, str):
            raise SkillSyncError(f"invalid content for skill {name}")
        raw_files = skill.get("files") or {}
        if not isinstance(raw_files, dict) or len(raw_files) > _MAX_FILE_COUNT:
            raise SkillSyncError(f"invalid files for skill {name}")
        files = {"SKILL.md": _skill_md(name, description, body)}
        total = 0
        for raw_path, content in raw_files.items():
            rel = _safe_rel(raw_path)
            if not isinstance(content, str) or len(content) > _MAX_FILE_CHARS:
                raise SkillSyncError(f"invalid file content for skill {name}")
            total += len(content)
            files[rel] = content
        if total > _MAX_FILES_TOTAL_CHARS:
            raise SkillSyncError(f"skill files too large for {name}")
        return name, files

    def _safe_target(self, name: str, rel: str) -> Path:
        _safe_name(name)
        if rel != "SKILL.md":
            _safe_rel(rel)
        return self.skills_dir / name / rel

    @staticmethod
    def _write_one(target: Path, content: str) -> bool:
        try:
            if target.read_text(encoding="utf-8") == content:
                return False
        except OSError:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".ringo-tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        return True

    def _remove_one(self, name: str, rel: str) -> bool:
        target = self._safe_target(name, rel)
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        parent = target.parent
        skill_root = self.skills_dir / name
        while parent != skill_root.parent:
            try:
                parent.rmdir()
            except OSError:
                break
            if parent == skill_root:
                break
            parent = parent.parent
        return True

    def _current_hash(self, name: str, paths: list[str]) -> str:
        current: dict[str, str] = {}
        for rel in paths:
            target = self._safe_target(name, rel)
            try:
                current[rel] = target.read_text(encoding="utf-8")
            except OSError:
                current[rel] = "<missing>"
        return _content_hash(current)

    def _bind_manifest(
        self,
        manifest: dict[str, Any],
        *,
        project_id: str,
        agent_id: str,
    ) -> None:
        existing_project = manifest.get("project_id")
        existing_agent = manifest.get("agent_id")
        if existing_project and existing_project != project_id:
            raise SkillSyncError("skill manifest project mismatch")
        if existing_agent and existing_agent != agent_id:
            raise SkillSyncError("skill manifest agent mismatch")
        manifest["project_id"] = project_id
        manifest["agent_id"] = agent_id

    def apply_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        blocked_names: set[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            raise SkillSyncError("unsupported skill snapshot")
        project_id = _uuid(snapshot.get("project_id"), "project_id")
        agent_id = _uuid(snapshot.get("agent_id"), "agent_id")
        revision = str(snapshot.get("revision") or "").strip()
        raw_skills = snapshot.get("skills")
        if not revision or not isinstance(raw_skills, list):
            raise SkillSyncError("invalid skill snapshot")
        blocked = blocked_names or set()

        with self._lock:
            manifest = self._load_manifest()
            self._bind_manifest(
                manifest,
                project_id=project_id,
                agent_id=agent_id,
            )
            previous = manifest["skills"]
            next_records: dict[str, dict[str, Any]] = {}
            desired_names: set[str] = set()
            changed = 0
            removed = 0

            for skill in raw_skills:
                if not isinstance(skill, dict):
                    raise SkillSyncError("invalid skill entry")
                name, files = self._desired_files(skill)
                if name in desired_names:
                    raise SkillSyncError("duplicate skill name")
                desired_names.add(name)
                if name in blocked:
                    if name in previous:
                        next_records[name] = previous[name]
                    continue
                old = previous.get(name) if isinstance(previous.get(name), dict) else {}
                old_paths = set(old.get("managed_paths") or [])
                new_paths = set(files)
                for rel in sorted(old_paths - new_paths):
                    removed += int(self._remove_one(name, rel))
                for rel, content in sorted(files.items()):
                    changed += int(
                        self._write_one(self._safe_target(name, rel), content)
                    )
                next_records[name] = {
                    "revision": str(skill.get("revision") or ""),
                    "origin": str(skill.get("origin") or ""),
                    "editable": bool(skill.get("editable")),
                    "managed_paths": sorted(new_paths),
                    "content_hash": _content_hash(files),
                }

            for name, record in previous.items():
                if name in desired_names or not isinstance(record, dict):
                    continue
                for rel in sorted(set(record.get("managed_paths") or [])):
                    removed += int(self._remove_one(name, rel))

            manifest["skills"] = next_records
            manifest["snapshot_revision"] = revision
            self._save_manifest(manifest)
            if changed or removed:
                self._clear_prompt_cache()
            self._health.update(
                {
                    "status": "ready",
                    "last_pull_revision": revision,
                    "last_error": None,
                }
            )
            return {
                "status": "ready",
                "revision": revision,
                "skills": len(next_records),
                "changed_files": changed,
                "removed_files": removed,
            }

    def _control(self) -> tuple[str, str]:
        base_url = (os.environ.get("RINGO_IE_MCP_URL") or "").strip().rstrip("/")
        api_key = (os.environ.get("RINGO_IE_MCP_KEY") or "").strip()
        if base_url.endswith("/mcp"):
            base_url = base_url[:-4]
        if not base_url or not api_key:
            raise SkillSyncError("IE control channel unavailable")
        return base_url, api_key

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _, api_key = self._control()
        data = (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        )
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read())
            except Exception:
                detail = {"status": "error", "code": f"http_{exc.code}"}
            if exc.code == 409:
                return {
                    "status": "conflict",
                    "detail": detail,
                }
            raise SkillSyncError(f"IE request failed: HTTP {exc.code}") from exc
        except Exception as exc:
            raise SkillSyncError(
                f"IE request failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise SkillSyncError("IE returned a non-object response")
        return payload

    def fetch_snapshot(self, agent_id: str) -> dict[str, Any]:
        base_url, _ = self._control()
        query = urllib.parse.urlencode({"agent_id": _uuid(agent_id, "agent_id")})
        return self._request_json(
            base_url + "/api/v1/agent/skills/snapshot?" + query
        )

    def read_local_skill(self, name: str) -> dict[str, Any]:
        name = _safe_name(name)
        skill_dir = self.skills_dir / name
        skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        description = ""
        body = skill_md
        if skill_md.startswith("---"):
            boundary = skill_md.find("\n---", 3)
            if boundary >= 0:
                frontmatter = skill_md[3:boundary]
                body = skill_md[boundary + 4 :].lstrip("\r\n")
                try:
                    import yaml

                    parsed = yaml.safe_load(frontmatter) or {}
                    if isinstance(parsed, dict):
                        description = str(parsed.get("description") or "")
                except Exception:
                    description = ""
        files: dict[str, str] = {}
        for subdir in sorted(_ALLOWED_SUBDIRS):
            root = skill_dir / subdir
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                rel = path.relative_to(skill_dir).as_posix()
                _safe_rel(rel)
                files[rel] = path.read_text(encoding="utf-8")
        if len(files) > _MAX_FILE_COUNT:
            raise SkillSyncError(f"too many local files for skill {name}")
        if any(len(content) > _MAX_FILE_CHARS for content in files.values()):
            raise SkillSyncError(f"local file too large for skill {name}")
        if sum(map(len, files.values())) > _MAX_FILES_TOTAL_CHARS:
            raise SkillSyncError(f"local files too large for skill {name}")
        return {
            "name": name,
            "description": description,
            "body": body,
            "files": files,
        }

    def update_local_revision(
        self,
        name: str,
        revision: str,
        *,
        content_hash: str | None = None,
        managed_paths: list[str] | None = None,
    ) -> None:
        with self._lock:
            manifest = self._load_manifest()
            record = manifest["skills"].get(name)
            if not isinstance(record, dict):
                local = self.read_local_skill(name)
                paths = managed_paths or ["SKILL.md", *sorted(local["files"])]
                record = {
                    "origin": "agent",
                    "editable": True,
                    "managed_paths": paths,
                }
                manifest["skills"][name] = record
            elif managed_paths is not None:
                record["managed_paths"] = managed_paths
            record["revision"] = revision
            record["content_hash"] = (
                content_hash
                if content_hash is not None
                else self._current_hash(
                    name,
                    list(record.get("managed_paths") or []),
                )
            )
            self._save_manifest(manifest)

    def push_local(self, name: str, *, deleted: bool = False) -> dict[str, Any]:
        with self._lock:
            manifest = self._load_manifest()
            project_id = _uuid(manifest.get("project_id"), "project_id")
            agent_id = _uuid(manifest.get("agent_id"), "agent_id")
            record = manifest["skills"].get(name)
            if isinstance(record, dict) and not record.get("editable"):
                raise SkillSyncError(f"skill {name} is IE-managed")
            base_revision = (
                str(record.get("revision") or "") if isinstance(record, dict) else None
            )
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "name": _safe_name(name),
            "base_revision": base_revision,
            "deleted": deleted,
        }
        sent_hash: str | None = None
        sent_paths: list[str] | None = None
        if not deleted:
            local = self.read_local_skill(name)
            body.update(local)
            sent_paths = ["SKILL.md", *sorted(local["files"])]
            sent_hash = self._current_hash(name, sent_paths)
        base_url, _ = self._control()
        result = self._request_json(
            base_url + "/api/v1/agent/skills/corrections",
            method="POST",
            body=body,
        )
        if result.get("status") == "conflict":
            conflicts = list(self._health.get("conflicts") or [])
            if name not in conflicts:
                conflicts.append(name)
            self._health["conflicts"] = conflicts[-20:]
            return result
        if deleted:
            with self._lock:
                manifest = self._load_manifest()
                manifest["skills"].pop(name, None)
                self._save_manifest(manifest)
        else:
            revision = str(
                result.get("revision")
                or (result.get("skill") or {}).get("revision")
                or ""
            )
            if not revision:
                raise SkillSyncError("IE correction response omitted revision")
            self.update_local_revision(
                name,
                revision,
                content_hash=sent_hash,
                managed_paths=sent_paths,
            )
        self._health.update(
            {
                "status": "ready",
                "last_push_revision": result.get("revision"),
                "last_error": None,
            }
        )
        return result

    def _dirty_agent_skills(self) -> list[str]:
        with self._lock:
            manifest = self._load_manifest()
            dirty = []
            for name, record in manifest["skills"].items():
                if not isinstance(record, dict) or not record.get("editable"):
                    continue
                paths = list(record.get("managed_paths") or [])
                if self._current_hash(name, paths) != record.get("content_hash"):
                    dirty.append(name)
            return dirty

    def reconcile(self, *, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise SkillSyncError("invalid reconcile request")
        project_id = _uuid(request.get("project_id"), "project_id")
        payload = request.get("payload")
        if not isinstance(payload, dict):
            raise SkillSyncError("invalid reconcile payload")
        agent_id = _uuid(
            payload.get("agent_id")
            or os.environ.get("RINGO_AGENT_ID"),
            "agent_id",
        )
        blocked: set[str] = set()
        try:
            with self._lock:
                manifest = self._load_manifest()
                self._bind_manifest(
                    manifest,
                    project_id=project_id,
                    agent_id=agent_id,
                )
                self._save_manifest(manifest)
            for name in self._dirty_agent_skills():
                result = self.push_local(name)
                if result.get("status") == "conflict":
                    blocked.add(name)
            snapshot = self.fetch_snapshot(agent_id)
            if _uuid(snapshot.get("project_id"), "project_id") != project_id:
                raise SkillSyncError("snapshot project mismatch")
            if _uuid(snapshot.get("agent_id"), "agent_id") != agent_id:
                raise SkillSyncError("snapshot agent mismatch")
            result = self.apply_snapshot(snapshot, blocked_names=blocked)
            if blocked:
                result["conflicts"] = sorted(blocked)
            return result
        except Exception as exc:
            self._health.update(
                {
                    "status": "error",
                    "last_error": type(exc).__name__,
                }
            )
            raise

    def _run_worker(self) -> None:
        while True:
            name, value = self._queue.get()
            try:
                if name is None:
                    self.reconcile(request=value)
                else:
                    self.push_local(name, deleted=bool(value))
            except Exception as exc:
                self._health.update(
                    {
                        "status": "error",
                        "last_error": type(exc).__name__,
                    }
                )
            finally:
                if name is not None:
                    with self._lock:
                        self._queued.discard((name, bool(value)))
                self._queue.task_done()
                self._health["queue_depth"] = self._queue.qsize()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._run_worker,
                name="ringo-skill-sync",
                daemon=True,
            )
            self._worker.start()

    def queue_local_sync(self, name: str, deleted: bool = False) -> None:
        item = (_safe_name(name), bool(deleted))
        with self._lock:
            if item in self._queued:
                return
            self._queued.add(item)
            self._queue.put(item)
            self._health["queue_depth"] = self._queue.qsize()
        self._ensure_worker()

    def queue_reconcile(self, *, project_id: str, agent_id: str) -> None:
        request = {
            "project_id": _uuid(project_id, "project_id"),
            "payload": {"agent_id": _uuid(agent_id, "agent_id")},
        }
        self._queue.put((None, request))
        self._health["queue_depth"] = self._queue.qsize()
        self._ensure_worker()

    def observe_tool(
        self,
        *,
        tool_name: object = "",
        args: object = None,
        result: object = None,
        status: object = None,
        **_: object,
    ) -> None:
        if status != "ok" or not isinstance(args, dict):
            return
        parsed = result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except ValueError:
                parsed = result
        if isinstance(parsed, dict) and (
            parsed.get("success") is False or parsed.get("error")
        ):
            return
        if tool_name == "skill_manage":
            action = str(args.get("action") or "")
            if action not in {
                "create",
                "edit",
                "patch",
                "delete",
                "write_file",
                "remove_file",
            }:
                return
            name = args.get("name")
            if isinstance(name, str):
                self.queue_local_sync(name, deleted=action == "delete")
            return
        if tool_name not in {"patch", "write_file"}:
            return
        raw_path = args.get("path")
        if not isinstance(raw_path, str):
            return
        try:
            path = Path(raw_path).expanduser().resolve()
            relative = path.relative_to(self.skills_dir.resolve())
        except (OSError, ValueError):
            return
        if len(relative.parts) >= 2:
            self.queue_local_sync(relative.parts[0])

    def health(self) -> dict[str, Any]:
        return dict(self._health)

    @staticmethod
    def _clear_prompt_cache() -> None:
        try:
            from agent.prompt_builder import clear_skills_system_prompt_cache

            clear_skills_system_prompt_cache(clear_snapshot=True)
        except Exception:
            pass

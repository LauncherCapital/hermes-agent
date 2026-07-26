"""Project-bound, ACL-checked virtual entity ``SKILL.md`` routing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .storage import EncryptedSkillStore, EncryptedSkillStoreError


SCHEMA_VERSION = 2
LEASE_MINUTES = 15
MAX_SKILL_BYTES = 30_000
MAX_CONTEXT_BYTES = 60_000
MAX_COMPLETED_TURNS = 500
ENTITY_KINDS = ("users", "channels", "teams", "organizations")
_COMPONENT = re.compile(r"^[A-Za-z0-9._%-]{1,128}$")
_TEAM_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_LANGUAGE = re.compile(
    r"(?mi)^\s*language_preference\s*:\s*"
    r"(ko|en|ja|zh-CN|zh-TW)\s*$"
)
_RESTRICTED_TYPES = {"group", "private", "private_channel", "shared"}


class EntitySkillError(RuntimeError):
    """Raised when an entity skill operation is not safely scoped."""


def _uuid(value: object, field: str) -> str:
    try:
        return str(uuid.UUID(str(value or "")))
    except (TypeError, ValueError) as exc:
        raise EntitySkillError(f"invalid {field}") from exc


def _component(value: object, field: str) -> str:
    text = str(value or "").strip()
    if _COMPONENT.fullmatch(text) is None or text in {".", ".."}:
        raise EntitySkillError(f"invalid {field}")
    return text


def _optional_component(value: object, field: str) -> str:
    if value is None or value == "":
        return ""
    return _component(value, field)


def _optional_uuid(value: object, field: str) -> str:
    if value is None or value == "":
        return ""
    return _uuid(value, field)


def _hash(content: str | None) -> str | None:
    if content is None:
        return None
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ie_rest_base_url() -> str:
    base = str(os.getenv("RINGO_IE_MCP_URL") or "").rstrip("/")
    if base.endswith("/mcp"):
        base = base[:-4]
    return base


class EntitySkillService:
    def __init__(
        self,
        home: Path,
        *,
        access_checker: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.home = Path(home)
        self.skills_root = (self.home / "skills").resolve()
        self.manifest_path = self.home / "state" / "ringo-entity-skills.json"
        self.documents = EncryptedSkillStore(self.skills_root)
        self._access_checker = access_checker
        self._lock = threading.RLock()
        self._health: dict[str, Any] = {
            "name": "ringo_entity_skills",
            "status": "idle",
            "active_reviews": 0,
            "completed_reviews": 0,
            "changed_files": 0,
            "storage_encryption": "AES-256-GCM SKILL.md",
            "last_error": None,
        }

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "schema_version": SCHEMA_VERSION,
                "bindings": {},
                "completed_turns": [],
            }
        except (OSError, TypeError, ValueError) as exc:
            raise EntitySkillError("entity skill manifest is malformed") from exc
        if isinstance(payload, dict) and payload.get("schema_version") == 1:
            payload = {
                **{
                    key: payload[key]
                    for key in ("project_id", "agent_id", "workspace_id")
                    if key in payload
                },
                "schema_version": SCHEMA_VERSION,
                "bindings": {},
                "completed_turns": [
                    item
                    for item in payload.get("completed_turns") or []
                    if isinstance(item, str)
                ][-MAX_COMPLETED_TURNS:],
            }
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
            or not isinstance(payload.get("bindings"), dict)
            or not isinstance(payload.get("completed_turns"), list)
        ):
            raise EntitySkillError("unsupported entity skill manifest")
        return payload

    def _save(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_name(
            self.manifest_path.name + ".ringo-tmp"
        )
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                os.chmod(tmp, 0o600)
                json.dump(
                    manifest,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.manifest_path)
            os.chmod(self.manifest_path, 0o600)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _bind_identity(
        manifest: dict[str, Any],
        *,
        project_id: str,
        agent_id: str,
        workspace_id: str,
    ) -> None:
        for key, value in (
            ("project_id", project_id),
            ("agent_id", agent_id),
            ("workspace_id", workspace_id),
        ):
            if manifest.get(key) not in {None, value}:
                raise EntitySkillError(f"entity skill {key} mismatch")
            manifest[key] = value

    def _path(self, kind: str, entity_id: str) -> Path:
        if kind not in ENTITY_KINDS:
            raise EntitySkillError("invalid entity kind")
        component = (
            self._team_slug(entity_id)
            if kind == "teams"
            else _component(entity_id, f"{kind} id")
        )
        path = (self.skills_root / kind / component / "SKILL.md").resolve()
        if not path.is_relative_to(self.skills_root):
            raise EntitySkillError("entity skill path escapes root")
        return path

    @staticmethod
    def _team_slug(value: object) -> str:
        text = str(value or "").strip()
        if _TEAM_SLUG.fullmatch(text) is None:
            raise EntitySkillError("invalid team_slug")
        return text

    def _document(
        self,
        project_id: str,
        kind: str,
        entity_id: str,
    ) -> str | None:
        try:
            return self.documents.get(project_id, kind, entity_id)
        except EncryptedSkillStoreError as exc:
            raise EntitySkillError("encrypted entity skill store unavailable") from exc

    def _put_document(
        self,
        project_id: str,
        kind: str,
        entity_id: str,
        content: str,
    ) -> None:
        if len(content.encode("utf-8")) > MAX_SKILL_BYTES:
            raise EntitySkillError("entity skill exceeds size limit")
        try:
            self.documents.put(project_id, kind, entity_id, content)
        except EncryptedSkillStoreError as exc:
            raise EntitySkillError("encrypted entity skill store unavailable") from exc

    def _migrate_one(
        self,
        *,
        project_id: str,
        kind: str,
        entity_id: str,
    ) -> None:
        target = self._path(kind, entity_id)
        if not target.is_file() or target.is_symlink():
            return
        try:
            current = self._document(project_id, kind, entity_id)
        except EntitySkillError:
            try:
                candidate = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise EntitySkillError(
                    "plaintext entity skill could not be read"
                ) from exc
            if not candidate.lstrip().startswith(("---", "#")):
                raise
            current = candidate
        else:
            return
        self._put_document(project_id, kind, entity_id, current)

    def _encrypt_plaintext_context(
        self,
        *,
        project_id: str,
        workspace_id: str,
        user_id: str,
    ) -> None:
        self._migrate_one(
            project_id=project_id,
            kind="organizations",
            entity_id=workspace_id,
        )
        if user_id:
            self._migrate_one(
                project_id=project_id,
                kind="users",
                entity_id=user_id,
            )

    def _entities(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        workspace_id = _component(payload.get("workspace_id"), "workspace_id")
        entities: list[dict[str, str]] = []

        def add(kind: str, entity_id: str, **metadata: str) -> None:
            if any(
                item["kind"] == kind and item["id"] == entity_id
                for item in entities
            ):
                return
            entities.append(
                {
                    "kind": kind,
                    "id": entity_id,
                    "path": str(self._path(kind, entity_id)),
                    **metadata,
                }
            )

        if payload.get("include_organization") is True:
            add("organizations", workspace_id)
        user_id = _optional_component(payload.get("user_id"), "user_id")
        if user_id:
            add("users", user_id)

        current_channel = _optional_component(
            payload.get("channel_id"),
            "channel_id",
        )
        channel_type = str(payload.get("channel_type") or "").strip().lower()
        if current_channel and channel_type == "channel":
            add(
                "channels",
                current_channel,
                visibility="public",
                channel_type=channel_type,
            )
        restricted_channel = _optional_component(
            payload.get("restricted_channel_id"),
            "restricted_channel_id",
        )
        if restricted_channel:
            visibility = str(payload.get("channel_visibility") or "")
            if (
                restricted_channel != current_channel
                or channel_type not in _RESTRICTED_TYPES
                or visibility not in {"private", "restricted"}
            ):
                raise EntitySkillError("invalid restricted channel binding")
            add(
                "channels",
                restricted_channel,
                visibility=visibility,
                channel_type=channel_type,
            )

        raw_selected = payload.get("public_channel_ids") or []
        if not isinstance(raw_selected, list) or len(raw_selected) > 5:
            raise EntitySkillError("invalid public_channel_ids")
        for raw in raw_selected:
            add(
                "channels",
                _component(raw, "public channel id"),
                visibility="public",
                channel_type="channel",
            )

        team_slug = str(payload.get("team_slug") or "").strip()
        if team_slug:
            member_ids = payload.get("team_member_ids")
            if (
                payload.get("team_verified") is not True
                or not isinstance(member_ids, list)
                or not user_id
                or user_id
                not in {
                    _component(item, "team member id") for item in member_ids
                }
            ):
                raise EntitySkillError("team binding lacks explicit membership")
            add("teams", self._team_slug(team_slug))
        return entities

    @staticmethod
    def _prune(manifest: dict[str, Any]) -> None:
        now = _now()
        for session_id, binding in list(manifest["bindings"].items()):
            try:
                expires_at = datetime.fromisoformat(str(binding["expires_at"]))
            except (KeyError, TypeError, ValueError):
                expires_at = now
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= now:
                manifest["bindings"].pop(session_id, None)

    def _check_channel_access(
        self,
        *,
        project_id: str,
        agent_id: str,
        workspace_id: str,
        channel_id: str,
        channel_type: str,
        principal_id: str,
        slack_user_id: str,
        session_id: str,
        operation: str,
    ) -> dict[str, Any]:
        payload = {
            "agent_id": agent_id,
            "principal_id": principal_id or None,
            "workspace_id": workspace_id,
            "channel_id": channel_id,
            "channel_type": channel_type,
            "slack_user_id": slack_user_id or None,
            "session_id": session_id,
            "operation": operation,
        }
        try:
            if self._access_checker is not None:
                result = self._access_checker(payload)
            else:
                base = _ie_rest_base_url()
                key = str(os.getenv("RINGO_IE_MCP_KEY") or "")
                if not base or not key:
                    raise EntitySkillError("entity skill ACL service unavailable")
                response = httpx.post(
                    f"{base}/api/v1/agent/entity-skills/channel-access",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
        except Exception as exc:
            raise EntitySkillError("entity skill channel access denied") from exc
        if (
            not isinstance(result, dict)
            or result.get("authorized") is not True
            or str(result.get("project_id") or "") != project_id
            or str(result.get("agent_id") or "") != agent_id
            or str(result.get("workspace_id") or "") != workspace_id
            or str(result.get("channel_id") or "") != channel_id
            or str(result.get("session_id") or "") != session_id
            or str(result.get("operation") or "") != operation
        ):
            raise EntitySkillError("entity skill channel access denied")
        return result

    def prepare(self, *, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise EntitySkillError("invalid prepare request")
        project_id = _uuid(request.get("project_id"), "project_id")
        payload = request.get("payload")
        if not isinstance(payload, dict):
            raise EntitySkillError("invalid prepare payload")
        agent_id = _uuid(payload.get("agent_id"), "agent_id")
        workspace_id = _component(payload.get("workspace_id"), "workspace_id")
        user_id = _optional_component(payload.get("user_id"), "user_id")
        slack_user_id = _optional_component(
            payload.get("slack_user_id") or user_id,
            "slack_user_id",
        )
        principal_id = _optional_uuid(payload.get("principal_id"), "principal_id")
        session_id = _component(payload.get("session_id"), "session_id")
        turn_id = _component(payload.get("turn_id"), "turn_id")
        self._encrypt_plaintext_context(
            project_id=project_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        entities = self._entities(payload)
        paths = {item["path"] for item in entities}
        for item in entities:
            if item["kind"] == "channels":
                self._check_channel_access(
                    project_id=project_id,
                    agent_id=agent_id,
                    workspace_id=workspace_id,
                    channel_id=item["id"],
                    channel_type=item.get("channel_type", "channel"),
                    principal_id=principal_id,
                    slack_user_id=slack_user_id,
                    session_id=session_id,
                    operation="materialize",
                )
        for item in entities:
            self._migrate_one(
                project_id=project_id,
                kind=item["kind"],
                entity_id=item["id"],
            )

        with self._lock:
            manifest = self._load()
            self._bind_identity(
                manifest,
                project_id=project_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
            )
            self._prune(manifest)
            if turn_id in manifest["completed_turns"]:
                return {"status": "duplicate", "turn_id": turn_id}
            existing = manifest["bindings"].get(session_id)
            if isinstance(existing, dict) and existing.get("turn_id") == turn_id:
                return {
                    "status": "ready",
                    "turn_id": turn_id,
                    "entities": existing["entities"],
                }
            for binding in manifest["bindings"].values():
                active_paths = {
                    str(item.get("path") or "")
                    for item in (binding.get("entities") if isinstance(binding, dict) else [])
                    if isinstance(item, dict)
                }
                if paths & active_paths:
                    return {"status": "busy", "turn_id": turn_id}
            baseline = {
                item["path"]: _hash(
                    self._document(project_id, item["kind"], item["id"])
                )
                for item in entities
            }
            manifest["bindings"][session_id] = {
                "turn_id": turn_id,
                "expires_at": (_now() + timedelta(minutes=LEASE_MINUTES)).isoformat(),
                "entities": entities,
                "baseline": baseline,
                "changed": [],
                "project_id": project_id,
                "agent_id": agent_id,
                "workspace_id": workspace_id,
                "principal_id": principal_id,
                "slack_user_id": slack_user_id,
            }
            self._save(manifest)
            self._health.update(
                {
                    "status": "ready",
                    "active_reviews": len(manifest["bindings"]),
                    "last_error": None,
                }
            )
        return {"status": "ready", "turn_id": turn_id, "entities": entities}

    def finish(self, *, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise EntitySkillError("invalid finish request")
        project_id = _uuid(request.get("project_id"), "project_id")
        payload = request.get("payload")
        if not isinstance(payload, dict):
            raise EntitySkillError("invalid finish payload")
        session_id = _component(payload.get("session_id"), "session_id")
        turn_id = _component(payload.get("turn_id"), "turn_id")
        success = payload.get("success") is True
        with self._lock:
            manifest = self._load()
            if manifest.get("project_id") != project_id:
                raise EntitySkillError("entity skill project mismatch")
            self._prune(manifest)
            binding = manifest["bindings"].get(session_id)
            if not isinstance(binding, dict) or binding.get("turn_id") != turn_id:
                if success and turn_id in manifest["completed_turns"]:
                    return {"status": "duplicate", "turn_id": turn_id}
                raise EntitySkillError("entity skill review binding missing")
            changed = [
                path
                for path in binding.get("changed") or []
                if isinstance(path, str)
            ]
            if success:
                completed = [
                    item
                    for item in manifest["completed_turns"]
                    if isinstance(item, str) and item != turn_id
                ]
                manifest["completed_turns"] = [
                    *completed,
                    turn_id,
                ][-MAX_COMPLETED_TURNS:]
            manifest["bindings"].pop(session_id, None)
            self._save(manifest)
            self._health.update(
                {
                    "status": "ready",
                    "active_reviews": len(manifest["bindings"]),
                    "completed_reviews": len(manifest["completed_turns"]),
                    "changed_files": int(self._health["changed_files"])
                    + len(changed),
                    "last_error": None,
                }
            )
        return {
            "status": (
                "released"
                if not success
                else "applied"
                if changed
                else "no_change"
            ),
            "turn_id": turn_id,
            "changed": changed,
        }

    def _context_payload(
        self,
        *,
        project_id: str,
        agent_id: str,
        workspace_id: str,
        user_id: str = "",
        principal_id: str = "",
        channel_id: str = "",
        channel_type: str = "",
        team_slug: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        self._encrypt_plaintext_context(
            project_id=project_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        requested = [("organizations", workspace_id)]
        if user_id:
            requested.append(("users", user_id))
        if channel_id and channel_type not in {"im", "dm"}:
            self._check_channel_access(
                project_id=project_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                channel_id=channel_id,
                channel_type=channel_type,
                principal_id=principal_id,
                slack_user_id=user_id,
                session_id=session_id,
                operation="read",
            )
            requested.append(("channels", channel_id))
        if team_slug:
            requested.append(("teams", self._team_slug(team_slug)))

        documents: list[dict[str, str]] = []
        total = 0
        for kind, entity_id in requested:
            self._migrate_one(
                project_id=project_id,
                kind=kind,
                entity_id=entity_id,
            )
            content = self._document(project_id, kind, entity_id)
            if content is None:
                continue
            size = len(content.encode("utf-8"))
            if size > MAX_SKILL_BYTES or total + size > MAX_CONTEXT_BYTES:
                raise EntitySkillError("entity skill context exceeds size limit")
            total += size
            documents.append(
                {
                    "kind": kind,
                    "id": entity_id,
                    "path": str(self._path(kind, entity_id)),
                    "content": content,
                }
            )
        user_content = next(
            (
                item["content"]
                for item in documents
                if item["kind"] == "users" and item["id"] == user_id
            ),
            "",
        )
        match = _LANGUAGE.search(user_content)
        return {
            "status": "ready",
            "documents": documents,
            "language_preference": match.group(1) if match else None,
        }

    def context(self, *, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise EntitySkillError("invalid context request")
        project_id = _uuid(request.get("project_id"), "project_id")
        payload = request.get("payload")
        if not isinstance(payload, dict):
            raise EntitySkillError("invalid context payload")
        workspace_id = _component(payload.get("workspace_id"), "workspace_id")
        agent_id = _uuid(payload.get("agent_id"), "agent_id")
        with self._lock:
            manifest = self._load()
            self._bind_identity(
                manifest,
                project_id=project_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
            )
            self._save(manifest)
        return self._context_payload(
            project_id=project_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            user_id=_optional_component(payload.get("user_id"), "user_id"),
            principal_id=_optional_uuid(payload.get("principal_id"), "principal_id"),
            channel_id=_optional_component(payload.get("channel_id"), "channel_id"),
            channel_type=str(payload.get("channel_type") or ""),
            team_slug=str(payload.get("team_slug") or ""),
            session_id=_component(payload.get("session_id"), "session_id"),
        )

    def _binding_context(
        self,
        binding: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        documents: list[dict[str, str]] = []
        total = 0
        for item in binding.get("entities") or []:
            if not isinstance(item, dict):
                continue
            if item.get("kind") == "channels":
                self._check_channel_access(
                    project_id=binding["project_id"],
                    agent_id=binding["agent_id"],
                    workspace_id=binding["workspace_id"],
                    channel_id=item["id"],
                    channel_type=item.get("channel_type", "channel"),
                    principal_id=binding.get("principal_id", ""),
                    slack_user_id=binding.get("slack_user_id", ""),
                    session_id=session_id,
                    operation="read",
                )
            self._migrate_one(
                project_id=binding["project_id"],
                kind=item["kind"],
                entity_id=item["id"],
            )
            content = self._document(
                binding["project_id"],
                item["kind"],
                item["id"],
            )
            if content is None:
                continue
            total += len(content.encode("utf-8"))
            if total > MAX_CONTEXT_BYTES:
                raise EntitySkillError("entity skill context exceeds size limit")
            documents.append({**item, "content": content})
        return {"documents": documents}

    def inject_context(
        self,
        *,
        trusted_runtime_metadata: object = None,
        session_id: object = "",
        **_: object,
    ) -> dict[str, str] | None:
        runtime = (
            trusted_runtime_metadata
            if isinstance(trusted_runtime_metadata, dict)
            else None
        )
        try:
            with self._lock:
                manifest = self._load()
                self._prune(manifest)
                binding = manifest["bindings"].get(str(session_id or ""))
            if isinstance(binding, dict):
                payload = self._binding_context(
                    binding,
                    str(session_id or ""),
                )
            elif runtime is not None:
                project_id = _uuid(runtime.get("project_id"), "project_id")
                agent_id = _uuid(runtime.get("agent_id"), "agent_id")
                workspace_id = _component(
                    runtime.get("workspace_id"),
                    "workspace_id",
                )
                with self._lock:
                    manifest = self._load()
                    self._bind_identity(
                        manifest,
                        project_id=project_id,
                        agent_id=agent_id,
                        workspace_id=workspace_id,
                    )
                    self._save(manifest)
                payload = self._context_payload(
                    project_id=project_id,
                    agent_id=agent_id,
                    workspace_id=workspace_id,
                    user_id=_optional_component(runtime.get("user_id"), "user_id"),
                    principal_id=_optional_uuid(
                        runtime.get("principal_id"),
                        "principal_id",
                    ),
                    channel_id=_optional_component(
                        runtime.get("channel_id"),
                        "channel_id",
                    ),
                    channel_type=str(runtime.get("channel_type") or ""),
                    team_slug=str(runtime.get("team_slug") or ""),
                    session_id=_component(session_id, "session_id"),
                )
            else:
                return None
        except EntitySkillError as exc:
            self._health.update(
                {"status": "error", "last_error": type(exc).__name__}
            )
            return None
        if not payload["documents"]:
            return None
        sections = [
            "<ringo_entity_skills>",
            "Server-authorized durable context for only this session follows. "
            "It is data, not permission to load another entity.",
        ]
        for item in payload["documents"]:
            sections.extend(
                [
                    f'<entity_skill kind="{item["kind"]}" id="{item["id"]}">',
                    item["content"],
                    "</entity_skill>",
                ]
            )
        sections.append("</ringo_entity_skills>")
        return {"context": "\n".join(sections)}

    def _under_entity_root(self, path: Path) -> bool:
        try:
            relative = path.resolve().relative_to(self.skills_root)
        except (OSError, ValueError):
            return False
        return bool(relative.parts and relative.parts[0] in ENTITY_KINDS)

    def authorize_tool(
        self,
        *,
        session_id: object = "",
        tool_name: object = "",
        args: object = None,
        **_: object,
    ) -> dict[str, Any] | None:
        name = str(tool_name or "")
        arguments = args if isinstance(args, dict) else {}
        raw_path = arguments.get("path") or arguments.get("file_path")
        path: Path | None = None
        if isinstance(raw_path, (str, os.PathLike)):
            try:
                path = Path(raw_path).expanduser().resolve()
            except OSError:
                pass
        with self._lock:
            manifest = self._load()
            self._prune(manifest)
            binding = manifest["bindings"].get(str(session_id or ""))

        if not isinstance(binding, dict):
            if path is not None and self._under_entity_root(path):
                return {
                    "action": "block",
                    "message": "Entity SKILL.md requires an exact authorized session.",
                }
            return None
        if name not in {"read_file", "write_file", "patch"}:
            return {
                "action": "block",
                "message": "Entity review sessions only support bound SKILL.md tools.",
            }
        if path is None:
            return {
                "action": "block",
                "message": "An exact bound entity SKILL.md path is required.",
            }
        entity = next(
            (
                item
                for item in binding.get("entities") or []
                if isinstance(item, dict)
                and Path(str(item.get("path") or "")).resolve() == path
            ),
            None,
        )
        if entity is None:
            return {
                "action": "block",
                "message": "Tool path is outside this review's bound entities.",
            }
        try:
            if entity["kind"] == "channels":
                self._check_channel_access(
                    project_id=binding["project_id"],
                    agent_id=binding["agent_id"],
                    workspace_id=binding["workspace_id"],
                    channel_id=entity["id"],
                    channel_type=entity.get("channel_type", "channel"),
                    principal_id=binding.get("principal_id", ""),
                    slack_user_id=binding.get("slack_user_id", ""),
                    session_id=str(session_id or ""),
                    operation="read" if name == "read_file" else "write",
                )
            if name == "read_file":
                return {
                    "action": "handled",
                    "result": json.dumps(
                        {
                            "ok": True,
                            "message": (
                                "The current entity SKILL.md was supplied in "
                                "ephemeral review context."
                            ),
                        }
                    ),
                    "redact_args": True,
                }
            current = self._document(
                binding["project_id"],
                entity["kind"],
                entity["id"],
            )
            if name == "write_file":
                content = arguments.get("content")
                if not isinstance(content, str):
                    raise EntitySkillError("entity SKILL.md content is required")
                updated = content
            else:
                if arguments.get("mode", "replace") != "replace":
                    raise EntitySkillError("entity patch mode must be replace")
                old = arguments.get("old_string")
                new = arguments.get("new_string")
                if (
                    current is None
                    or not isinstance(old, str)
                    or not old
                    or not isinstance(new, str)
                    or old not in current
                ):
                    raise EntitySkillError("entity patch target not found")
                count = current.count(old)
                if count > 1 and arguments.get("replace_all") is not True:
                    raise EntitySkillError("entity patch target is ambiguous")
                updated = current.replace(
                    old,
                    new,
                    -1 if arguments.get("replace_all") is True else 1,
                )
            with self._lock:
                manifest = self._load()
                live = manifest["bindings"].get(str(session_id or ""))
                if (
                    not isinstance(live, dict)
                    or live.get("turn_id") != binding.get("turn_id")
                ):
                    raise EntitySkillError("entity review binding expired")
                self._put_document(
                    binding["project_id"],
                    entity["kind"],
                    entity["id"],
                    updated,
                )
                changed = [
                    item
                    for item in live.get("changed") or []
                    if isinstance(item, str)
                ]
                if str(path) not in changed:
                    changed.append(str(path))
                live["changed"] = changed
                self._save(manifest)
        except EntitySkillError:
            return {
                "action": "block",
                "message": "Entity SKILL.md operation denied.",
            }
        self._health["status"] = "editing"
        return {
            "action": "handled",
            "result": json.dumps(
                {"ok": True, "message": "Encrypted entity SKILL.md updated."}
            ),
            "redact_args": True,
        }

    def observe_tool(self, **_: object) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return dict(self._health)

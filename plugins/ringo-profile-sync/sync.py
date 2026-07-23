"""Local materialization and reverse synchronization for IE profile documents."""

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
_PERSON_LABELS = {
    "Preference": "preference",
    "Working style": "working_style",
}
_ORGANIZATION_LABELS = {
    "Working norms": "working_norm",
    "Observed culture": "observed_culture",
}
_DOCUMENT_TYPES = {
    "person_profile",
    "person_character",
    "organization_profile",
    "organization_character",
}
_MAX_DOCUMENT_CHARS = 1_000_000
_MAX_ENTRIES = 20
_MAX_ENTRY_CHARS = 500
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._%-]+$")


class ProfileSyncError(RuntimeError):
    """Raised when profile state cannot be reconciled safely."""


def _uuid(value: object, field: str) -> str:
    try:
        return str(uuid.UUID(str(value or "")))
    except (TypeError, ValueError) as exc:
        raise ProfileSyncError(f"invalid {field}") from exc


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _component(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text or _SAFE_COMPONENT.fullmatch(text) is None:
        raise ProfileSyncError(f"invalid {field}")
    return text


def _document_identity(document: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    document_type = str(document.get("type") or "")
    if document_type not in _DOCUMENT_TYPES:
        raise ProfileSyncError("invalid profile document type")
    target = document.get("target")
    if not isinstance(target, dict):
        raise ProfileSyncError("invalid profile target")
    if document_type.startswith("person_"):
        principal_id = _uuid(target.get("principal_id"), "principal_id")
        suffix = "profile.md" if document_type == "person_profile" else "CHARACTER.md"
        path = f"profiles/{principal_id}/{suffix}"
        document_id = (
            f"person:{principal_id}:"
            + ("profile" if document_type == "person_profile" else "character")
        )
        normalized_target = {"principal_id": principal_id}
    else:
        provider = _component(target.get("provider"), "provider")
        workspace_id = _component(target.get("workspace_id"), "workspace_id")
        suffix = (
            "profile.md"
            if document_type == "organization_profile"
            else "ORGANIZATION.md"
        )
        path = f"organizations/{provider}/{workspace_id}/{suffix}"
        document_id = (
            f"organization:{provider}:{workspace_id}:"
            + ("profile" if document_type == "organization_profile" else "character")
        )
        normalized_target = {
            "provider": provider,
            "workspace_id": workspace_id,
        }
    if document.get("id") != document_id or document.get("path") != path:
        raise ProfileSyncError("profile document identity mismatch")
    return document_id, path, normalized_target


def _editable(document_type: str) -> bool:
    return document_type in {"person_character", "organization_character"}


def _validate_entries(
    entries: object,
    *,
    document_type: str,
) -> list[dict[str, str]]:
    if not _editable(document_type):
        return []
    if not isinstance(entries, list) or len(entries) > _MAX_ENTRIES:
        raise ProfileSyncError("invalid profile entries")
    allowed = (
        set(_PERSON_LABELS.values())
        if document_type == "person_character"
        else set(_ORGANIZATION_LABELS.values())
    )
    normalized: dict[tuple[str, str], dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProfileSyncError("invalid profile entry")
        kind = str(entry.get("kind") or "").strip()
        value = " ".join(str(entry.get("value") or "").split())
        if kind not in allowed or not 2 <= len(value) <= _MAX_ENTRY_CHARS:
            raise ProfileSyncError("invalid profile entry")
        normalized[(kind, value.casefold())] = {"kind": kind, "value": value}
    return sorted(normalized.values(), key=lambda item: (item["kind"], item["value"]))


class ProfileSyncService:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.manifest_path = self.home / "state" / "ringo-profile-sync.json"
        self._lock = threading.RLock()
        self._queue: queue.Queue[tuple[str | None, object]] = queue.Queue()
        self._queued: set[str] = set()
        self._worker: threading.Thread | None = None
        self._health: dict[str, Any] = {
            "name": "ringo_profile_sync",
            "status": "idle",
            "queue_depth": 0,
            "last_error": None,
            "conflicts": [],
        }

    def _load_manifest(self) -> dict[str, Any]:
        try:
            raw = self.manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"schema_version": SCHEMA_VERSION, "documents": {}}
        except OSError as exc:
            raise ProfileSyncError("profile sync manifest is unreadable") from exc
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ProfileSyncError("profile sync manifest is malformed") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
            or not isinstance(payload.get("documents"), dict)
        ):
            raise ProfileSyncError("unsupported profile sync manifest")
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

    @staticmethod
    def _bind(
        manifest: dict[str, Any],
        *,
        project_id: str,
        agent_id: str,
    ) -> None:
        if manifest.get("project_id") not in {None, project_id}:
            raise ProfileSyncError("profile manifest project mismatch")
        if manifest.get("agent_id") not in {None, agent_id}:
            raise ProfileSyncError("profile manifest agent mismatch")
        manifest["project_id"] = project_id
        manifest["agent_id"] = agent_id

    def _target(self, relative: str) -> Path:
        path = Path(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or not path.parts
            or path.parts[0] not in {"profiles", "organizations"}
        ):
            raise ProfileSyncError("invalid profile path")
        return self.home / path

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

    def _remove_one(self, relative: str) -> bool:
        target = self._target(relative)
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        parent = target.parent
        while parent != self.home:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        return True

    def apply_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        blocked_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            raise ProfileSyncError("unsupported profile snapshot")
        project_id = _uuid(snapshot.get("project_id"), "project_id")
        agent_id = _uuid(snapshot.get("agent_id"), "agent_id")
        revision = str(snapshot.get("revision") or "").strip()
        raw_documents = snapshot.get("documents")
        if not revision or not isinstance(raw_documents, list):
            raise ProfileSyncError("invalid profile snapshot")
        blocked = blocked_ids or set()

        with self._lock:
            manifest = self._load_manifest()
            self._bind(
                manifest,
                project_id=project_id,
                agent_id=agent_id,
            )
            previous = manifest["documents"]
            next_records: dict[str, dict[str, Any]] = {}
            desired_ids: set[str] = set()
            changed = 0
            removed = 0

            for document in raw_documents:
                if not isinstance(document, dict):
                    raise ProfileSyncError("invalid profile document")
                document_id, relative, target = _document_identity(document)
                if document_id in desired_ids:
                    raise ProfileSyncError("duplicate profile document")
                desired_ids.add(document_id)
                if document_id in blocked:
                    if isinstance(previous.get(document_id), dict):
                        next_records[document_id] = previous[document_id]
                    continue
                document_type = str(document["type"])
                content = document.get("content")
                if not isinstance(content, str) or len(content) > _MAX_DOCUMENT_CHARS:
                    raise ProfileSyncError("invalid profile content")
                entries = _validate_entries(
                    document.get("entries"),
                    document_type=document_type,
                )
                changed += int(self._write_one(self._target(relative), content))
                next_records[document_id] = {
                    "type": document_type,
                    "path": relative,
                    "target": target,
                    "editable": _editable(document_type),
                    "revision": str(document.get("revision") or ""),
                    "content_hash": _content_hash(content),
                    "entries": entries,
                }

            for document_id, record in previous.items():
                if document_id in desired_ids or not isinstance(record, dict):
                    continue
                relative = record.get("path")
                if isinstance(relative, str):
                    removed += int(self._remove_one(relative))

            manifest["documents"] = next_records
            manifest["snapshot_revision"] = revision
            self._save_manifest(manifest)
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
                "documents": len(next_records),
                "changed_files": changed,
                "removed_files": removed,
            }

    def _control(self) -> tuple[str, str]:
        base_url = (os.environ.get("RINGO_IE_MCP_URL") or "").strip().rstrip("/")
        api_key = (os.environ.get("RINGO_IE_MCP_KEY") or "").strip()
        if base_url.endswith("/mcp"):
            base_url = base_url[:-4]
        if not base_url or not api_key:
            raise ProfileSyncError("IE control channel unavailable")
        return base_url, api_key

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _, api_key = self._control()
        data = json.dumps(body).encode() if body is not None else None
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
                return {"status": "conflict", "detail": detail}
            raise ProfileSyncError(f"IE request failed: HTTP {exc.code}") from exc
        except Exception as exc:
            raise ProfileSyncError(
                f"IE request failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise ProfileSyncError("IE returned a non-object response")
        return payload

    def fetch_snapshot(self, agent_id: str) -> dict[str, Any]:
        base_url, _ = self._control()
        query = urllib.parse.urlencode({"agent_id": _uuid(agent_id, "agent_id")})
        return self._request_json(
            base_url + "/api/v1/agent/profiles/snapshot?" + query
        )

    @staticmethod
    def _parse_entries(
        content: str,
        *,
        document_type: str,
    ) -> list[dict[str, str]]:
        labels = (
            _PERSON_LABELS
            if document_type == "person_character"
            else _ORGANIZATION_LABELS
        )
        current_kind: str | None = None
        entries: list[dict[str, str]] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                current_kind = labels.get(line[3:].strip())
                continue
            if not line.startswith("- ") or current_kind is None:
                continue
            value = " ".join(line[2:].split())
            if not value or value.casefold() == "none":
                continue
            entries.append({"kind": current_kind, "value": value})
        return _validate_entries(entries, document_type=document_type)

    def _dirty_documents(self) -> list[str]:
        with self._lock:
            manifest = self._load_manifest()
            dirty = []
            for document_id, record in manifest["documents"].items():
                if not isinstance(record, dict) or not record.get("editable"):
                    continue
                try:
                    current = self._target(record["path"]).read_text(encoding="utf-8")
                except OSError:
                    current = "<missing>"
                if _content_hash(current) != record.get("content_hash"):
                    dirty.append(document_id)
            return dirty

    def _mark_pushed(
        self,
        document_id: str,
        *,
        revision: str,
        content_hash: str,
        entries: list[dict[str, str]],
    ) -> None:
        with self._lock:
            manifest = self._load_manifest()
            record = manifest["documents"].get(document_id)
            if not isinstance(record, dict) or not record.get("editable"):
                raise ProfileSyncError("profile document is not editable")
            record["revision"] = revision
            record["content_hash"] = content_hash
            record["entries"] = entries
            self._save_manifest(manifest)

    def push_local(self, document_id: str) -> dict[str, Any]:
        with self._lock:
            manifest = self._load_manifest()
            agent_id = _uuid(manifest.get("agent_id"), "agent_id")
            record = manifest["documents"].get(document_id)
            if not isinstance(record, dict) or not record.get("editable"):
                raise ProfileSyncError("profile document is not editable")
            record = dict(record)
        try:
            content = self._target(record["path"]).read_text(encoding="utf-8")
        except OSError as exc:
            raise ProfileSyncError("editable profile document is missing") from exc
        entries = self._parse_entries(content, document_type=record["type"])
        sent_hash = _content_hash(content)
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "document_type": record["type"],
            "base_revision": record.get("revision"),
            "entries": entries,
            **record["target"],
        }
        base_url, _ = self._control()
        result = self._request_json(
            base_url + "/api/v1/agent/profiles/corrections",
            method="POST",
            body=body,
        )
        if result.get("status") == "conflict":
            conflicts = list(self._health.get("conflicts") or [])
            if document_id not in conflicts:
                conflicts.append(document_id)
            self._health["conflicts"] = conflicts[-20:]
            return result
        revision = str(result.get("revision") or "")
        if not revision:
            raise ProfileSyncError("IE correction response omitted revision")
        self._mark_pushed(
            document_id,
            revision=revision,
            content_hash=sent_hash,
            entries=entries,
        )
        try:
            current_hash = _content_hash(
                self._target(record["path"]).read_text(encoding="utf-8")
            )
        except OSError:
            current_hash = "<missing>"
        self._health.update(
            {
                "status": "ready",
                "last_push_revision": revision,
                "last_error": None,
            }
        )
        if current_hash != sent_hash:
            return {**result, "status": "local_changed"}
        return result

    def reconcile(self, *, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ProfileSyncError("invalid reconcile request")
        project_id = _uuid(request.get("project_id"), "project_id")
        payload = request.get("payload")
        if not isinstance(payload, dict):
            raise ProfileSyncError("invalid reconcile payload")
        agent_id = _uuid(
            payload.get("agent_id") or os.environ.get("RINGO_AGENT_ID"),
            "agent_id",
        )
        blocked: set[str] = set()
        conflicts: set[str] = set()
        deferred: set[str] = set()
        try:
            with self._lock:
                manifest = self._load_manifest()
                self._bind(
                    manifest,
                    project_id=project_id,
                    agent_id=agent_id,
                )
                self._save_manifest(manifest)
            for document_id in self._dirty_documents():
                result = self.push_local(document_id)
                if result.get("status") == "conflict":
                    blocked.add(document_id)
                    conflicts.add(document_id)
                elif result.get("status") == "local_changed":
                    blocked.add(document_id)
                    deferred.add(document_id)
            snapshot = self.fetch_snapshot(agent_id)
            if _uuid(snapshot.get("project_id"), "project_id") != project_id:
                raise ProfileSyncError("snapshot project mismatch")
            if _uuid(snapshot.get("agent_id"), "agent_id") != agent_id:
                raise ProfileSyncError("snapshot agent mismatch")
            result = self.apply_snapshot(snapshot, blocked_ids=blocked)
            if conflicts:
                result["conflicts"] = sorted(conflicts)
            if deferred:
                result["deferred"] = sorted(deferred)
                for document_id in deferred:
                    self.queue_local_sync(document_id)
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
            document_id, value = self._queue.get()
            try:
                if document_id is None:
                    self.reconcile(request=value)
                else:
                    self.push_local(document_id)
            except Exception as exc:
                self._health.update(
                    {
                        "status": "error",
                        "last_error": type(exc).__name__,
                    }
                )
            finally:
                if document_id is not None:
                    with self._lock:
                        self._queued.discard(document_id)
                self._queue.task_done()
                self._health["queue_depth"] = self._queue.qsize()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._run_worker,
                name="ringo-profile-sync",
                daemon=True,
            )
            self._worker.start()

    def queue_local_sync(self, document_id: str) -> None:
        with self._lock:
            if document_id in self._queued:
                return
            self._queued.add(document_id)
            self._queue.put((document_id, None))
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
        if (
            status != "ok"
            or tool_name not in {"patch", "write_file"}
            or not isinstance(args, dict)
        ):
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
        raw_path = args.get("path")
        if not isinstance(raw_path, str):
            return
        try:
            path = Path(raw_path).expanduser().resolve()
        except OSError:
            return
        with self._lock:
            manifest = self._load_manifest()
            for document_id, record in manifest["documents"].items():
                if (
                    isinstance(record, dict)
                    and record.get("editable")
                    and self._target(record["path"]).resolve() == path
                ):
                    self.queue_local_sync(document_id)
                    return

    def health(self) -> dict[str, Any]:
        return dict(self._health)

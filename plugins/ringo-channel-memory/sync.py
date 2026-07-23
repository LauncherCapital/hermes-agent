"""Local file materialization and reverse sync for one bound channel session."""

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
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._%-]+$")
_DOCUMENT = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}\.md$")
_TYPES = {"project", "decision", "terminology", "context"}
_MAX_DOCUMENTS = 12
_MAX_DOCUMENT_CONTENT = 30_000
_MAX_TOTAL_CONTENT = 100_000


class ChannelMemorySyncError(RuntimeError):
    """Raised when a channel binding or synchronization is unsafe."""


def _uuid(value: object, field: str) -> str:
    try:
        return str(uuid.UUID(str(value or "")))
    except (TypeError, ValueError) as exc:
        raise ChannelMemorySyncError(f"invalid {field}") from exc


def _component(value: object, field: str) -> str:
    text = str(value or "").strip()
    if (
        not text
        or text in {".", ".."}
        or _SAFE_COMPONENT.fullmatch(text) is None
    ):
        raise ChannelMemorySyncError(f"invalid {field}")
    return text


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _document_type(name: str, existing: dict[str, Any] | None) -> str:
    if existing and existing.get("type") in _TYPES:
        return str(existing["type"])
    if name == "decisions.md":
        return "decision"
    if name == "terminology.md":
        return "terminology"
    if name == "project-context.md":
        return "project"
    return "context"


def _summary(content: str, name: str) -> str:
    for raw in content.splitlines():
        line = " ".join(raw.strip().lstrip("#-* ").split())
        if line and line != "---":
            return line[:500]
    return name.removesuffix(".md").replace("-", " ")[:500]


class ChannelMemorySyncService:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.manifest_path = self.home / "state" / "ringo-channel-memory.json"
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._queued: set[str] = set()
        self._worker: threading.Thread | None = None
        self._health: dict[str, Any] = {
            "name": "ringo_channel_memory",
            "status": "idle",
            "queue_depth": 0,
            "last_error": None,
            "conflicts": [],
        }

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "schema_version": SCHEMA_VERSION,
                "channels": {},
                "bindings": {},
            }
        except (OSError, TypeError, ValueError) as exc:
            raise ChannelMemorySyncError("channel memory manifest is malformed") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
            or not isinstance(payload.get("channels"), dict)
            or not isinstance(payload.get("bindings"), dict)
        ):
            raise ChannelMemorySyncError("unsupported channel memory manifest")
        return payload

    def _save(self, manifest: dict[str, Any]) -> None:
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

    def _root(self, workspace_id: str, channel_id: str) -> Path:
        base = (self.home / "slack").resolve()
        root = (base / workspace_id / "channel" / channel_id).resolve()
        if not root.is_relative_to(base):
            raise ChannelMemorySyncError("channel memory path escapes volume")
        return root

    @staticmethod
    def _key(workspace_id: str, channel_id: str) -> str:
        return f"{workspace_id}:{channel_id}"

    @staticmethod
    def _bind_identity(
        manifest: dict[str, Any],
        *,
        project_id: str,
        agent_id: str,
    ) -> None:
        if manifest.get("project_id") not in {None, project_id}:
            raise ChannelMemorySyncError("channel memory project mismatch")
        if manifest.get("agent_id") not in {None, agent_id}:
            raise ChannelMemorySyncError("channel memory agent mismatch")
        manifest["project_id"] = project_id
        manifest["agent_id"] = agent_id

    @staticmethod
    def _write(target: Path, content: str) -> bool:
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

    def _control(self) -> tuple[str, str]:
        base_url = (os.environ.get("RINGO_IE_MCP_URL") or "").strip().rstrip("/")
        api_key = (os.environ.get("RINGO_IE_MCP_KEY") or "").strip()
        if base_url.endswith("/mcp"):
            base_url = base_url[:-4]
        if not base_url or not api_key:
            raise ChannelMemorySyncError("IE control channel unavailable")
        return base_url, api_key

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _, api_key = self._control()
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode() if body is not None else None,
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
            if exc.code == 409:
                return {"status": "conflict"}
            raise ChannelMemorySyncError(
                f"IE request failed: HTTP {exc.code}"
            ) from exc
        except Exception as exc:
            raise ChannelMemorySyncError(
                f"IE request failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict):
            raise ChannelMemorySyncError("IE returned a non-object response")
        return payload

    def fetch_snapshot(
        self,
        *,
        agent_id: str,
        workspace_id: str,
        channel_id: str,
    ) -> dict[str, Any]:
        base_url, _ = self._control()
        query = urllib.parse.urlencode(
            {
                "agent_id": agent_id,
                "workspace_id": workspace_id,
                "channel_id": channel_id,
            }
        )
        return self._request_json(
            base_url + "/api/v1/agent/channel-memory/snapshot?" + query
        )

    def _apply_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        project_id: str,
        agent_id: str,
        workspace_id: str,
        channel_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            raise ChannelMemorySyncError("unsupported channel memory snapshot")
        if _uuid(snapshot.get("project_id"), "project_id") != project_id:
            raise ChannelMemorySyncError("channel memory snapshot project mismatch")
        if _uuid(snapshot.get("agent_id"), "agent_id") != agent_id:
            raise ChannelMemorySyncError("channel memory snapshot agent mismatch")
        revision = str(snapshot.get("revision") or "").strip()
        profile = snapshot.get("profile")
        if not revision or not isinstance(profile, dict):
            raise ChannelMemorySyncError("invalid channel memory snapshot")
        if (
            profile.get("provider") != "slack"
            or str(profile.get("workspace_id") or "") != workspace_id
            or str(profile.get("channel_id") or "") != channel_id
        ):
            raise ChannelMemorySyncError("channel memory snapshot scope mismatch")
        raw_documents = profile.get("documents")
        if not isinstance(raw_documents, list) or len(raw_documents) > _MAX_DOCUMENTS:
            raise ChannelMemorySyncError("invalid channel memory documents")

        root = self._root(workspace_id, channel_id)
        records: dict[str, dict[str, Any]] = {}
        changed = 0
        total = 0
        for document in raw_documents:
            if not isinstance(document, dict):
                raise ChannelMemorySyncError("invalid channel memory document")
            name = str(document.get("name") or "")
            content = document.get("content")
            document_type = str(document.get("type") or "")
            if (
                _DOCUMENT.fullmatch(name) is None
                or name == "MEMORY.md"
                or not isinstance(content, str)
                or len(content) > _MAX_DOCUMENT_CONTENT
                or document_type not in _TYPES
                or name in records
            ):
                raise ChannelMemorySyncError("invalid channel memory document")
            total += len(content)
            records[name] = {
                "type": document_type,
                "description": str(document.get("description") or ""),
                "summary": str(document.get("summary") or ""),
                "metadata": document.get("metadata") or {},
                "content_hash": _hash(content),
            }
            changed += int(self._write(root / name, content))
        if total > _MAX_TOTAL_CONTENT:
            raise ChannelMemorySyncError("channel memory content limit exceeded")

        with self._lock:
            manifest = self._load()
            self._bind_identity(
                manifest,
                project_id=project_id,
                agent_id=agent_id,
            )
            key = self._key(workspace_id, channel_id)
            previous = manifest["channels"].get(key) or {}
            for name in (previous.get("documents") or {}):
                if name in records:
                    continue
                try:
                    (root / name).unlink()
                    changed += 1
                except FileNotFoundError:
                    pass
            index = str(profile.get("index_markdown") or "")
            changed += int(self._write(root / "MEMORY.md", index))
            manifest["channels"][key] = {
                "workspace_id": workspace_id,
                "channel_id": channel_id,
                "revision": revision,
                "documents": records,
            }
            manifest["bindings"][session_id] = key
            while len(manifest["bindings"]) > 200:
                manifest["bindings"].pop(next(iter(manifest["bindings"])))
            self._save(manifest)
        self._health.update({"status": "ready", "last_error": None})
        return {
            "status": "ready" if changed else "unchanged",
            "root_path": str(root),
            "index_path": str(root / "MEMORY.md"),
            "documents": len(records),
            "revision": revision,
        }

    def prepare(self, *, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ChannelMemorySyncError("invalid prepare request")
        project_id = _uuid(request.get("project_id"), "project_id")
        payload = request.get("payload")
        if not isinstance(payload, dict):
            raise ChannelMemorySyncError("invalid prepare payload")
        agent_id = _uuid(payload.get("agent_id"), "agent_id")
        workspace_id = _component(payload.get("workspace_id"), "workspace_id")
        channel_id = _component(payload.get("channel_id"), "channel_id")
        session_id = _component(payload.get("session_id"), "session_id")
        key = self._key(workspace_id, channel_id)
        if self._dirty(key):
            result = self.push_local(key)
            if result.get("status") in {"conflict", "local_changed"}:
                raise ChannelMemorySyncError(
                    "channel memory has an unresolved local edit"
                )
        snapshot = self.fetch_snapshot(
            agent_id=agent_id,
            workspace_id=workspace_id,
            channel_id=channel_id,
        )
        return self._apply_snapshot(
            snapshot,
            project_id=project_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            channel_id=channel_id,
            session_id=session_id,
        )

    def _dirty(self, key: str) -> bool:
        with self._lock:
            manifest = self._load()
            channel = manifest["channels"].get(key)
            if not isinstance(channel, dict):
                return False
            documents = dict(channel.get("documents") or {})
        root = self._root(
            str(channel["workspace_id"]),
            str(channel["channel_id"]),
        )
        for name, record in documents.items():
            try:
                content = (root / name).read_text(encoding="utf-8")
            except OSError:
                return True
            if _hash(content) != (record or {}).get("content_hash"):
                return True
        return any(
            path.is_file()
            and path.name != "MEMORY.md"
            and _DOCUMENT.fullmatch(path.name)
            and path.name not in documents
            for path in root.iterdir()
        )

    def _documents(self, key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with self._lock:
            manifest = self._load()
            channel = manifest["channels"].get(key)
            if not isinstance(channel, dict):
                raise ChannelMemorySyncError("unknown channel memory binding")
            channel = dict(channel)
            known = dict(channel.get("documents") or {})
        workspace_id = str(channel["workspace_id"])
        channel_id = str(channel["channel_id"])
        root = self._root(workspace_id, channel_id)
        paths = sorted(
            path
            for path in root.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.name != "MEMORY.md"
            and _DOCUMENT.fullmatch(path.name)
        )
        if len(paths) > _MAX_DOCUMENTS:
            raise ChannelMemorySyncError("channel memory documents limit exceeded")
        documents = []
        total = 0
        for path in paths:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                raise ChannelMemorySyncError("empty channel memory document")
            if len(content) > _MAX_DOCUMENT_CONTENT:
                raise ChannelMemorySyncError(
                    "channel memory document content limit exceeded"
                )
            total += len(content)
            previous = known.get(path.name)
            summary = _summary(content, path.name)
            documents.append(
                {
                    "name": path.name,
                    "type": _document_type(path.name, previous),
                    "description": (
                        str((previous or {}).get("description") or "").strip()
                        or summary[:300]
                    ),
                    "summary": summary,
                    "content": content,
                    "metadata": (
                        (previous or {}).get("metadata")
                        or {"origin": "hermes_channel_curator"}
                    ),
                }
            )
        if total > _MAX_TOTAL_CONTENT:
            raise ChannelMemorySyncError("channel memory content limit exceeded")
        return channel, documents

    def push_local(self, key: str) -> dict[str, Any]:
        channel, documents = self._documents(key)
        sent_hashes = {
            document["name"]: _hash(document["content"]) for document in documents
        }
        with self._lock:
            manifest = self._load()
            agent_id = _uuid(manifest.get("agent_id"), "agent_id")
        workspace_id = str(channel["workspace_id"])
        channel_id = str(channel["channel_id"])
        base_url, _ = self._control()
        result = self._request_json(
            base_url + "/api/v1/agent/channel-memory/corrections",
            method="POST",
            body={
                "agent_id": agent_id,
                "workspace_id": workspace_id,
                "channel_id": channel_id,
                "base_revision": channel["revision"],
                "documents": documents,
            },
        )
        if result.get("status") == "conflict":
            conflicts = list(self._health.get("conflicts") or [])
            if key not in conflicts:
                conflicts.append(key)
            self._health["conflicts"] = conflicts[-20:]
            return result
        revision = str(result.get("revision") or "")
        profile = result.get("profile")
        if not revision or not isinstance(profile, dict):
            raise ChannelMemorySyncError("IE correction response is incomplete")
        _, current_documents = self._documents(key)
        current_hashes = {
            document["name"]: _hash(document["content"])
            for document in current_documents
        }
        if current_hashes != sent_hashes:
            return {**result, "status": "local_changed"}
        with self._lock:
            manifest = self._load()
            record = manifest["channels"].get(key)
            if not isinstance(record, dict):
                raise ChannelMemorySyncError("channel binding disappeared")
            record["revision"] = revision
            record["documents"] = {
                document["name"]: {
                    "type": document["type"],
                    "description": document["description"],
                    "summary": document["summary"],
                    "metadata": document["metadata"],
                    "content_hash": _hash(document["content"]),
                }
                for document in documents
            }
            self._write(
                self._root(workspace_id, channel_id) / "MEMORY.md",
                str(profile.get("index_markdown") or ""),
            )
            self._save(manifest)
        self._health.update(
            {
                "status": "ready",
                "last_push_revision": revision,
                "last_error": None,
            }
        )
        return result

    def _run_worker(self) -> None:
        while True:
            key = self._queue.get()
            retry = False
            try:
                retry = self.push_local(key).get("status") == "local_changed"
            except Exception as exc:
                self._health.update(
                    {"status": "error", "last_error": type(exc).__name__}
                )
            finally:
                with self._lock:
                    self._queued.discard(key)
                self._queue.task_done()
                self._health["queue_depth"] = self._queue.qsize()
            if retry:
                self.queue_sync(key)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._run_worker,
                name="ringo-channel-memory",
                daemon=True,
            )
            self._worker.start()

    def queue_sync(self, key: str) -> None:
        with self._lock:
            if key in self._queued:
                return
            self._queued.add(key)
            self._queue.put(key)
            self._health["queue_depth"] = self._queue.qsize()
        self._ensure_worker()

    def resume_dirty(self) -> int:
        """Recover persisted local edits after an instance/plugin restart."""
        try:
            with self._lock:
                keys = list(self._load()["channels"])
            dirty = [key for key in keys if self._dirty(key)]
            for key in dirty:
                self.queue_sync(key)
            return len(dirty)
        except Exception as exc:
            self._health.update(
                {"status": "error", "last_error": type(exc).__name__}
            )
            return 0

    def observe_tool(
        self,
        *,
        session_id: object = "",
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
        raw_path = args.get("path") or args.get("file_path")
        if not isinstance(raw_path, str):
            return
        try:
            path = Path(raw_path).expanduser().resolve()
        except OSError:
            return
        with self._lock:
            manifest = self._load()
            key = manifest["bindings"].get(str(session_id or ""))
            channel = manifest["channels"].get(key) if key else None
            if not isinstance(channel, dict):
                return
            root = self._root(
                str(channel["workspace_id"]),
                str(channel["channel_id"]),
            ).resolve()
        if (
            path.parent != root
            or path.name == "MEMORY.md"
            or _DOCUMENT.fullmatch(path.name) is None
        ):
            return
        self.queue_sync(str(key))

    def authorize_tool(
        self,
        *,
        session_id: object = "",
        tool_name: object = "",
        args: object = None,
        **_: object,
    ) -> dict[str, str] | None:
        """Confine a bound private curator session to its exact channel files."""
        with self._lock:
            manifest = self._load()
            key = manifest["bindings"].get(str(session_id or ""))
            channel = manifest["channels"].get(key) if key else None
        if not isinstance(channel, dict):
            return None
        name = str(tool_name or "")
        arguments = args if isinstance(args, dict) else {}
        if name not in {"read_file", "write_file", "patch"}:
            return {
                "action": "block",
                "message": "Private channel-memory sessions may only read or edit their bound channel files.",
            }
        if name == "patch" and arguments.get("mode", "replace") != "replace":
            return {
                "action": "block",
                "message": "Private channel-memory sessions may only patch one bound file at a time.",
            }
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str):
            return {
                "action": "block",
                "message": "A bound channel-memory file path is required.",
            }
        try:
            path = Path(raw_path).expanduser().resolve()
            root = self._root(
                str(channel["workspace_id"]),
                str(channel["channel_id"]),
            ).resolve()
        except OSError:
            path = None
            root = None
        allowed = bool(
            path is not None
            and root is not None
            and path.parent == root
            and (name == "read_file" or path.name != "MEMORY.md")
            and (
                path.name == "MEMORY.md"
                or _DOCUMENT.fullmatch(path.name) is not None
            )
        )
        if allowed:
            return None
        return {
            "action": "block",
            "message": "Tool path is outside the session's bound channel-memory files.",
        }

    def health(self) -> dict[str, Any]:
        return dict(self._health)

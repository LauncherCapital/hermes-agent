"""Bounded, metadata-first inspection of the active Hermes home."""

from __future__ import annotations

import os
import re
import stat
from collections import deque
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


MAX_DEPTH = 6
MAX_NODES = 500
MAX_PREVIEW_BYTES = 32_000

_CANONICAL_CHANNEL_SKILL = re.compile(
    r"^skills/channels/(?P<channel_id>[A-Za-z0-9._%-]{1,128})/SKILL\.md$"
)
_CANONICAL_CHANNEL_PATH = re.compile(
    r"^skills/channels(?:/[A-Za-z0-9._%-]{1,128}(?:/.*)?)?$"
)
_LEGACY_CHANNEL_PATH = re.compile(
    r"^slack/[^/]+/channel/[^/]+(?:/.*)?$"
)
_LOCKED_ROOTS = frozenset(
    {
        "credentials",
        "auth",
    }
)
_LOCKED_SUFFIXES = frozenset(
    {
        ".db",
        ".db-shm",
        ".db-wal",
        ".key",
        ".pem",
        ".p12",
        ".pfx",
        ".sqlite",
        ".sqlite3",
        ".sqlcipher",
        ".token",
    }
)
_TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".csv",
        ".html",
        ".ini",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)


class VolumeInspectorError(ValueError):
    def __init__(self, code: str, message: str, *, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _root_path(home: Path | None = None) -> Path:
    configured = Path(home) if home is not None else get_hermes_home()
    if configured.is_symlink():
        raise VolumeInspectorError(
            "symlink_denied",
            "Hermes home cannot be a symlink",
            status=403,
        )
    try:
        return configured.resolve(strict=configured.exists())
    except OSError as exc:
        raise VolumeInspectorError(
            "volume_unavailable",
            "Hermes home is unavailable",
            status=503,
        ) from exc


def _is_locked(relative_path: str) -> bool:
    parts = relative_path.split("/")
    lowered = [part.casefold() for part in parts]
    name = lowered[-1]
    suffixes = Path(name).suffixes
    return bool(
        lowered[0] in _LOCKED_ROOTS
        or any(part.startswith(".") for part in parts)
        or any(
            marker in name
            for marker in ("credential", "secret", "token", "api_key", "apikey")
        )
        or any("".join(suffixes).endswith(suffix) for suffix in _LOCKED_SUFFIXES)
    )


def classify_path(
    relative_path: str,
    *,
    node_type: str,
) -> dict[str, Any]:
    canonical = _CANONICAL_CHANNEL_SKILL.fullmatch(relative_path)
    canonical_path = _CANONICAL_CHANNEL_PATH.fullmatch(relative_path)
    legacy = _LEGACY_CHANNEL_PATH.fullmatch(relative_path)
    provenance = (
        "canonical"
        if canonical_path
        else "legacy"
        if relative_path == "slack"
        or relative_path.startswith("slack/")
        else "other"
    )

    if node_type == "symlink":
        return {
            "category": "symlink",
            "status": "locked",
            "provenance": provenance,
            "previewable": False,
        }
    if _is_locked(relative_path):
        return {
            "category": "sensitive",
            "status": "locked",
            "provenance": provenance,
            "previewable": False,
        }
    if node_type == "directory":
        return {
            "category": "directory",
            "status": "browseable",
            "provenance": provenance,
            "previewable": False,
        }
    if canonical:
        return {
            "category": "channel_skill",
            "status": "previewable",
            "provenance": "canonical",
            "previewable": True,
        }
    if legacy:
        return {
            "category": "legacy_channel_memory",
            "status": "previewable",
            "provenance": "legacy",
            "previewable": True,
        }
    if Path(relative_path).suffix.casefold() in _TEXT_SUFFIXES:
        return {
            "category": "human_readable",
            "status": "previewable",
            "provenance": provenance,
            "previewable": True,
        }
    return {
        "category": "binary_or_state",
        "status": "locked",
        "provenance": provenance,
        "previewable": False,
    }


def _node_type(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def inspect_volume(
    home: Path | None = None,
    *,
    max_depth: int = MAX_DEPTH,
    max_nodes: int = MAX_NODES,
) -> dict[str, Any]:
    if max_depth < 1 or max_depth > MAX_DEPTH:
        raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")
    if max_nodes < 1 or max_nodes > MAX_NODES:
        raise ValueError(f"max_nodes must be between 1 and {MAX_NODES}")

    root = _root_path(home)
    nodes: list[dict[str, Any]] = []
    truncated = False

    pending: deque[tuple[Path, int]] = deque()
    if root.is_dir():
        pending.append((root, 1))

    while pending:
        directory, depth = pending.popleft()
        remaining = max_nodes - len(nodes)
        if remaining <= 0:
            truncated = True
            break
        try:
            with os.scandir(directory) as scan:
                entries = [
                    Path(entry.path)
                    for entry in islice(scan, remaining + 1)
                ]
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            continue
        if len(entries) > remaining:
            truncated = True
            entries = entries[:remaining]

        def sort_key(path: Path) -> tuple[bool, str]:
            try:
                is_dir = stat.S_ISDIR(path.lstat().st_mode)
            except OSError:
                is_dir = False
            return (not is_dir, path.name.casefold())

        child_directories: list[Path] = []
        for entry in sorted(entries, key=sort_key):
            try:
                metadata = entry.lstat()
                relative = entry.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            kind = _node_type(metadata.st_mode)
            classification = classify_path(relative, node_type=kind)
            nodes.append(
                {
                    "path": relative,
                    "name": entry.name,
                    "depth": depth,
                    "type": kind,
                    "size_bytes": metadata.st_size if kind != "directory" else 0,
                    "modified_at": datetime.fromtimestamp(
                        metadata.st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                    **classification,
                }
            )
            if kind != "directory" or classification["status"] == "locked":
                continue
            if depth >= max_depth:
                truncated = True
                continue
            child_directories.append(entry)
        pending.extend((child, depth + 1) for child in child_directories)

    if pending:
        truncated = True
    return {
        "root": "HERMES_HOME",
        "nodes": nodes,
        "truncated": truncated,
        "limits": {
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "max_preview_bytes": MAX_PREVIEW_BYTES,
        },
    }


def _relative_parts(raw_path: object) -> tuple[str, ...]:
    if not isinstance(raw_path, str):
        raise VolumeInspectorError("invalid_path", "path must be a string")
    path = raw_path.strip()
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
    ):
        raise VolumeInspectorError("invalid_path", "path must be relative")
    parts = tuple(path.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise VolumeInspectorError(
            "path_traversal_denied",
            "path traversal is not allowed",
            status=403,
        )
    return parts


def validate_preview_target(
    raw_path: object,
    home: Path | None = None,
) -> tuple[Path, str | None]:
    parts = _relative_parts(raw_path)
    relative = "/".join(parts)
    match = _CANONICAL_CHANNEL_SKILL.fullmatch(relative)
    if (
        _is_locked(relative)
        or (
            match is None
            and Path(relative).suffix.casefold() not in _TEXT_SUFFIXES
        )
    ):
        raise VolumeInspectorError(
            "preview_locked",
            "preview is not allowed for this file",
            status=403,
        )

    root = _root_path(home)
    current = root
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise VolumeInspectorError(
                "file_not_found",
                "preview file was not found",
                status=404,
            ) from exc
        except OSError as exc:
            raise VolumeInspectorError(
                "file_unavailable",
                "preview file is unavailable",
                status=503,
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise VolumeInspectorError(
                "symlink_denied",
                "symlink previews are not allowed",
                status=403,
            )
    if not current.is_file():
        raise VolumeInspectorError(
            "preview_locked",
            "preview is not allowed for this entry",
            status=403,
        )
    try:
        current.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise VolumeInspectorError(
            "path_escape_denied",
            "preview path escapes Hermes home",
            status=403,
        ) from exc
    return current, match.group("channel_id") if match is not None else None


def read_preview_file(
    target: Path,
    relative_path: str,
    max_bytes: int = MAX_PREVIEW_BYTES,
) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise VolumeInspectorError(
                    "preview_locked",
                    "preview is not allowed for this entry",
                    status=403,
                )
            encoded = stream.read(max_bytes + 1)
    except VolumeInspectorError:
        raise
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise VolumeInspectorError(
            "file_unavailable",
            "preview file is unavailable",
            status=503,
        ) from exc

    bounded = encoded[:max_bytes]
    if b"\x00" in bounded:
        raise VolumeInspectorError(
            "preview_locked",
            "preview is not allowed for binary content",
            status=403,
        )
    return {
        "path": relative_path,
        "content": bounded.decode("utf-8", errors="ignore"),
        "encoding": "utf-8",
        "truncated": len(encoded) > max_bytes,
    }


def truncate_utf8(
    content: str,
    max_bytes: int = MAX_PREVIEW_BYTES,
) -> tuple[str, bool]:
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True

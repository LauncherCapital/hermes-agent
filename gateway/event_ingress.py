"""Generic signed event ingress authentication and durable replay guard."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


MAX_EVENT_BYTES = 2_000_000
DEFAULT_MAX_SKEW_SECONDS = 300
REPLAY_RETENTION_SECONDS = 7 * 24 * 60 * 60
PROJECT_MARKER = "project.json"


class EventIngressError(ValueError):
    def __init__(self, code: str, message: str, *, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def state_dir() -> Path:
    path = get_hermes_home() / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_marker_path() -> Path:
    return state_dir() / PROJECT_MARKER


def read_project_marker() -> dict[str, Any] | None:
    try:
        data = json.loads(project_marker_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not str(data.get("project_id") or "").strip():
        return None
    return data


def write_project_marker(
    project_id: str,
    *,
    event_verifiers: dict[str, str] | None = None,
    recovery_public_keys: dict[str, str] | None = None,
    active_key_version: int | None = None,
) -> dict[str, Any]:
    """Claim a Volume once; a different project can never reuse it."""
    project_id = str(project_id or "").strip()
    if not project_id:
        raise EventIngressError("invalid_project", "project id is required")
    try:
        import uuid

        project_id = str(uuid.UUID(project_id))
    except (TypeError, ValueError) as exc:
        raise EventIngressError("invalid_project", "project id must be a UUID") from exc

    current = read_project_marker()
    if current and current["project_id"] != project_id:
        raise EventIngressError(
            "project_already_claimed",
            "this Hermes Volume is already claimed by another project",
            status=409,
        )
    marker = dict(current or {})
    marker["project_id"] = project_id
    if event_verifiers is not None:
        marker["event_verifiers"] = {
            str(key): str(value)
            for key, value in event_verifiers.items()
            if str(key).strip() and str(value).strip()
        }
    if recovery_public_keys is not None:
        marker["recovery_public_keys"] = {
            str(key): str(value)
            for key, value in recovery_public_keys.items()
            if str(key).strip() and str(value).strip()
        }
    if active_key_version is not None:
        try:
            version = int(active_key_version)
        except (TypeError, ValueError) as exc:
            raise EventIngressError(
                "invalid_key_version", "active key version must be a positive integer"
            ) from exc
        if version < 1:
            raise EventIngressError(
                "invalid_key_version", "active key version must be a positive integer"
            )
        marker["active_key_version"] = version
    marker.setdefault("claimed_at", int(time.time()))
    path = project_marker_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    return marker


def canonical_request(
    *,
    timestamp: str,
    project_id: str,
    body_sha256: str,
    method: str = "POST",
    path: str = "/v1/events",
) -> bytes:
    return "\n".join(
        [method.upper(), path, timestamp, project_id, body_sha256]
    ).encode("utf-8")


def _decode_signature(value: str) -> bytes:
    raw = str(value or "").strip()
    if not raw:
        raise EventIngressError("missing_signature", "event signature is required", status=401)
    try:
        return base64.b64decode(raw, validate=True)
    except ValueError:
        try:
            return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        except Exception as exc:
            raise EventIngressError("invalid_signature", "event signature is malformed", status=401) from exc


@dataclass(frozen=True)
class VerifiedEvent:
    envelope: dict[str, Any]
    delivery_id: str
    project_id: str
    body_sha256: str


def verify_event_request(
    body: bytes,
    headers: Any,
    *,
    now: float | None = None,
) -> VerifiedEvent:
    if len(body) > MAX_EVENT_BYTES:
        raise EventIngressError("event_too_large", "event body is too large", status=413)
    try:
        envelope = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise EventIngressError("invalid_json", "event body must be valid JSON") from exc
    if not isinstance(envelope, dict):
        raise EventIngressError("invalid_envelope", "event body must be a JSON object")

    marker = read_project_marker()
    if marker is None:
        raise EventIngressError("project_unclaimed", "runtime is not claimed", status=503)
    project_id = str(envelope.get("project_id") or "").strip()
    delivery_id = str(envelope.get("delivery_id") or "").strip()
    if project_id != marker["project_id"]:
        raise EventIngressError("project_mismatch", "event targets another project", status=403)
    if not delivery_id or len(delivery_id) > 128:
        raise EventIngressError("invalid_delivery_id", "delivery_id is required")

    timestamp = str(headers.get("X-Ringo-Timestamp") or "").strip()
    try:
        timestamp_value = int(timestamp)
    except ValueError as exc:
        raise EventIngressError("invalid_timestamp", "event timestamp is required", status=401) from exc
    if abs((now if now is not None else time.time()) - timestamp_value) > DEFAULT_MAX_SKEW_SECONDS:
        raise EventIngressError("stale_timestamp", "event timestamp is outside the allowed window", status=401)

    digest = hashlib.sha256(body).hexdigest()
    supplied_digest = str(headers.get("X-Ringo-Content-SHA256") or "").strip().lower()
    if not supplied_digest or not _constant_time_equal(supplied_digest, digest):
        raise EventIngressError("body_hash_mismatch", "event body hash does not match", status=401)

    key_id = str(headers.get("X-Ringo-Key-Id") or "").strip()
    pem = (marker.get("event_verifiers") or {}).get(key_id)
    if not key_id or not pem:
        raise EventIngressError("unknown_key", "event signing key is not trusted", status=401)
    signature = _decode_signature(headers.get("X-Ringo-Signature") or "")
    try:
        from cryptography.hazmat.primitives import serialization

        public_key = serialization.load_pem_public_key(str(pem).encode("utf-8"))
        public_key.verify(
            signature,
            canonical_request(
                timestamp=timestamp,
                project_id=project_id,
                body_sha256=digest,
            ),
        )
    except EventIngressError:
        raise
    except Exception as exc:
        raise EventIngressError("invalid_signature", "event signature is invalid", status=401) from exc
    return VerifiedEvent(
        envelope=envelope,
        delivery_id=delivery_id,
        project_id=project_id,
        body_sha256=digest,
    )


def _constant_time_equal(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left.encode("ascii", "ignore"), right.encode("ascii", "ignore"))


class EventReplayGuard:
    """Reserve signed requests and persist only successfully applied deliveries."""

    def __init__(self, path: Path | None = None):
        self.path = path or (state_dir() / "event_replays.db")
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS seen_requests ("
                "delivery_id TEXT PRIMARY KEY, body_sha256 TEXT NOT NULL, "
                "seen_at INTEGER NOT NULL, committed INTEGER NOT NULL DEFAULT 0)"
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(seen_requests)")
            }
            if "committed" not in columns:
                conn.execute(
                    "ALTER TABLE seen_requests "
                    "ADD COLUMN committed INTEGER NOT NULL DEFAULT 0"
                )
            # A row that was merely reserved by a process which no longer exists
            # is not proof that the plugin committed it. Replaying it is safe:
            # the project message store has its own atomic delivery dedup.
            conn.execute("DELETE FROM seen_requests WHERE committed = 0")
        os.chmod(self.path, 0o600)

    def reserve(self, delivery_id: str, body_sha256: str) -> str:
        with self._lock, self._connect() as conn:
            cutoff = int(time.time()) - REPLAY_RETENTION_SECONDS
            conn.execute("DELETE FROM seen_requests WHERE seen_at < ?", (cutoff,))
            try:
                conn.execute(
                    "INSERT INTO seen_requests("
                    "delivery_id, body_sha256, seen_at, committed"
                    ") VALUES (?, ?, ?, 0)",
                    (delivery_id, body_sha256, int(time.time())),
                )
                return "new"
            except sqlite3.IntegrityError:
                existing = conn.execute(
                    "SELECT body_sha256, committed FROM seen_requests "
                    "WHERE delivery_id = ?",
                    (delivery_id,),
                ).fetchone()
                if (
                    existing
                    and existing[0] == body_sha256
                    and bool(existing[1])
                ):
                    return "duplicate"
                if existing and existing[0] == body_sha256:
                    return "pending"
                return "conflict"

    def commit(self, delivery_id: str, body_sha256: str) -> None:
        with self._lock, self._connect() as conn:
            result = conn.execute(
                "UPDATE seen_requests SET committed = 1, seen_at = ? "
                "WHERE delivery_id = ? AND body_sha256 = ?",
                (int(time.time()), delivery_id, body_sha256),
            )
            if result.rowcount != 1:
                raise RuntimeError("event replay reservation was lost before commit")

    def release(self, delivery_id: str, body_sha256: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM seen_requests WHERE delivery_id = ? AND body_sha256 = ?",
                (delivery_id, body_sha256),
            )

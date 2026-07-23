"""Project-local SQLite store skeleton and operational health."""

from __future__ import annotations

import logging
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from .crypto import KEY_VERSION, ensure_project_encryption_key
from .database import EncryptedDatabase


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 3
MESSAGE_RETENTION_DAYS = 30
DELIVERY_RETENTION_DAYS = 7
RETENTION_INTERVAL_SECONDS = 3600
MAX_BATCH_EVENTS = 500


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL UNIQUE,
    provider_event_id TEXT,
    payload_hash TEXT NOT NULL,
    received_at TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS delivery_cursor (
    stream TEXT PRIMARY KEY,
    last_sequence INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS delivery_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    expected_sequence INTEGER NOT NULL,
    received_sequence INTEGER NOT NULL,
    detected_at TEXT NOT NULL,
    repaired_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    parent_message_id TEXT,
    sender_id TEXT,
    message_type TEXT,
    text TEXT,
    provider_payload_json TEXT,
    provider_version TEXT,
    occurred_at TEXT NOT NULL,
    edited_at TEXT,
    deleted_at TEXT,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, provider, workspace_id, conversation_id, provider_message_id)
);
CREATE TABLE IF NOT EXISTS reactions (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    reaction_name TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    deleted_at TEXT,
    PRIMARY KEY(project_id, provider, workspace_id, conversation_id,
                provider_message_id, reaction_name, actor_id)
);
CREATE TABLE IF NOT EXISTS conversations (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    conversation_type TEXT,
    title TEXT,
    is_private INTEGER NOT NULL DEFAULT 0,
    is_archived INTEGER NOT NULL DEFAULT 0,
    collection_state TEXT NOT NULL DEFAULT 'DISCOVERED',
    metadata_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, provider, workspace_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS identities (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    external_user_id TEXT NOT NULL,
    display_name TEXT,
    is_bot INTEGER NOT NULL DEFAULT 0,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, provider, workspace_id, external_user_id)
);
CREATE TABLE IF NOT EXISTS coverage (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    contiguous_since TEXT,
    last_sequence INTEGER,
    last_event_at TEXT,
    last_reconciled_at TEXT,
    state TEXT NOT NULL DEFAULT 'DISCOVERED',
    gap_reason TEXT,
    PRIMARY KEY(project_id, provider, workspace_id, conversation_id)
);
CREATE INDEX IF NOT EXISTS ix_messages_conversation_time
    ON messages(project_id, provider, workspace_id, conversation_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_messages_thread_time
    ON messages(project_id, provider, workspace_id, parent_message_id, occurred_at);
CREATE INDEX IF NOT EXISTS ix_messages_project_time
    ON messages(project_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_deliveries_applied_at ON deliveries(applied_at);
"""
_MIGRATION_V2 = """
CREATE TABLE IF NOT EXISTS conversation_memberships (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    external_user_id TEXT NOT NULL,
    is_member INTEGER NOT NULL DEFAULT 1,
    provider_version TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, provider, workspace_id, conversation_id, external_user_id)
);
CREATE INDEX IF NOT EXISTS ix_conversation_memberships_user
    ON conversation_memberships(project_id, provider, workspace_id, external_user_id,
                                is_member);
"""
_MIGRATION_V3 = """
CREATE TABLE IF NOT EXISTS reconciliation_cycles (
    cycle_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    floor_at TEXT NOT NULL,
    ceiling_at TEXT NOT NULL,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reconciliation_seen (
    cycle_id TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    parent_message_id TEXT,
    PRIMARY KEY(cycle_id, provider_message_id),
    FOREIGN KEY(cycle_id) REFERENCES reconciliation_cycles(cycle_id) ON DELETE CASCADE
);
"""
_MIGRATIONS = {1: _SCHEMA, 2: _MIGRATION_V2, 3: _MIGRATION_V3}


class MessageStore:
    def __init__(
        self,
        project_id: str,
        path: Path | None = None,
        *,
        key_version: int | None = None,
    ):
        from gateway.event_ingress import read_project_marker

        self.project_id = str(project_id)
        marker = read_project_marker() or {}
        self.key_version = int(key_version or marker.get("active_key_version") or KEY_VERSION)
        self.path = path or (get_hermes_home() / "state" / "message_store.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer_lock = threading.RLock()
        self._last_retention_at = 0.0
        self.journal_mode = "unknown"
        self.database = EncryptedDatabase(self.path)
        self.database.prepare()
        ensure_project_encryption_key(self.project_id, self.key_version)
        self._migrate()

    def _connect(self):
        return self.database.connect()

    def _migrate(self) -> None:
        with self._writer_lock, self._connect() as conn:
            try:
                mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                self.journal_mode = str(mode).lower()
            except Exception:
                self.journal_mode = "delete"
                conn.execute("PRAGMA journal_mode=DELETE")
            current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"message store schema {current_version} is newer than supported "
                    f"schema {SCHEMA_VERSION}"
                )
            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                script = _MIGRATIONS.get(version)
                if script is None:
                    raise RuntimeError(f"missing message store migration {version}")
                try:
                    conn.executescript(
                        "BEGIN IMMEDIATE;\n"
                        f"{script}\n"
                        f"PRAGMA user_version={version};\n"
                        "COMMIT;"
                    )
                except Exception:
                    conn.rollback()
                    raise
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('database_key_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(self.database.active_key_version),),
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def record_envelope(self, envelope: dict[str, Any], body_sha256: str) -> dict:
        """Commit delivery metadata and cursor atomically; duplicates are harmless."""
        if str(envelope.get("project_id") or "") != self.project_id:
            raise ValueError("project mismatch")
        delivery_id = str(envelope.get("delivery_id") or "").strip()
        sequence = int(envelope.get("sequence"))
        if not delivery_id or sequence < 1:
            raise ValueError("delivery_id and positive sequence are required")
        payload_hash = str(envelope.get("payload_hash") or body_sha256)
        now = datetime.now(timezone.utc).isoformat()
        with self._writer_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT sequence, payload_hash FROM deliveries WHERE delivery_id = ?",
                (delivery_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != payload_hash or existing["sequence"] != sequence:
                    raise ValueError("conflicting duplicate delivery")
                conn.commit()
                return {"status": "duplicate", "sequence": sequence}

            cursor = conn.execute(
                "SELECT last_sequence FROM delivery_cursor WHERE stream = 'project'"
            ).fetchone()
            last_sequence = int(cursor[0]) if cursor else 0
            if sequence > last_sequence + 1:
                unresolved = conn.execute(
                    "SELECT 1 FROM delivery_gaps WHERE expected_sequence = ? "
                    "AND received_sequence = ? AND repaired_at IS NULL",
                    (last_sequence + 1, sequence),
                ).fetchone()
                if unresolved is None:
                    conn.execute(
                        "INSERT INTO delivery_gaps(expected_sequence, received_sequence, detected_at) "
                        "VALUES (?, ?, ?)",
                        (last_sequence + 1, sequence, now),
                    )
                conn.commit()
                return {
                    "status": "gap_detected",
                    "sequence": sequence,
                    "expected_sequence": last_sequence + 1,
                }
            elif sequence <= last_sequence:
                by_sequence = conn.execute(
                    "SELECT delivery_id, payload_hash FROM deliveries WHERE sequence = ?",
                    (sequence,),
                ).fetchone()
                if by_sequence and by_sequence["payload_hash"] == payload_hash:
                    conn.commit()
                    return {"status": "duplicate", "sequence": sequence}
                raise ValueError("conflicting stale sequence")
            else:
                status = "accepted"

            conn.execute(
                "INSERT INTO deliveries(delivery_id, sequence, provider_event_id, "
                "payload_hash, received_at, applied_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    delivery_id,
                    sequence,
                    str(envelope.get("event_id") or "") or None,
                    payload_hash,
                    now,
                    now,
                ),
            )
            if status == "accepted":
                self._apply_normalized_event(conn, envelope, now)
                conn.execute(
                    "INSERT INTO delivery_cursor(stream, last_sequence, updated_at) "
                    "VALUES ('project', ?, ?) ON CONFLICT(stream) DO UPDATE SET "
                    "last_sequence=excluded.last_sequence, updated_at=excluded.updated_at",
                    (sequence, now),
                )
                conn.execute(
                    "UPDATE delivery_gaps SET repaired_at = ? "
                    "WHERE repaired_at IS NULL AND received_sequence <= ?",
                    (now, sequence),
                )
            conn.commit()
        self.maybe_run_retention()
        return {"status": status, "sequence": sequence, "expected_sequence": last_sequence + 1}

    def _apply_normalized_event(
        self,
        conn: Any,
        event: dict[str, Any],
        applied_at: str,
    ) -> None:
        event_type = str(event.get("event_type") or "")
        if not event_type:
            return  # P2 fixture/health deliveries carry only cursor metadata.
        if event_type == "events.batch":
            events = event.get("events")
            if (
                not isinstance(events, list)
                or not events
                or len(events) > MAX_BATCH_EVENTS
            ):
                raise ValueError("normalized event batch size is invalid")
            for child in events:
                if not isinstance(child, dict):
                    raise ValueError("normalized event batch contains a non-object")
                child_type = str(child.get("event_type") or "")
                if not child_type or child_type == "events.batch":
                    raise ValueError("normalized event batch child type is invalid")
                child_event = {
                    **child,
                    "project_id": self.project_id,
                    "provider": str(
                        child.get("provider") or event.get("provider") or ""
                    ),
                    "workspace_id": str(
                        child.get("workspace_id") or event.get("workspace_id") or ""
                    ),
                    "sequence": int(event["sequence"]),
                }
                self._apply_normalized_event(conn, child_event, applied_at)
            return
        provider = str(event.get("provider") or "")
        workspace_id = str(event.get("workspace_id") or "")
        if not provider or not workspace_id:
            raise ValueError("normalized event provider/workspace is required")
        if event_type == "reconciliation.started":
            self._apply_reconciliation_started(conn, event, applied_at)
        elif event_type == "reconciliation.completed":
            self._apply_reconciliation_completed(conn, event, applied_at)
        elif event_type == "coverage.completed":
            self._apply_coverage_completed(conn, event, applied_at)
        elif event_type.startswith("message."):
            self._apply_message(conn, event, applied_at)
        elif event_type.startswith("reaction."):
            self._apply_reaction(conn, event, applied_at)
        elif event_type == "conversation.upsert":
            self._apply_conversation(conn, event, applied_at)
        elif event_type == "identity.upsert":
            self._apply_identity(conn, event, applied_at)
        elif event_type == "membership.changed":
            self._apply_membership(conn, event, applied_at)
        elif event_type == "workspace.purge":
            self._apply_workspace_purge(conn, provider, workspace_id)
        else:
            raise ValueError(f"unsupported normalized event type: {event_type}")

        conversation_id = str(event.get("conversation_id") or "")
        if conversation_id:
            conn.execute(
                "INSERT INTO coverage(project_id, provider, workspace_id, conversation_id, "
                "last_sequence, last_event_at, state) VALUES (?, ?, ?, ?, ?, ?, 'COLLECTING') "
                "ON CONFLICT(project_id, provider, workspace_id, conversation_id) "
                "DO UPDATE SET last_sequence=excluded.last_sequence, "
                "last_event_at=excluded.last_event_at, state='COLLECTING', gap_reason=NULL",
                (
                    self.project_id,
                    str(event["provider"]),
                    str(event["workspace_id"]),
                    conversation_id,
                    int(event["sequence"]),
                    str(event.get("occurred_at") or applied_at),
                ),
            )

    def _apply_workspace_purge(
        self,
        conn: Any,
        provider: str,
        workspace_id: str,
    ) -> None:
        """Delete exactly one provider workspace while preserving project peers."""
        scope = (self.project_id, provider, workspace_id)
        conn.execute(
            "DELETE FROM reconciliation_seen WHERE cycle_id IN (SELECT cycle_id FROM "
            "reconciliation_cycles WHERE project_id=? AND provider=? AND workspace_id=?)",
            scope,
        )
        conn.execute(
            "DELETE FROM reconciliation_cycles WHERE project_id=? AND provider=? "
            "AND workspace_id=?",
            scope,
        )
        for table in (
            "reactions",
            "messages",
            "conversation_memberships",
            "coverage",
            "conversations",
            "identities",
        ):
            conn.execute(
                f"DELETE FROM {table} WHERE project_id=? AND provider=? AND workspace_id=?",
                scope,
            )

    def _apply_message(
        self,
        conn: Any,
        event: dict[str, Any],
        applied_at: str,
    ) -> None:
        conversation_id = str(event.get("conversation_id") or "")
        message_id = str(event.get("message_id") or "")
        event_occurred_at = str(event.get("occurred_at") or "")
        message_occurred_at = str(
            event.get("message_occurred_at") or event_occurred_at
        )
        provider_version = str(event.get("provider_version") or event_occurred_at)
        if (
            not conversation_id
            or not message_id
            or not event_occurred_at
            or not message_occurred_at
            or not provider_version
        ):
            raise ValueError("normalized message identifiers and timestamps are required")
        deleted_at = (
            event_occurred_at if event["event_type"] == "message.deleted" else None
        )
        text_value = None if deleted_at else event.get("text")
        provider_payload = event.get("provider_payload")
        conn.execute(
            "INSERT INTO messages(project_id, provider, workspace_id, conversation_id, "
            "provider_message_id, parent_message_id, sender_id, message_type, text, "
            "provider_payload_json, provider_version, occurred_at, edited_at, deleted_at, "
            "inserted_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, provider, workspace_id, conversation_id, "
            "provider_message_id) DO UPDATE SET "
            "parent_message_id=excluded.parent_message_id, sender_id=excluded.sender_id, "
            "message_type=excluded.message_type, text=excluded.text, "
            "provider_payload_json=excluded.provider_payload_json, "
            "provider_version=excluded.provider_version, occurred_at=excluded.occurred_at, "
            "edited_at=excluded.edited_at, deleted_at=excluded.deleted_at, "
            "updated_at=excluded.updated_at WHERE messages.provider_version IS NULL "
            "OR excluded.provider_version >= messages.provider_version",
            (
                self.project_id,
                str(event["provider"]),
                str(event["workspace_id"]),
                conversation_id,
                message_id,
                str(event.get("parent_message_id") or "") or None,
                str(event.get("sender_id") or "") or None,
                str(event.get("message_type") or "message"),
                text_value,
                json.dumps(provider_payload, separators=(",", ":"), sort_keys=True)
                if provider_payload is not None
                else None,
                provider_version,
                message_occurred_at,
                str(event.get("edited_at") or "") or None,
                deleted_at,
                applied_at,
                applied_at,
            ),
        )
        cycle_id = str(event.get("reconciliation_cycle_id") or "")
        if cycle_id:
            cycle = conn.execute(
                "SELECT 1 FROM reconciliation_cycles WHERE cycle_id = ?",
                (cycle_id,),
            ).fetchone()
            if cycle is None:
                raise ValueError("reconciliation cycle is not started")
            conn.execute(
                "INSERT INTO reconciliation_seen(cycle_id, provider_message_id, "
                "parent_message_id) VALUES (?, ?, ?) ON CONFLICT(cycle_id, "
                "provider_message_id) DO UPDATE SET "
                "parent_message_id=excluded.parent_message_id",
                (
                    cycle_id,
                    message_id,
                    str(event.get("parent_message_id") or "") or None,
                ),
            )
            self._sync_snapshot_reactions(conn, event)

    def _sync_snapshot_reactions(
        self,
        conn: Any,
        event: dict[str, Any],
    ) -> None:
        snapshot_at = str(event.get("reconciled_at") or event.get("occurred_at") or "")
        provider_payload = event.get("provider_payload")
        if not snapshot_at or not isinstance(provider_payload, dict):
            raise ValueError("reconciled message snapshot metadata is required")
        present: set[tuple[str, str]] = set()
        for reaction in provider_payload.get("reactions") or []:
            if not isinstance(reaction, dict):
                continue
            name = str(reaction.get("name") or "")
            for actor in reaction.get("users") or []:
                actor = str(actor or "")
                if not name or not actor:
                    continue
                present.add((name, actor))
                conn.execute(
                    "INSERT INTO reactions(project_id, provider, workspace_id, "
                    "conversation_id, provider_message_id, reaction_name, actor_id, "
                    "occurred_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL) "
                    "ON CONFLICT(project_id, provider, workspace_id, conversation_id, "
                    "provider_message_id, reaction_name, actor_id) DO UPDATE SET "
                    "occurred_at=excluded.occurred_at, deleted_at=NULL "
                    "WHERE reactions.occurred_at <= excluded.occurred_at",
                    (
                        self.project_id,
                        str(event["provider"]),
                        str(event["workspace_id"]),
                        str(event["conversation_id"]),
                        str(event["message_id"]),
                        name,
                        actor,
                        snapshot_at,
                    ),
                )
        existing = conn.execute(
            "SELECT reaction_name, actor_id, occurred_at FROM reactions WHERE "
            "project_id = ? AND provider = ? AND workspace_id = ? AND "
            "conversation_id = ? AND provider_message_id = ? AND deleted_at IS NULL",
            (
                self.project_id,
                str(event["provider"]),
                str(event["workspace_id"]),
                str(event["conversation_id"]),
                str(event["message_id"]),
            ),
        ).fetchall()
        for row in existing:
            if (row["reaction_name"], row["actor_id"]) not in present and row[
                "occurred_at"
            ] <= snapshot_at:
                conn.execute(
                    "UPDATE reactions SET deleted_at = ?, occurred_at = ? WHERE "
                    "project_id = ? AND provider = ? AND workspace_id = ? AND "
                    "conversation_id = ? AND provider_message_id = ? AND "
                    "reaction_name = ? AND actor_id = ?",
                    (
                        snapshot_at,
                        snapshot_at,
                        self.project_id,
                        str(event["provider"]),
                        str(event["workspace_id"]),
                        str(event["conversation_id"]),
                        str(event["message_id"]),
                        row["reaction_name"],
                        row["actor_id"],
                    ),
                )

    def _apply_reconciliation_started(
        self,
        conn: Any,
        event: dict[str, Any],
        applied_at: str,
    ) -> None:
        values = (
            str(event.get("reconciliation_cycle_id") or ""),
            self.project_id,
            str(event["provider"]),
            str(event["workspace_id"]),
            str(event.get("conversation_id") or ""),
            str(event.get("floor_at") or ""),
            str(event.get("ceiling_at") or ""),
            applied_at,
        )
        if not all(values):
            raise ValueError("reconciliation start fields are required")
        conn.execute(
            "INSERT INTO reconciliation_cycles(cycle_id, project_id, provider, "
            "workspace_id, conversation_id, floor_at, ceiling_at, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(cycle_id) DO NOTHING",
            values,
        )

    def _apply_reconciliation_completed(
        self,
        conn: Any,
        event: dict[str, Any],
        applied_at: str,
    ) -> None:
        cycle_id = str(event.get("reconciliation_cycle_id") or "")
        cycle = conn.execute(
            "SELECT * FROM reconciliation_cycles WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        if cycle is None:
            raise ValueError("reconciliation cycle is not started")
        version = str(event.get("provider_version") or "")
        if not version:
            raise ValueError("reconciliation completion version is required")
        scope_params = (
            applied_at,
            version,
            self.project_id,
            cycle["provider"],
            cycle["workspace_id"],
            cycle["conversation_id"],
            cycle["floor_at"],
            cycle["ceiling_at"],
            cycle_id,
        )
        conn.execute(
            "UPDATE messages SET text=NULL, deleted_at=?, provider_version=?, "
            "updated_at=? WHERE project_id=? AND provider=? AND workspace_id=? AND "
            "conversation_id=? AND parent_message_id IS NULL AND occurred_at>=? AND "
            "occurred_at<=? AND (provider_version IS NULL OR provider_version<=?) AND "
            "provider_message_id NOT IN (SELECT provider_message_id FROM "
            "reconciliation_seen WHERE cycle_id=?)",
            (
                applied_at,
                version,
                applied_at,
                *scope_params[2:8],
                version,
                scope_params[8],
            ),
        )
        for thread_ts in event.get("completed_thread_ts") or []:
            conn.execute(
                "UPDATE messages SET text=NULL, deleted_at=?, provider_version=?, "
                "updated_at=? WHERE project_id=? AND provider=? AND workspace_id=? AND "
                "conversation_id=? AND parent_message_id=? AND occurred_at>=? AND "
                "occurred_at<=? AND (provider_version IS NULL OR provider_version<=?) "
                "AND provider_message_id NOT IN (SELECT provider_message_id FROM "
                "reconciliation_seen WHERE cycle_id=?)",
                (
                    applied_at,
                    version,
                    applied_at,
                    self.project_id,
                    cycle["provider"],
                    cycle["workspace_id"],
                    cycle["conversation_id"],
                    str(thread_ts),
                    cycle["floor_at"],
                    cycle["ceiling_at"],
                    version,
                    cycle_id,
                ),
            )
        conn.execute("DELETE FROM reconciliation_cycles WHERE cycle_id = ?", (cycle_id,))

    def _apply_coverage_completed(
        self,
        conn: Any,
        event: dict[str, Any],
        applied_at: str,
    ) -> None:
        conversation_id = str(event.get("conversation_id") or "")
        contiguous_since = str(event.get("contiguous_since") or "")
        if not conversation_id or not contiguous_since:
            raise ValueError("coverage completion fields are required")
        conn.execute(
            "INSERT INTO coverage(project_id, provider, workspace_id, conversation_id, "
            "contiguous_since, last_sequence, last_event_at, state, gap_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'COLLECTING', NULL) ON CONFLICT(project_id, "
            "provider, workspace_id, conversation_id) DO UPDATE SET "
            "contiguous_since=CASE WHEN coverage.contiguous_since IS NULL OR "
            "excluded.contiguous_since < coverage.contiguous_since THEN "
            "excluded.contiguous_since ELSE coverage.contiguous_since END, "
            "last_sequence=excluded.last_sequence, last_event_at=excluded.last_event_at, "
            "state='COLLECTING', gap_reason=NULL",
            (
                self.project_id,
                str(event["provider"]),
                str(event["workspace_id"]),
                conversation_id,
                contiguous_since,
                int(event["sequence"]),
                str(event.get("occurred_at") or applied_at),
            ),
        )

    def query(self, request: dict[str, Any]) -> dict[str, Any]:
        """Bounded project-local read after IE has resolved current ACL."""
        operation = str(request.get("operation") or "")
        if operation not in {
            "recent_activity",
            "fetch_history",
            "fetch_snapshot",
            "ingest_window",
        }:
            raise ValueError("unsupported message store query operation")
        start = str(request.get("start") or "")
        end = str(request.get("end") or "")
        if not start or not end or start > end:
            raise ValueError("bounded start/end are required")
        limit = max(1, min(int(request.get("limit") or 100), 1000))
        per_conversation = max(
            1, min(int(request.get("per_conversation") or limit), 100)
        )
        providers = {str(value) for value in request.get("providers") or [] if value}
        workspaces = {
            str(value) for value in request.get("workspace_ids") or [] if value
        }
        conversations = {
            str(value) for value in request.get("conversation_ids") or [] if value
        }
        raw_sources = request.get("allowed_source_ids")
        allowed: set[tuple[str, str, str]] | None = None
        if raw_sources is not None:
            allowed = set()
            for source_id in raw_sources:
                parts = str(source_id).split(":", 2)
                if len(parts) != 3 or not all(parts):
                    raise ValueError("invalid allowed source id")
                allowed.add((parts[0], parts[1], parts[2]))
            if not allowed:
                return {
                    "messages": [],
                    "coverage_complete": True,
                    "reason": "acl_empty",
                    "covered_since": start,
                }
            allowed = {
                item
                for item in allowed
                if (not providers or item[0] in providers)
                and (not workspaces or item[1] in workspaces)
                and (not conversations or item[2] in conversations)
            }
            if not allowed:
                return {
                    "messages": [],
                    "coverage_complete": True,
                    "reason": "acl_empty",
                    "covered_since": start,
                }
            providers = {item[0] for item in allowed}
            workspaces = {item[1] for item in allowed}
            conversations = {item[2] for item in allowed}
        if operation == "fetch_history" and (
            len(conversations) != 1 or (allowed is not None and len(allowed) != 1)
        ):
            raise ValueError("fetch_history requires one authorized conversation")
        snapshot_keys: set[tuple[str, str]] = set()
        if operation == "fetch_snapshot":
            raw_keys = request.get("provider_message_keys")
            if not isinstance(raw_keys, list) or not raw_keys:
                raise ValueError("fetch_snapshot requires provider message keys")
            if len(raw_keys) > 1000:
                raise ValueError("fetch_snapshot provider message key limit is 1000")
            for item in raw_keys:
                if not isinstance(item, dict):
                    raise ValueError("invalid provider message key")
                key = (
                    str(item.get("conversation_id") or ""),
                    str(item.get("provider_message_id") or ""),
                )
                if not all(key):
                    raise ValueError("invalid provider message key")
                snapshot_keys.add(key)
            if len(snapshot_keys) != len(raw_keys):
                raise ValueError("duplicate provider message key")
            if conversations and any(
                conversation_id not in conversations
                for conversation_id, _message_id in snapshot_keys
            ):
                raise ValueError("provider message key is outside conversation scope")

        with self._writer_lock, self._connect() as conn:
            unresolved_gap = conn.execute(
                "SELECT 1 FROM delivery_gaps WHERE repaired_at IS NULL LIMIT 1"
            ).fetchone()
            if unresolved_gap is not None:
                return {
                    "messages": [],
                    "coverage_complete": False,
                    "reason": "delivery_gap",
                }
            coverage_rows = []
            if operation != "fetch_snapshot":
                coverage_sql = (
                    "SELECT provider, workspace_id, conversation_id, contiguous_since, "
                    "state FROM coverage WHERE project_id = ?"
                )
                coverage_params: list[Any] = [self.project_id]
                coverage_sql, coverage_params = self._query_filters(
                    coverage_sql,
                    coverage_params,
                    providers=providers,
                    workspaces=workspaces,
                    conversations=conversations,
                    allowed=allowed,
                )
                coverage_rows = conn.execute(coverage_sql, coverage_params).fetchall()
                covered_sources = {
                    (
                        str(row["provider"]),
                        str(row["workspace_id"]),
                        str(row["conversation_id"]),
                    )
                    for row in coverage_rows
                }
                if allowed is not None and covered_sources != allowed:
                    return {
                        "messages": [],
                        "coverage_complete": False,
                        "reason": "coverage_missing",
                    }
                if allowed is None and conversations:
                    covered_ids = {item[2] for item in covered_sources}
                    if covered_ids != conversations:
                        return {
                            "messages": [],
                            "coverage_complete": False,
                            "reason": "coverage_missing",
                        }
                if not coverage_rows:
                    return {
                        "messages": [],
                        "coverage_complete": False,
                        "reason": "coverage_missing",
                    }
                if any(
                    row["state"] != "COLLECTING"
                    or not row["contiguous_since"]
                    or str(row["contiguous_since"]) > start
                    for row in coverage_rows
                ):
                    return {
                        "messages": [],
                        "coverage_complete": False,
                        "reason": "coverage_incomplete",
                    }

            time_column = (
                "m.updated_at" if operation == "ingest_window" else "m.occurred_at"
            )
            sql = (
                "SELECT m.*, c.title AS conversation_title, "
                "i.display_name AS sender_display_name FROM messages m "
                "LEFT JOIN conversations c ON c.project_id=m.project_id AND "
                "c.provider=m.provider AND c.workspace_id=m.workspace_id AND "
                "c.conversation_id=m.conversation_id LEFT JOIN identities i ON "
                "i.project_id=m.project_id AND i.provider=m.provider AND "
                "i.workspace_id=m.workspace_id AND i.external_user_id=m.sender_id "
                f"WHERE m.project_id=? AND {time_column}>=? AND {time_column}<=? "
                "AND m.deleted_at IS NULL"
            )
            params: list[Any] = [self.project_id, start, end]
            sql, params = self._query_filters(
                sql,
                params,
                providers=providers,
                workspaces=workspaces,
                conversations=conversations,
                prefix="m.",
                allowed=allowed,
            )
            parent = str(request.get("parent_message_id") or "")
            if operation == "fetch_snapshot":
                clauses = []
                for conversation_id, message_id in sorted(snapshot_keys):
                    clauses.append(
                        "(m.conversation_id=? AND m.provider_message_id=?)"
                    )
                    params.extend([conversation_id, message_id])
                sql += " AND (" + " OR ".join(clauses) + ")"
            elif parent:
                sql += " AND (m.provider_message_id=? OR m.parent_message_id=?)"
                params.extend([parent, parent])
            elif operation == "fetch_history":
                sql += " AND m.parent_message_id IS NULL"
            cursor = request.get("cursor")
            if cursor is not None:
                if operation != "ingest_window" or not isinstance(cursor, dict):
                    raise ValueError("cursor is only supported for ingest_window")
                cursor_values = [
                    str(cursor.get(name) or "")
                    for name in (
                        "changed_at",
                        "provider",
                        "workspace_id",
                        "conversation_id",
                        "provider_message_id",
                    )
                ]
                if not all(cursor_values):
                    raise ValueError("invalid ingest cursor")
                sql += (
                    " AND (m.updated_at, m.provider, m.workspace_id, "
                    "m.conversation_id, m.provider_message_id) < (?, ?, ?, ?, ?)"
                )
                params.extend(cursor_values)
            sql += (
                f" ORDER BY {time_column} DESC, m.provider DESC, "
                "m.workspace_id DESC, m.conversation_id DESC, "
                "m.provider_message_id DESC LIMIT ?"
            )
            params.append(
                limit * 10
                if operation == "recent_activity"
                else max(limit, len(snapshot_keys))
            )
            rows = conn.execute(sql, params).fetchall()
            selected_rows = []
            per_counts: dict[str, int] = {}
            for row in rows:
                conversation_id = str(row["conversation_id"])
                count = per_counts.get(conversation_id, 0)
                if operation == "recent_activity" and count >= per_conversation:
                    continue
                selected_rows.append(row)
                per_counts[conversation_id] = count + 1
                if len(selected_rows) >= limit:
                    break

            selected_keys = {
                (
                    str(row["provider"]),
                    str(row["workspace_id"]),
                    str(row["conversation_id"]),
                    str(row["provider_message_id"]),
                )
                for row in selected_rows
            }
            reactions_by_message: dict[
                tuple[str, str, str, str], dict[str, list[str]]
            ] = {}
            message_ids = sorted({key[3] for key in selected_keys})
            if message_ids:
                reaction_rows = conn.execute(
                    "SELECT provider, workspace_id, conversation_id, "
                    "provider_message_id, reaction_name, actor_id FROM reactions "
                    "WHERE project_id=? AND deleted_at IS NULL AND "
                    f"provider_message_id IN ({','.join('?' for _ in message_ids)})",
                    [self.project_id, *message_ids],
                ).fetchall()
                for reaction in reaction_rows:
                    key = (
                        str(reaction["provider"]),
                        str(reaction["workspace_id"]),
                        str(reaction["conversation_id"]),
                        str(reaction["provider_message_id"]),
                    )
                    if key not in selected_keys:
                        continue
                    reactions_by_message.setdefault(key, {}).setdefault(
                        str(reaction["reaction_name"]), []
                    ).append(str(reaction["actor_id"]))

            messages: list[dict[str, Any]] = []
            for row in selected_rows:
                conversation_id = str(row["conversation_id"])
                try:
                    provider_payload = json.loads(row["provider_payload_json"] or "{}")
                except json.JSONDecodeError:
                    provider_payload = {}
                reaction_map: dict[str, dict[str, Any]] = {}
                for reaction in provider_payload.get("reactions") or []:
                    if not isinstance(reaction, dict):
                        continue
                    name = str(reaction.get("name") or "")
                    if not name:
                        continue
                    users = [
                        str(actor)
                        for actor in reaction.get("users") or []
                        if str(actor)
                    ]
                    reaction_map[name] = {
                        "count": max(int(reaction.get("count") or 0), len(users)),
                        "users": users,
                    }
                stored_reactions = reactions_by_message.get(
                    (
                        str(row["provider"]),
                        str(row["workspace_id"]),
                        conversation_id,
                        str(row["provider_message_id"]),
                    ),
                    {},
                )
                for name, actors in stored_reactions.items():
                    current = reaction_map.setdefault(
                        name,
                        {"count": 0, "users": []},
                    )
                    current["users"] = sorted(
                        set(current["users"]) | set(actors)
                    )
                    current["count"] = max(
                        int(current["count"]),
                        len(current["users"]),
                    )
                messages.append(
                    {
                        "provider": row["provider"],
                        "workspace_id": row["workspace_id"],
                        "conversation_id": conversation_id,
                        "conversation_title": row["conversation_title"],
                        "provider_message_id": row["provider_message_id"],
                        "parent_message_id": row["parent_message_id"],
                        "sender_id": row["sender_id"],
                        "sender_display_name": row["sender_display_name"],
                        "text": row["text"],
                        "occurred_at": row["occurred_at"],
                        "changed_at": row["updated_at"],
                        "edited_at": row["edited_at"],
                        "provider_payload": provider_payload,
                        "reactions": [
                            {
                                "name": name,
                                "count": value["count"],
                                "users": value["users"],
                            }
                            for name, value in sorted(reaction_map.items())
                        ],
                    }
                )
            if operation == "fetch_snapshot":
                returned_keys = {
                    (
                        str(message["conversation_id"]),
                        str(message["provider_message_id"]),
                    )
                    for message in messages
                }
                missing = snapshot_keys - returned_keys
                cursor = conn.execute(
                    "SELECT last_sequence FROM delivery_cursor WHERE stream='project'"
                ).fetchone()
                return {
                    "messages": messages,
                    "coverage_complete": not missing,
                    "reason": "snapshot_missing" if missing else "snapshot_exact",
                    "last_sequence": int(cursor[0]) if cursor else 0,
                }
            floor = max(str(row["contiguous_since"]) for row in coverage_rows)
            result = {
                "messages": messages,
                "coverage_complete": True,
                "covered_since": floor,
                "last_sequence": max(
                    int(row["last_sequence"] or 0)
                    for row in conn.execute(
                        "SELECT last_sequence FROM coverage WHERE project_id=?",
                        (self.project_id,),
                    ).fetchall()
                ),
            }
            if operation == "ingest_window" and len(rows) == limit:
                last = rows[-1]
                result["next_cursor"] = {
                    "changed_at": str(last["updated_at"]),
                    "provider": str(last["provider"]),
                    "workspace_id": str(last["workspace_id"]),
                    "conversation_id": str(last["conversation_id"]),
                    "provider_message_id": str(last["provider_message_id"]),
                }
            return result

    @staticmethod
    def _query_filters(
        sql: str,
        params: list[Any],
        *,
        providers: set[str],
        workspaces: set[str],
        conversations: set[str],
        prefix: str = "",
        allowed: set[tuple[str, str, str]] | None = None,
    ) -> tuple[str, list[Any]]:
        for column, values in (
            ("provider", providers),
            ("workspace_id", workspaces),
            ("conversation_id", conversations),
        ):
            if values:
                ordered = sorted(values)
                sql += f" AND {prefix}{column} IN ({','.join('?' for _ in ordered)})"
                params.extend(ordered)
        if allowed is not None:
            clauses: list[str] = []
            for provider, workspace, conversation in sorted(allowed):
                clauses.append(
                    f"({prefix}provider=? AND {prefix}workspace_id=? "
                    f"AND {prefix}conversation_id=?)"
                )
                params.extend([provider, workspace, conversation])
            sql += " AND (" + " OR ".join(clauses) + ")"
        return sql, params

    def _apply_reaction(
        self,
        conn: Any,
        event: dict[str, Any],
        applied_at: str,
    ) -> None:
        values = (
            self.project_id,
            str(event["provider"]),
            str(event["workspace_id"]),
            str(event.get("conversation_id") or ""),
            str(event.get("message_id") or ""),
            str(event.get("reaction_name") or ""),
            str(event.get("actor_id") or ""),
            str(event.get("occurred_at") or ""),
        )
        if not all(values[3:]):
            raise ValueError("normalized reaction fields are required")
        if event["event_type"] == "reaction.removed":
            conn.execute(
                "INSERT INTO reactions(project_id, provider, workspace_id, conversation_id, "
                "provider_message_id, reaction_name, actor_id, occurred_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(project_id, provider, "
                "workspace_id, conversation_id, provider_message_id, reaction_name, actor_id) "
                "DO UPDATE SET occurred_at=excluded.occurred_at, "
                "deleted_at=excluded.deleted_at WHERE excluded.occurred_at >= "
                "reactions.occurred_at",
                (*values, values[7]),
            )
        else:
            conn.execute(
                "INSERT INTO reactions(project_id, provider, workspace_id, conversation_id, "
                "provider_message_id, reaction_name, actor_id, occurred_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL) ON CONFLICT(project_id, provider, "
                "workspace_id, conversation_id, provider_message_id, reaction_name, actor_id) "
                "DO UPDATE SET occurred_at=excluded.occurred_at, deleted_at=NULL "
                "WHERE excluded.occurred_at >= reactions.occurred_at",
                values,
            )
        conn.execute(
            "UPDATE messages SET updated_at=? WHERE project_id=? AND provider=? AND "
            "workspace_id=? AND conversation_id=? AND provider_message_id=? AND "
            "updated_at<=?",
            (
                applied_at,
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                applied_at,
            ),
        )

    def _apply_conversation(
        self, conn: Any, event: dict[str, Any], applied_at: str
    ) -> None:
        conversation_id = str(event.get("conversation_id") or "")
        if not conversation_id:
            raise ValueError("normalized conversation id is required")
        existing = conn.execute(
            "SELECT conversation_type, title, is_private, is_archived FROM conversations "
            "WHERE project_id = ? AND provider = ? AND workspace_id = ? "
            "AND conversation_id = ?",
            (
                self.project_id,
                str(event["provider"]),
                str(event["workspace_id"]),
                conversation_id,
            ),
        ).fetchone()

        def merged(name: str, fallback: Any) -> Any:
            if name in event and event[name] is not None:
                return event[name]
            return existing[name] if existing is not None else fallback

        conn.execute(
            "INSERT INTO conversations(project_id, provider, workspace_id, conversation_id, "
            "conversation_type, title, is_private, is_archived, collection_state, "
            "metadata_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, provider, workspace_id, conversation_id) DO UPDATE SET "
            "conversation_type=excluded.conversation_type, title=excluded.title, "
            "is_private=excluded.is_private, is_archived=excluded.is_archived, "
            "metadata_json=excluded.metadata_json, updated_at=excluded.updated_at",
            (
                self.project_id,
                str(event["provider"]),
                str(event["workspace_id"]),
                conversation_id,
                str(merged("conversation_type", "") or "") or None,
                str(merged("title", "") or "") or None,
                int(bool(merged("is_private", False))),
                int(bool(merged("is_archived", False))),
                str(event.get("collection_state") or "DISCOVERED"),
                json.dumps(
                    event.get("provider_payload") or {},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                applied_at,
            ),
        )

    def _apply_identity(
        self, conn: Any, event: dict[str, Any], applied_at: str
    ) -> None:
        external_user_id = str(event.get("external_user_id") or "")
        if not external_user_id:
            raise ValueError("normalized identity id is required")
        conn.execute(
            "INSERT INTO identities(project_id, provider, workspace_id, external_user_id, "
            "display_name, is_bot, is_deleted, metadata_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(project_id, provider, "
            "workspace_id, external_user_id) DO UPDATE SET display_name=excluded.display_name, "
            "is_bot=excluded.is_bot, is_deleted=excluded.is_deleted, "
            "metadata_json=excluded.metadata_json, updated_at=excluded.updated_at",
            (
                self.project_id,
                str(event["provider"]),
                str(event["workspace_id"]),
                external_user_id,
                str(event.get("display_name") or "") or None,
                int(bool(event.get("is_bot"))),
                int(bool(event.get("is_deleted"))),
                json.dumps(
                    event.get("provider_payload") or {},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                applied_at,
            ),
        )

    def _apply_membership(
        self, conn: Any, event: dict[str, Any], applied_at: str
    ) -> None:
        conversation_id = str(event.get("conversation_id") or "")
        external_user_id = str(event.get("external_user_id") or "")
        if not conversation_id or not external_user_id:
            raise ValueError("normalized membership fields are required")
        conn.execute(
            "INSERT INTO conversation_memberships(project_id, provider, workspace_id, "
            "conversation_id, external_user_id, is_member, provider_version, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(project_id, provider, workspace_id, "
            "conversation_id, external_user_id) DO UPDATE SET is_member=excluded.is_member, "
            "provider_version=excluded.provider_version, updated_at=excluded.updated_at "
            "WHERE conversation_memberships.provider_version IS NULL OR "
            "excluded.provider_version >= conversation_memberships.provider_version",
            (
                self.project_id,
                str(event["provider"]),
                str(event["workspace_id"]),
                conversation_id,
                external_user_id,
                int(bool(event.get("is_member"))),
                str(event.get("provider_version") or event.get("occurred_at") or applied_at),
                applied_at,
            ),
        )

    def maybe_run_retention(self) -> None:
        now = time.time()
        if now - self._last_retention_at < RETENTION_INTERVAL_SECONDS:
            return
        self.run_retention()
        self._last_retention_at = now

    def run_retention(self, *, now: datetime | None = None) -> dict[str, int]:
        current = now or datetime.now(timezone.utc)
        message_cutoff = (current - timedelta(days=MESSAGE_RETENTION_DAYS)).isoformat()
        delivery_cutoff = (current - timedelta(days=DELIVERY_RETENTION_DAYS)).isoformat()
        with self._writer_lock, self._connect() as conn:
            deliveries = conn.execute(
                "DELETE FROM deliveries WHERE applied_at < ?", (delivery_cutoff,)
            ).rowcount
            reactions = conn.execute(
                "DELETE FROM reactions WHERE occurred_at < ?", (message_cutoff,)
            ).rowcount
            messages = conn.execute(
                "DELETE FROM messages WHERE occurred_at < ?", (message_cutoff,)
            ).rowcount
        return {"deliveries": deliveries, "reactions": reactions, "messages": messages}

    def health(self) -> dict[str, Any]:
        self.maybe_run_retention()
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT last_sequence FROM delivery_cursor WHERE stream = 'project'"
            ).fetchone()
            gaps = conn.execute(
                "SELECT COUNT(*) FROM delivery_gaps WHERE repaired_at IS NULL"
            ).fetchone()[0]
            coverage = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT state, COUNT(*) FROM coverage GROUP BY state"
                ).fetchall()
            }
            collection = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT collection_state, COUNT(*) FROM conversations GROUP BY collection_state"
                ).fetchall()
            }
            latest = conn.execute(
                "SELECT MAX(applied_at) FROM deliveries"
            ).fetchone()[0]
        lag_seconds = None
        if latest:
            try:
                lag_seconds = max(
                    0.0,
                    (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(str(latest))
                    ).total_seconds(),
                )
            except ValueError:
                pass
        wal = self.path.with_name(self.path.name + "-wal")
        return {
            "name": "ringo_message_store",
            "status": "ready",
            "project_id": self.project_id,
            "schema_version": SCHEMA_VERSION,
            "key_version": self.key_version,
            "storage_encryption": "sqlcipher",
            "database_key_version": self.database.active_key_version,
            "database_key_opened_version": self.database.opened_key_version,
            "encryption_migration": self.database.migration_status,
            "encryption_integrity": self.database.integrity_status,
            "cipher_version": self.database.cipher_version,
            "journal_mode": self.journal_mode,
            "db_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "wal_bytes": wal.stat().st_size if wal.exists() else 0,
            "last_sequence": int(cursor[0]) if cursor else 0,
            "lag_seconds": lag_seconds,
            "unresolved_gaps": int(gaps),
            "coverage_states": coverage,
            "collection_states": collection,
        }

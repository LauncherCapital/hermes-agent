"""Project-local SQLite store skeleton and operational health."""

from __future__ import annotations

import logging
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from .crypto import KEY_VERSION, ensure_project_encryption_key
from .database import EncryptedDatabase
from .file_processing import (
    FileProcessingError,
    cleanup_stale_temp_files,
    embed_text,
    embed_texts,
    inspect_slack_image,
    process_slack_file,
)
from .search_text import (
    normalize_search_text,
    search_feature_set,
    search_features,
    search_index_terms,
    search_segments,
)


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 9
PROTOCOL_VERSION = 2
PROTOCOL_CAPABILITIES = (
    "acl_metadata",
    "allowed_source_ids",
    "detailed_health",
    "event_batch",
    "file_index_parent_context_v1",
    "file_index_v1",
    "ingest_window",
    "message_search_v1",
    "reconciliation_events",
    "stable_cursor",
)
MESSAGE_RETENTION_DAYS = 30
DELIVERY_RETENTION_DAYS = 7
RETENTION_INTERVAL_SECONDS = 3600
MAX_BATCH_EVENTS = 500
FILE_SEARCH_RETRIEVE_LIMIT = 50
FILE_SEARCH_RERANK_LIMIT = 10
FILE_SEARCH_IMAGE_INSPECT_LIMIT = 3
FILE_SEARCH_MIN_SEMANTIC_SCORE = 0.80
FILE_SEARCH_MIN_PARENT_SEMANTIC_SCORE = 0.25
FILE_SEARCH_RRF_K = 60
FILE_SEARCH_MAX_INDEX_MATCHES = 5_000
FILE_SEARCH_DIRECT_LEXICAL_LANE_WEIGHT = 0.25
FILE_SEARCH_CONTEXT_LANE_WEIGHT = 1.0
FILE_SEARCH_CONTEXT_DENSE_LANE_WEIGHT = 2.0
FILE_EMBEDDING_BACKFILL_LIMIT = 20
MAX_FILE_PROCESSING_ATTEMPTS = 3
MESSAGE_SEARCH_ADJACENCY_SECONDS = 10 * 60
MESSAGE_SEARCH_CANDIDATE_LIMIT = 200
MESSAGE_SEARCH_CONTEXT_ROW_LIMIT = 5_000
MESSAGE_SEARCH_MAX_ROWS_PER_ROOT = 100
MESSAGE_SEARCH_MIN_FEATURE_COVERAGE = 0.30
MESSAGE_SEARCH_MAX_QUERY_TERMS = 96


def _search_words(text: str) -> list[str]:
    return search_segments(text)


def _search_features(text: str) -> Counter[str]:
    return search_features(text)


def _row_search_text(row: Any) -> str:
    return " ".join(
        value
        for value in (
            str(row["text"] or ""),
            str(row["conversation_title"] or ""),
            str(row["sender_display_name"] or ""),
        )
        if value
    )


def _row_timestamp(row: Any) -> float:
    try:
        return datetime.fromisoformat(str(row["occurred_at"])).timestamp()
    except ValueError:
        return 0.0


def _search_message(row: Any) -> dict[str, Any]:
    try:
        provider_payload = json.loads(row["provider_payload_json"] or "{}")
    except json.JSONDecodeError:
        provider_payload = {}
    return {
        "provider": row["provider"],
        "workspace_id": row["workspace_id"],
        "conversation_id": row["conversation_id"],
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
        "reactions": [],
    }


def _rank_message_rows(
    rows: list[Any],
    *,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    query_words = set(_search_words(query))
    query_features = _search_features(query)
    if not query_words or not query_features:
        return []

    units: list[dict[str, Any]] = []
    by_conversation: dict[tuple[str, str, str], list[Any]] = {}
    for row in rows:
        by_conversation.setdefault(
            (
                str(row["provider"]),
                str(row["workspace_id"]),
                str(row["conversation_id"]),
            ),
            [],
        ).append(row)
    for conversation_rows in by_conversation.values():
        grouped: dict[str, list[Any]] = {}
        for row in conversation_rows:
            root_id = str(row["parent_message_id"] or row["provider_message_id"])
            grouped.setdefault(root_id, []).append(row)
        conversation_units = [
            {
                "core": sorted(group, key=_row_timestamp),
                "context": sorted(group, key=_row_timestamp),
            }
            for group in grouped.values()
        ]
        conversation_units.sort(key=lambda item: _row_timestamp(item["core"][0]))
        for index, unit in enumerate(conversation_units):
            adjacent: list[Any] = []
            for neighbor_index in (index - 1, index + 1):
                if not 0 <= neighbor_index < len(conversation_units):
                    continue
                neighbor = conversation_units[neighbor_index]
                gap = min(
                    abs(_row_timestamp(left) - _row_timestamp(right))
                    for left in unit["core"]
                    for right in neighbor["core"]
                )
                if gap <= MESSAGE_SEARCH_ADJACENCY_SECONDS:
                    adjacent.extend(neighbor["core"])
            unit["adjacent"] = adjacent
            unit["context"] = sorted(
                [*unit["core"], *adjacent],
                key=_row_timestamp,
            )
            units.append(unit)

    document_features: list[Counter[str]] = []
    for unit in units:
        core_text = " ".join(_row_search_text(row) for row in unit["core"])
        adjacent_text = " ".join(_row_search_text(row) for row in unit["adjacent"])
        features = _search_features(core_text)
        for feature, count in _search_features(adjacent_text).items():
            features[feature] += count * 0.35
        document_features.append(features)

    document_count = len(units)
    if not document_count:
        return []
    document_frequency = {
        feature: sum(1 for features in document_features if features[feature] > 0)
        for feature in query_features
    }
    lengths = [sum(features.values()) for features in document_features]
    average_length = sum(lengths) / document_count or 1.0
    ranked: list[tuple[float, dict[str, Any]]] = []
    normalized_query = normalize_search_text(query)
    total_query_features = sum(query_features.values())
    for index, unit in enumerate(units):
        matched_feature_count = sum(
            min(count, document_features[index][feature])
            for feature, count in query_features.items()
        )
        feature_coverage = (
            matched_feature_count / total_query_features
            if total_query_features
            else 0.0
        )
        if feature_coverage < MESSAGE_SEARCH_MIN_FEATURE_COVERAGE:
            continue
        score = 0.0
        for feature, query_count in query_features.items():
            term_frequency = document_features[index][feature]
            if term_frequency <= 0:
                continue
            frequency = document_frequency[feature]
            inverse_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            denominator = term_frequency + 1.2 * (
                0.25 + 0.75 * lengths[index] / average_length
            )
            score += (
                inverse_frequency * (term_frequency * 2.2 / denominator) * query_count
            )
        core_text = normalize_search_text(
            " ".join(_row_search_text(row) for row in unit["core"])
        )
        context_text = normalize_search_text(
            " ".join(_row_search_text(row) for row in unit["context"])
        )
        if normalized_query in core_text:
            score += 2.0
        elif normalized_query in context_text:
            score += 0.7
        primary = max(
            unit["context"],
            key=lambda row: (
                sum(
                    min(count, _search_features(_row_search_text(row))[feature])
                    for feature, count in query_features.items()
                ),
                _row_timestamp(row),
            ),
        )
        ranked.append((
            score,
            {
                "score": round(score, 6),
                "message": _search_message(primary),
                "context": [_search_message(row) for row in unit["context"]],
            },
        ))

    ranked.sort(
        key=lambda item: (
            item[0],
            _row_timestamp(
                next(
                    row
                    for unit in units
                    for row in unit["context"]
                    if str(row["provider_message_id"])
                    == str(item[1]["message"]["provider_message_id"])
                    and str(row["conversation_id"])
                    == str(item[1]["message"]["conversation_id"])
                )
            ),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_contexts: list[set[tuple[str, str]]] = []
    for _score, hit in ranked:
        context_keys = {
            (
                str(message["conversation_id"]),
                str(message["provider_message_id"]),
            )
            for message in hit["context"]
        }
        if any(context_keys & existing for existing in selected_contexts):
            continue
        selected.append(hit)
        selected_contexts.append(context_keys)
        if len(selected) >= limit:
            break
    return selected


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
_MIGRATION_V4 = """
CREATE TABLE IF NOT EXISTS file_contents (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    content_version INTEGER NOT NULL,
    content_sha256 TEXT,
    file_name TEXT,
    mime_type TEXT,
    byte_size INTEGER,
    uploaded_at TEXT,
    permalink TEXT,
    processing_status TEXT NOT NULL,
    caption_ocr TEXT,
    text_content_embedding_json TEXT,
    image_embedding_json TEXT,
    caption_model TEXT,
    caption_prompt_version TEXT,
    text_embedding_model TEXT,
    text_embedding_dimension INTEGER,
    image_embedding_model TEXT,
    image_embedding_dimension INTEGER,
    last_error_code TEXT,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, provider, workspace_id, file_id),
    CHECK(processing_status IN
          ('indexed', 'metadata_only', 'unsupported', 'unavailable', 'deleted'))
);
CREATE TABLE IF NOT EXISTS file_shares (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    uploader_id TEXT,
    upload_text TEXT,
    thread_context TEXT,
    context_version INTEGER NOT NULL,
    shared_at TEXT,
    tombstone_version INTEGER,
    tombstoned_at TEXT,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, provider, workspace_id, file_id,
                conversation_id, provider_message_id)
);
CREATE TABLE IF NOT EXISTS file_tombstones (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    source_version INTEGER NOT NULL,
    tombstoned_at TEXT NOT NULL,
    PRIMARY KEY(project_id, provider, workspace_id, file_id)
);
CREATE TABLE IF NOT EXISTS file_index_runs (
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    store_generation TEXT NOT NULL,
    window_started_at TEXT NOT NULL,
    window_ended_at TEXT NOT NULL,
    allowed_scope_revision TEXT NOT NULL,
    message_scan_cursor_json TEXT,
    status TEXT NOT NULL,
    processing_signature TEXT NOT NULL,
    scanned_messages INTEGER NOT NULL DEFAULT 0,
    discovered_shares INTEGER NOT NULL DEFAULT 0,
    indexed_contents INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY(provider, workspace_id, store_generation,
                window_started_at, window_ended_at),
    CHECK(status IN ('pending', 'running', 'complete', 'invalidated'))
);
CREATE INDEX IF NOT EXISTS ix_file_contents_workspace_time
    ON file_contents(project_id, provider, workspace_id, uploaded_at DESC);
CREATE INDEX IF NOT EXISTS ix_file_contents_workspace_hash
    ON file_contents(project_id, provider, workspace_id, content_sha256)
    WHERE content_sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_file_shares_source
    ON file_shares(project_id, provider, workspace_id,
                   conversation_id, shared_at DESC);
CREATE INDEX IF NOT EXISTS ix_file_index_runs_status
    ON file_index_runs(status, updated_at);
"""
_MIGRATION_V5 = """
CREATE TABLE IF NOT EXISTS file_graph_ingest_state (
    project_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    content_version INTEGER NOT NULL,
    context_version INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY(project_id, provider, workspace_id, file_id,
                conversation_id, provider_message_id)
);
CREATE INDEX IF NOT EXISTS ix_file_graph_ingest_workspace
    ON file_graph_ingest_state(project_id, provider, workspace_id, ingested_at);
"""
_MIGRATION_V6 = """
ALTER TABLE file_contents
    ADD COLUMN processing_attempts INTEGER NOT NULL DEFAULT 0;
"""
_MIGRATION_V7 = """
ALTER TABLE file_shares
    ADD COLUMN parent_embedding_json TEXT;
ALTER TABLE file_shares
    ADD COLUMN parent_embedding_model TEXT;
ALTER TABLE file_shares
    ADD COLUMN parent_embedding_dimension INTEGER;
ALTER TABLE file_shares
    ADD COLUMN parent_embedding_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE file_shares
    ADD COLUMN parent_embedding_error TEXT;
"""
_MIGRATION_V8 = ""
_MIGRATION_V9 = """
UPDATE file_contents
SET processing_attempts=0, last_error_code=NULL
WHERE processing_status='indexed'
  AND caption_ocr IS NOT NULL AND caption_ocr!=''
  AND text_content_embedding_json IS NULL;
UPDATE file_shares
SET parent_embedding_attempts=0, parent_embedding_error=NULL
WHERE tombstoned_at IS NULL
  AND parent_embedding_json IS NULL
  AND (thread_context IS NOT NULL OR upload_text IS NOT NULL);
"""
_MIGRATIONS = {
    1: _SCHEMA,
    2: _MIGRATION_V2,
    3: _MIGRATION_V3,
    4: _MIGRATION_V4,
    5: _MIGRATION_V5,
    6: _MIGRATION_V6,
    7: _MIGRATION_V7,
    8: _MIGRATION_V8,
    9: _MIGRATION_V9,
}

_MESSAGE_SEARCH_INDEX_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS message_search_fts USING fts5(
    text,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS message_search_trigram USING fts5(
    text,
    tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS message_search_insert
AFTER INSERT ON messages WHEN new.deleted_at IS NULL BEGIN
    INSERT INTO message_search_fts(rowid, text)
    VALUES (new.rowid, ringo_search_index_text(new.text));
    INSERT INTO message_search_trigram(rowid, text)
    VALUES (new.rowid, ringo_search_normalize(new.text));
END;
CREATE TRIGGER IF NOT EXISTS message_search_delete
AFTER DELETE ON messages BEGIN
    DELETE FROM message_search_fts WHERE rowid = old.rowid;
    DELETE FROM message_search_trigram WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS message_search_update
AFTER UPDATE OF text, deleted_at ON messages BEGIN
    DELETE FROM message_search_fts WHERE rowid = old.rowid;
    DELETE FROM message_search_trigram WHERE rowid = old.rowid;
    INSERT INTO message_search_fts(rowid, text)
    SELECT new.rowid, ringo_search_index_text(new.text)
    WHERE new.deleted_at IS NULL;
    INSERT INTO message_search_trigram(rowid, text)
    SELECT new.rowid, ringo_search_normalize(new.text)
    WHERE new.deleted_at IS NULL;
END;
"""

_FILE_SEARCH_INDEX_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS file_content_search_fts USING fts5(
    file_name,
    caption_ocr,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS file_content_search_trigram USING fts5(
    file_name,
    caption_ocr,
    tokenize='trigram'
);
CREATE VIRTUAL TABLE IF NOT EXISTS file_share_search_fts USING fts5(
    upload_text,
    thread_context,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS file_share_search_trigram USING fts5(
    upload_text,
    thread_context,
    tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS file_content_search_insert
AFTER INSERT ON file_contents WHEN new.processing_status!='deleted' BEGIN
    INSERT INTO file_content_search_fts(rowid, file_name, caption_ocr)
    VALUES (new.rowid, ringo_search_index_text(new.file_name),
            ringo_search_index_text(new.caption_ocr));
    INSERT INTO file_content_search_trigram(rowid, file_name, caption_ocr)
    VALUES (new.rowid, ringo_search_normalize(new.file_name),
            ringo_search_normalize(new.caption_ocr));
END;
CREATE TRIGGER IF NOT EXISTS file_content_search_delete
AFTER DELETE ON file_contents BEGIN
    DELETE FROM file_content_search_fts WHERE rowid=old.rowid;
    DELETE FROM file_content_search_trigram WHERE rowid=old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS file_content_search_update
AFTER UPDATE OF file_name, caption_ocr, processing_status ON file_contents BEGIN
    DELETE FROM file_content_search_fts WHERE rowid=old.rowid;
    DELETE FROM file_content_search_trigram WHERE rowid=old.rowid;
    INSERT INTO file_content_search_fts(rowid, file_name, caption_ocr)
    SELECT new.rowid, ringo_search_index_text(new.file_name),
           ringo_search_index_text(new.caption_ocr)
    WHERE new.processing_status!='deleted';
    INSERT INTO file_content_search_trigram(rowid, file_name, caption_ocr)
    SELECT new.rowid, ringo_search_normalize(new.file_name),
           ringo_search_normalize(new.caption_ocr)
    WHERE new.processing_status!='deleted';
END;
CREATE TRIGGER IF NOT EXISTS file_share_search_insert
AFTER INSERT ON file_shares WHEN new.tombstoned_at IS NULL BEGIN
    INSERT INTO file_share_search_fts(rowid, upload_text, thread_context)
    VALUES (new.rowid, ringo_search_index_text(new.upload_text),
            ringo_search_index_text(new.thread_context));
    INSERT INTO file_share_search_trigram(rowid, upload_text, thread_context)
    VALUES (new.rowid, ringo_search_normalize(new.upload_text),
            ringo_search_normalize(new.thread_context));
END;
CREATE TRIGGER IF NOT EXISTS file_share_search_delete
AFTER DELETE ON file_shares BEGIN
    DELETE FROM file_share_search_fts WHERE rowid=old.rowid;
    DELETE FROM file_share_search_trigram WHERE rowid=old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS file_share_search_update
AFTER UPDATE OF upload_text, thread_context, tombstoned_at ON file_shares BEGIN
    DELETE FROM file_share_search_fts WHERE rowid=old.rowid;
    DELETE FROM file_share_search_trigram WHERE rowid=old.rowid;
    INSERT INTO file_share_search_fts(rowid, upload_text, thread_context)
    SELECT new.rowid, ringo_search_index_text(new.upload_text),
           ringo_search_index_text(new.thread_context)
    WHERE new.tombstoned_at IS NULL;
    INSERT INTO file_share_search_trigram(rowid, upload_text, thread_context)
    SELECT new.rowid, ringo_search_normalize(new.upload_text),
           ringo_search_normalize(new.thread_context)
    WHERE new.tombstoned_at IS NULL;
END;
"""


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
        self.message_search_index_available = False
        self.file_search_index_available = False
        self._migrate()
        cleanup_stale_temp_files()

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
            rotate_file_index_generation = 0 < current_version < 7
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"message store schema {current_version} is newer than supported "
                    f"schema {SCHEMA_VERSION}"
                )
            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                script = _MIGRATIONS.get(version)
                if script is None:
                    raise RuntimeError(f"missing message store migration {version}")
                if version == 6 and any(
                    str(row[1]) == "processing_attempts"
                    for row in conn.execute("PRAGMA table_info(file_contents)")
                ):
                    script = ""
                if version == 7:
                    existing_columns = {
                        str(row[1])
                        for row in conn.execute("PRAGMA table_info(file_shares)")
                    }
                    if "parent_embedding_json" in existing_columns:
                        script = ""
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
            generation = str(uuid.uuid4())
            if rotate_file_index_generation:
                conn.execute(
                    "INSERT INTO schema_meta(key, value) VALUES "
                    "('file_index_store_generation', ?) ON CONFLICT(key) "
                    "DO UPDATE SET value=excluded.value",
                    (generation,),
                )
            else:
                conn.execute(
                    "INSERT INTO schema_meta(key, value) VALUES "
                    "('file_index_store_generation', ?) ON CONFLICT(key) DO NOTHING",
                    (generation,),
                )
            self.store_generation = str(
                conn.execute(
                    "SELECT value FROM schema_meta "
                    "WHERE key='file_index_store_generation'"
                ).fetchone()[0]
            )
            self.message_search_index_available = (
                self._ensure_message_search_index(conn)
            )
            self.file_search_index_available = self._ensure_file_search_index(
                conn
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _ensure_message_search_index(conn: Any) -> bool:
        """Create and backfill optional FTS indexes without risking store startup."""
        try:
            conn.executescript(_MESSAGE_SEARCH_INDEX_SQL)
            version = conn.execute(
                "SELECT value FROM schema_meta "
                "WHERE key='message_search_index_version'"
            ).fetchone()
            if version is None or str(version[0]) != "2":
                conn.execute("DELETE FROM message_search_fts")
                conn.execute("DELETE FROM message_search_trigram")
                conn.execute(
                    "INSERT INTO message_search_fts(rowid, text) "
                    "SELECT rowid, ringo_search_index_text(text) FROM messages "
                    "WHERE deleted_at IS NULL"
                )
                conn.execute(
                    "INSERT INTO message_search_trigram(rowid, text) "
                    "SELECT rowid, ringo_search_normalize(text) FROM messages "
                    "WHERE deleted_at IS NULL"
                )
                conn.execute(
                    "INSERT INTO schema_meta(key, value) VALUES "
                    "('message_search_index_version', '2') "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
                )
            return True
        except Exception as exc:
            logger.warning(
                "Message search index unavailable; local search will fail open: %s",
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _ensure_file_search_index(conn: Any) -> bool:
        """Create and backfill optional file FTS indexes without blocking startup."""
        try:
            conn.executescript(_FILE_SEARCH_INDEX_SQL)
            version = conn.execute(
                "SELECT value FROM schema_meta "
                "WHERE key='file_search_index_version'"
            ).fetchone()
            if version is None or str(version[0]) != "1":
                for table in (
                    "file_content_search_fts",
                    "file_content_search_trigram",
                    "file_share_search_fts",
                    "file_share_search_trigram",
                ):
                    conn.execute(f"DELETE FROM {table}")
                conn.execute(
                    "INSERT INTO file_content_search_fts"
                    "(rowid, file_name, caption_ocr) "
                    "SELECT rowid, ringo_search_index_text(file_name), "
                    "ringo_search_index_text(caption_ocr) FROM file_contents "
                    "WHERE processing_status!='deleted'"
                )
                conn.execute(
                    "INSERT INTO file_content_search_trigram"
                    "(rowid, file_name, caption_ocr) "
                    "SELECT rowid, ringo_search_normalize(file_name), "
                    "ringo_search_normalize(caption_ocr) FROM file_contents "
                    "WHERE processing_status!='deleted'"
                )
                conn.execute(
                    "INSERT INTO file_share_search_fts"
                    "(rowid, upload_text, thread_context) "
                    "SELECT rowid, ringo_search_index_text(upload_text), "
                    "ringo_search_index_text(thread_context) FROM file_shares "
                    "WHERE tombstoned_at IS NULL"
                )
                conn.execute(
                    "INSERT INTO file_share_search_trigram"
                    "(rowid, upload_text, thread_context) "
                    "SELECT rowid, ringo_search_normalize(upload_text), "
                    "ringo_search_normalize(thread_context) FROM file_shares "
                    "WHERE tombstoned_at IS NULL"
                )
                conn.execute(
                    "INSERT INTO schema_meta(key, value) VALUES "
                    "('file_search_index_version', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
                )
            return True
        except Exception as exc:
            logger.warning(
                "File search index unavailable; local search will fail open: %s",
                type(exc).__name__,
            )
            return False

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
        elif event_type in {"file.changed", "file.deleted", "file.unshared"}:
            # File lifecycle is applied by the independent, versioned command
            # path after this encrypted delivery is ACKed.
            pass
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
        conn.execute(
            "DELETE FROM file_index_runs WHERE provider=? AND workspace_id=?",
            (provider, workspace_id),
        )
        for table in (
            "file_graph_ingest_state",
            "file_tombstones",
            "file_shares",
            "file_contents",
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

    def apply_file_command(self, request: dict[str, Any]) -> dict[str, Any]:
        """Apply one independently authenticated file-index command."""
        request = dict(request)
        access_token = str(request.pop("provider_access_token", "") or "")
        if str(request.get("project_id") or "") != self.project_id:
            raise ValueError("file index project mismatch")
        if str(request.get("store_generation") or "") != self.store_generation:
            raise ValueError("file index store generation mismatch")
        now = datetime.now(timezone.utc).isoformat()
        with self._writer_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            hydrated = self._hydrate_file_command(conn, request)
            if hydrated is None:
                conn.commit()
                return {
                    "status": "deferred",
                    "reason": "message_not_available",
                    "store_generation": self.store_generation,
                }
            status = self._apply_file_command(conn, hydrated, now)
            conn.commit()
        indexed = 0
        if (
            status == "applied"
            and str(hydrated.get("operation") or "")
            in {"upsert_share", "refresh_file"}
            and access_token
        ):
            indexed = int(
                self._process_file_content(
                    provider=str(hydrated.get("provider") or ""),
                    workspace_id=str(hydrated.get("workspace_id") or ""),
                    file_id=str(hydrated.get("file_id") or ""),
                    access_token=access_token,
                )
            )
            if str(hydrated.get("operation") or "") == "upsert_share":
                self._process_pending_parent_embeddings(
                    provider=str(hydrated.get("provider") or ""),
                    workspace_id=str(hydrated.get("workspace_id") or ""),
                    limit=1,
                )
            if not indexed:
                with self._connect() as conn:
                    content = conn.execute(
                        "SELECT processing_status, last_error_code "
                        "FROM file_contents WHERE project_id=? AND provider=? "
                        "AND workspace_id=? AND file_id=?",
                        (
                            self.project_id,
                            str(hydrated.get("provider") or ""),
                            str(hydrated.get("workspace_id") or ""),
                            str(hydrated.get("file_id") or ""),
                        ),
                    ).fetchone()
                if (
                    content is not None
                    and str(content["processing_status"]) == "metadata_only"
                    and content["last_error_code"]
                ):
                    return {
                        "status": "deferred",
                        "reason": str(content["last_error_code"]),
                        "store_generation": self.store_generation,
                        "indexed_contents": 0,
                    }
        return {
            "status": status,
            "store_generation": self.store_generation,
            "indexed_contents": indexed,
        }

    def _hydrate_file_command(
        self,
        conn: Any,
        request: dict[str, Any],
    ) -> dict[str, Any] | None:
        if str(request.get("operation") or "") != "upsert_share":
            return request
        if any(
            key in request
            for key in (
                "content_version",
                "context_version",
                "file_name",
                "content_sha256",
                "processing_status",
            )
        ):
            return request
        provider = str(request.get("provider") or "")
        workspace_id = str(request.get("workspace_id") or "")
        conversation_id = str(request.get("conversation_id") or "")
        message_id = str(request.get("provider_message_id") or "")
        file_id = str(request.get("file_id") or "")
        row = conn.execute(
            "SELECT * FROM messages WHERE project_id=? AND provider=? "
            "AND workspace_id=? AND conversation_id=? AND provider_message_id=? "
            "AND deleted_at IS NULL",
            (
                self.project_id,
                provider,
                workspace_id,
                conversation_id,
                message_id,
            ),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["provider_payload_json"] or "{}")
        except json.JSONDecodeError:
            return None
        message_payload = (
            payload.get("message")
            if isinstance(payload, dict)
            and isinstance(payload.get("message"), dict)
            else payload
        )
        files = (
            message_payload.get("files")
            if isinstance(message_payload, dict)
            else None
        )
        file_data = next(
            (
                item
                for item in files or []
                if isinstance(item, dict) and str(item.get("id") or "") == file_id
            ),
            None,
        )
        if file_data is None:
            return None
        context_version = self._canonical_file_version(row["updated_at"])
        content_version = self._canonical_file_version(
            file_data.get("timestamp")
            or file_data.get("created")
            or row["occurred_at"]
        )
        return {
            **request,
            "source_version": max(
                int(request.get("source_version") or 0),
                content_version,
                context_version,
            ),
            "content_version": content_version,
            "context_version": context_version,
            "file_name": str(
                file_data.get("name") or file_data.get("title") or ""
            )
            or None,
            "mime_type": str(file_data.get("mimetype") or "") or None,
            "byte_size": file_data.get("size"),
            "uploaded_at": str(
                file_data.get("timestamp")
                or file_data.get("created")
                or row["occurred_at"]
            ),
            "permalink": str(
                file_data.get("permalink")
                or file_data.get("permalink_public")
                or ""
            )
            or None,
            "uploader_id": str(
                file_data.get("user") or row["sender_id"] or ""
            )
            or None,
            "upload_text": str(row["text"] or "") or None,
            "thread_context": self._file_thread_context(conn, row),
            "shared_at": str(row["occurred_at"]),
        }

    @staticmethod
    def _canonical_file_version(value: Any) -> int:
        raw = str(value or "").strip()
        if not raw:
            return 0
        try:
            return int(Decimal(raw) * 1_000_000)
        except (InvalidOperation, ValueError):
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return 0
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1_000_000)

    @staticmethod
    def _allowed_file_sources(
        raw_sources: Any,
        *,
        provider: str,
        workspace_id: str,
    ) -> list[str]:
        if not isinstance(raw_sources, list):
            raise ValueError("file index allowed sources are required")
        conversations: set[str] = set()
        for source_id in raw_sources:
            parts = str(source_id).split(":", 2)
            if len(parts) != 3 or not all(parts):
                raise ValueError("invalid allowed source id")
            if parts[0] == provider and parts[1] == workspace_id:
                conversations.add(parts[2])
        return sorted(conversations)

    def reconcile_file_window(self, request: dict[str, Any]) -> dict[str, Any]:
        """Scan one bounded batch of already-encrypted local messages."""
        request = dict(request)
        access_token = str(request.pop("provider_access_token", "") or "")
        if str(request.get("project_id") or "") != self.project_id:
            raise ValueError("file index project mismatch")
        if str(request.get("store_generation") or "") != self.store_generation:
            raise ValueError("file index store generation mismatch")
        provider = str(request.get("provider") or "")
        workspace_id = str(request.get("workspace_id") or "")
        start = str(request.get("window_started_at") or "")
        end = str(request.get("window_ended_at") or "")
        scope_revision = str(request.get("allowed_scope_revision") or "")
        processing_signature = str(request.get("processing_signature") or "")
        if (
            not provider
            or not workspace_id
            or not start
            or not end
            or start >= end
            or not scope_revision
            or not processing_signature
        ):
            raise ValueError("bounded file reconcile scope is required")
        conversations = self._allowed_file_sources(
            request.get("allowed_source_ids"),
            provider=provider,
            workspace_id=workspace_id,
        )
        limit = max(1, min(int(request.get("batch_limit") or 100), 500))
        now = datetime.now(timezone.utc).isoformat()
        run_scope = (
            provider,
            workspace_id,
            self.store_generation,
            start,
            end,
        )

        with self._writer_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                "SELECT * FROM file_index_runs WHERE provider=? AND workspace_id=? "
                "AND store_generation=? AND window_started_at=? AND window_ended_at=?",
                run_scope,
            ).fetchone()
            reset = (
                run is not None
                and (
                    str(run["allowed_scope_revision"]) != scope_revision
                    or str(run["processing_signature"]) != processing_signature
                )
            )
            if run is None:
                conn.execute(
                    "INSERT INTO file_index_runs(provider, workspace_id, "
                    "store_generation, window_started_at, window_ended_at, "
                    "allowed_scope_revision, status, processing_signature, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)",
                    (*run_scope, scope_revision, processing_signature, now),
                )
                cursor: dict[str, str] | None = None
            elif reset:
                conn.execute(
                    "UPDATE file_index_runs SET allowed_scope_revision=?, "
                    "message_scan_cursor_json=NULL, status='running', "
                    "processing_signature=?, scanned_messages=0, "
                    "discovered_shares=0, indexed_contents=0, updated_at=?, "
                    "completed_at=NULL WHERE provider=? AND workspace_id=? "
                    "AND store_generation=? AND window_started_at=? "
                    "AND window_ended_at=?",
                    (scope_revision, processing_signature, now, *run_scope),
                )
                cursor = None
            else:
                try:
                    cursor = json.loads(run["message_scan_cursor_json"] or "null")
                except json.JSONDecodeError:
                    cursor = None

            if not conversations:
                conn.execute(
                    "UPDATE file_index_runs SET status='complete', updated_at=?, "
                    "completed_at=? WHERE provider=? AND workspace_id=? "
                    "AND store_generation=? AND window_started_at=? "
                    "AND window_ended_at=?",
                    (now, now, *run_scope),
                )
                conn.commit()
                return {
                    "status": "complete",
                    "store_generation": self.store_generation,
                    "scanned_messages": 0,
                    "discovered_shares": 0,
                    "indexed_contents": 0,
                }

            placeholders = ",".join("?" for _ in conversations)
            sql = (
                "SELECT * FROM messages WHERE project_id=? AND provider=? "
                "AND workspace_id=? AND conversation_id IN ("
                + placeholders
                + ") AND occurred_at>=? AND occurred_at<=? AND deleted_at IS NULL"
            )
            params: list[Any] = [
                self.project_id,
                provider,
                workspace_id,
                *conversations,
                start,
                end,
            ]
            if isinstance(cursor, dict):
                cursor_values = [
                    str(cursor.get(key) or "")
                    for key in (
                        "occurred_at",
                        "conversation_id",
                        "provider_message_id",
                    )
                ]
                if all(cursor_values):
                    sql += (
                        " AND (occurred_at, conversation_id, provider_message_id) "
                        "> (?, ?, ?)"
                    )
                    params.extend(cursor_values)
            sql += (
                " ORDER BY occurred_at, conversation_id, provider_message_id LIMIT ?"
            )
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            discovered = 0
            indexed = 0
            for row in rows:
                try:
                    payload = json.loads(row["provider_payload_json"] or "{}")
                except json.JSONDecodeError:
                    payload = {}
                message_payload = (
                    payload.get("message")
                    if isinstance(payload, dict)
                    and isinstance(payload.get("message"), dict)
                    else payload
                )
                files = (
                    message_payload.get("files")
                    if isinstance(message_payload, dict)
                    else None
                )
                if not isinstance(files, list):
                    continue
                thread_context = self._file_thread_context(conn, row)
                context_version = self._canonical_file_version(row["updated_at"])
                for file_data in files:
                    if not isinstance(file_data, dict):
                        continue
                    file_id = str(file_data.get("id") or "")
                    if not file_id:
                        continue
                    content_version = self._canonical_file_version(
                        file_data.get("timestamp")
                        or file_data.get("created")
                        or row["occurred_at"]
                    )
                    self._apply_file_command(
                        conn,
                        {
                            "operation": "upsert_share",
                            "provider": provider,
                            "workspace_id": workspace_id,
                            "file_id": file_id,
                            "conversation_id": str(row["conversation_id"]),
                            "provider_message_id": str(
                                row["provider_message_id"]
                            ),
                            "source_version": max(
                                content_version,
                                context_version,
                            ),
                            "content_version": content_version,
                            "context_version": context_version,
                            "file_name": str(
                                file_data.get("name")
                                or file_data.get("title")
                                or ""
                            )
                            or None,
                            "mime_type": str(file_data.get("mimetype") or "")
                            or None,
                            "byte_size": file_data.get("size"),
                            "uploaded_at": str(
                                file_data.get("timestamp")
                                or file_data.get("created")
                                or row["occurred_at"]
                            ),
                            "permalink": str(
                                file_data.get("permalink")
                                or file_data.get("permalink_public")
                                or ""
                            )
                            or None,
                            "uploader_id": str(
                                file_data.get("user")
                                or row["sender_id"]
                                or ""
                            )
                            or None,
                            "upload_text": str(row["text"] or "") or None,
                            "thread_context": thread_context,
                            "shared_at": str(row["occurred_at"]),
                        },
                        now,
                    )
                    discovered += 1

            scan_complete = len(rows) < limit
            next_cursor = None
            if rows:
                last = rows[-1]
                next_cursor = {
                    "occurred_at": str(last["occurred_at"]),
                    "conversation_id": str(last["conversation_id"]),
                    "provider_message_id": str(last["provider_message_id"]),
                }
            conn.execute(
                "UPDATE file_index_runs SET message_scan_cursor_json=?, status=?, "
                "scanned_messages=scanned_messages+?, "
                "discovered_shares=discovered_shares+?, "
                "indexed_contents=indexed_contents+?, updated_at=?, completed_at=? "
                "WHERE provider=? AND workspace_id=? AND store_generation=? "
                "AND window_started_at=? AND window_ended_at=?",
                (
                    json.dumps(next_cursor, separators=(",", ":"))
                    if next_cursor is not None
                    else None,
                    "complete" if scan_complete else "running",
                    len(rows),
                    discovered,
                    indexed,
                    now,
                    now if scan_complete else None,
                    *run_scope,
                ),
            )
            conn.commit()
        processed = 0
        contextualized = self._process_pending_parent_embeddings(
            provider=provider,
            workspace_id=workspace_id,
            limit=FILE_EMBEDDING_BACKFILL_LIMIT,
        )
        embedded_contents = self._process_pending_content_embeddings(
            provider=provider,
            workspace_id=workspace_id,
            limit=FILE_EMBEDDING_BACKFILL_LIMIT,
        )
        pending = 0
        pending_content_embeddings = 0
        pending_parent_embeddings = 0
        if access_token:
            processed = self._process_pending_file_contents(
                provider=provider,
                workspace_id=workspace_id,
                access_token=access_token,
                limit=1,
            )
            with self._writer_lock, self._connect() as conn:
                pending = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM file_contents WHERE project_id=? "
                        "AND provider=? AND workspace_id=? "
                        "AND processing_status='metadata_only'",
                        (self.project_id, provider, workspace_id),
                    ).fetchone()[0]
                )
                retryable_pending = (
                    conn.execute(
                        "SELECT 1 FROM file_contents WHERE project_id=? "
                        "AND provider=? AND workspace_id=? "
                        "AND processing_status='metadata_only' "
                        "AND last_error_code IS NOT NULL LIMIT 1",
                        (self.project_id, provider, workspace_id),
                    ).fetchone()
                    is not None
                )
                pending_content_embeddings = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM file_contents WHERE project_id=? "
                        "AND provider=? AND workspace_id=? "
                        "AND processing_status='indexed' "
                        "AND caption_ocr IS NOT NULL AND caption_ocr!='' "
                        "AND text_content_embedding_json IS NULL "
                        "AND processing_attempts<?",
                        (
                            self.project_id,
                            provider,
                            workspace_id,
                            MAX_FILE_PROCESSING_ATTEMPTS,
                        ),
                    ).fetchone()[0]
                )
                pending_parent_embeddings = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM file_shares WHERE project_id=? "
                        "AND provider=? AND workspace_id=? "
                        "AND tombstoned_at IS NULL "
                        "AND parent_embedding_json IS NULL "
                        "AND parent_embedding_attempts<? "
                        "AND (thread_context IS NOT NULL OR upload_text IS NOT NULL)",
                        (
                            self.project_id,
                            provider,
                            workspace_id,
                            MAX_FILE_PROCESSING_ATTEMPTS,
                        ),
                    ).fetchone()[0]
                )
                complete = (
                    scan_complete
                    and pending == 0
                    and pending_content_embeddings == 0
                    and pending_parent_embeddings == 0
                )
                conn.execute(
                    "UPDATE file_index_runs SET status=?, "
                    "indexed_contents=indexed_contents+?, updated_at=?, "
                    "completed_at=? WHERE provider=? AND workspace_id=? "
                    "AND store_generation=? AND window_started_at=? "
                    "AND window_ended_at=?",
                    (
                        "complete" if complete else "running",
                        processed,
                        datetime.now(timezone.utc).isoformat(),
                        datetime.now(timezone.utc).isoformat()
                        if complete
                        else None,
                        *run_scope,
                    ),
                )
                conn.commit()
        else:
            complete = scan_complete
            retryable_pending = False
        if retryable_pending:
            return {
                "status": "deferred",
                "reason": "file_content_retryable",
                "store_generation": self.store_generation,
                "scanned_messages": len(rows),
                "discovered_shares": discovered,
                "indexed_contents": processed,
                "pending_contents": pending,
                "embedded_contents": embedded_contents,
                "pending_content_embeddings": pending_content_embeddings,
                "contextualized_shares": contextualized,
                "pending_parent_embeddings": pending_parent_embeddings,
            }
        if complete:
            self._expire_file_window(
                provider=provider,
                workspace_id=workspace_id,
                window_started_at=start,
            )
        return {
            "status": "complete" if complete else "continuation",
            "store_generation": self.store_generation,
            "scanned_messages": len(rows),
            "discovered_shares": discovered,
            "indexed_contents": processed,
            "pending_contents": pending,
            "embedded_contents": embedded_contents,
            "pending_content_embeddings": pending_content_embeddings,
            "contextualized_shares": contextualized,
            "pending_parent_embeddings": pending_parent_embeddings,
        }

    def _expire_file_window(
        self,
        *,
        provider: str,
        workspace_id: str,
        window_started_at: str,
    ) -> None:
        tombstone_version = self._canonical_file_version(window_started_at)
        now = datetime.now(timezone.utc).isoformat()
        with self._writer_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE file_shares SET tombstone_version=?, tombstoned_at=?, "
                "updated_at=? WHERE project_id=? AND provider=? "
                "AND workspace_id=? AND tombstoned_at IS NULL "
                "AND shared_at IS NOT NULL AND shared_at<?",
                (
                    tombstone_version,
                    now,
                    now,
                    self.project_id,
                    provider,
                    workspace_id,
                    window_started_at,
                ),
            )
            conn.execute(
                "DELETE FROM file_contents WHERE project_id=? AND provider=? "
                "AND workspace_id=? AND NOT EXISTS ("
                "SELECT 1 FROM file_shares fs WHERE "
                "fs.project_id=file_contents.project_id AND "
                "fs.provider=file_contents.provider AND "
                "fs.workspace_id=file_contents.workspace_id AND "
                "fs.file_id=file_contents.file_id AND fs.tombstoned_at IS NULL)",
                (self.project_id, provider, workspace_id),
            )
            conn.execute(
                "DELETE FROM file_graph_ingest_state WHERE project_id=? "
                "AND provider=? AND workspace_id=? AND NOT EXISTS ("
                "SELECT 1 FROM file_shares fs WHERE "
                "fs.project_id=file_graph_ingest_state.project_id AND "
                "fs.provider=file_graph_ingest_state.provider AND "
                "fs.workspace_id=file_graph_ingest_state.workspace_id AND "
                "fs.file_id=file_graph_ingest_state.file_id AND "
                "fs.conversation_id=file_graph_ingest_state.conversation_id AND "
                "fs.provider_message_id=file_graph_ingest_state.provider_message_id "
                "AND fs.tombstoned_at IS NULL)",
                (self.project_id, provider, workspace_id),
            )
            conn.commit()

    def _process_pending_file_contents(
        self,
        *,
        provider: str,
        workspace_id: str,
        access_token: str,
        limit: int,
    ) -> int:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_id FROM file_contents WHERE project_id=? "
                "AND provider=? AND workspace_id=? "
                "AND processing_status='metadata_only' ORDER BY uploaded_at, file_id "
                "LIMIT ?",
                (self.project_id, provider, workspace_id, max(1, min(limit, 3))),
            ).fetchall()
        return sum(
            int(
                self._process_file_content(
                    provider=provider,
                    workspace_id=workspace_id,
                    file_id=str(row["file_id"]),
                    access_token=access_token,
                )
            )
            for row in rows
        )

    def _process_file_content(
        self,
        *,
        provider: str,
        workspace_id: str,
        file_id: str,
        access_token: str,
    ) -> bool:
        if provider != "slack" or not file_id or not access_token:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_version, processing_status, processing_attempts "
                "FROM file_contents "
                "WHERE project_id=? AND provider=? AND workspace_id=? AND file_id=?",
                (self.project_id, provider, workspace_id, file_id),
            ).fetchone()
        if row is None or str(row["processing_status"]) != "metadata_only":
            return False
        content_version = int(row["content_version"])
        result = process_slack_file(
            file_id,
            access_token,
            lambda digest, mime_type: self._derived_content_by_hash(
                provider=provider,
                workspace_id=workspace_id,
                file_id=file_id,
                content_sha256=digest,
                mime_type=mime_type,
            ),
        )
        now = datetime.now(timezone.utc).isoformat()
        processing_status = str(result["processing_status"])
        processing_attempts = 0
        if processing_status == "metadata_only":
            processing_attempts = int(row["processing_attempts"]) + 1
            if processing_attempts >= MAX_FILE_PROCESSING_ATTEMPTS:
                processing_status = "unavailable"
        text_embedding = result.get("text_content_embedding")
        image_embedding = result.get("image_embedding")
        with self._writer_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                "UPDATE file_contents SET content_sha256=?, "
                "file_name=COALESCE(?, file_name), mime_type=COALESCE(?, mime_type), "
                "byte_size=COALESCE(?, byte_size), "
                "uploaded_at=COALESCE(?, uploaded_at), "
                "permalink=COALESCE(?, permalink), processing_status=?, "
                "caption_ocr=?, text_content_embedding_json=?, "
                "image_embedding_json=?, caption_model=?, "
                "caption_prompt_version=?, text_embedding_model=?, "
                "text_embedding_dimension=?, image_embedding_model=?, "
                "image_embedding_dimension=?, last_error_code=?, "
                "processing_attempts=?, updated_at=? "
                "WHERE project_id=? AND provider=? AND workspace_id=? AND file_id=? "
                "AND content_version=? AND processing_status='metadata_only'",
                (
                    result.get("content_sha256"),
                    result.get("file_name"),
                    result.get("mime_type"),
                    result.get("byte_size"),
                    result.get("uploaded_at"),
                    result.get("permalink"),
                    processing_status,
                    result.get("caption_ocr"),
                    json.dumps(text_embedding, separators=(",", ":"))
                    if text_embedding is not None
                    else None,
                    json.dumps(image_embedding, separators=(",", ":"))
                    if image_embedding is not None
                    else None,
                    result.get("caption_model"),
                    result.get("caption_prompt_version"),
                    result.get("text_embedding_model"),
                    result.get("text_embedding_dimension"),
                    result.get("image_embedding_model"),
                    result.get("image_embedding_dimension"),
                    result.get("last_error_code"),
                    processing_attempts,
                    now,
                    self.project_id,
                    provider,
                    workspace_id,
                    file_id,
                    content_version,
                ),
            ).rowcount
            conn.commit()
        return bool(updated and processing_status != "metadata_only")

    def _process_pending_content_embeddings(
        self,
        *,
        provider: str,
        workspace_id: str,
        limit: int,
    ) -> int:
        """Backfill embeddings from stored captions without re-downloading files."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rowid AS content_rowid, file_name, mime_type, caption_ocr "
                "FROM file_contents WHERE project_id=? AND provider=? "
                "AND workspace_id=? AND processing_status='indexed' "
                "AND caption_ocr IS NOT NULL AND caption_ocr!='' "
                "AND text_content_embedding_json IS NULL "
                "AND processing_attempts<? ORDER BY uploaded_at, file_id LIMIT ?",
                (
                    self.project_id,
                    provider,
                    workspace_id,
                    MAX_FILE_PROCESSING_ATTEMPTS,
                    max(1, min(limit, FILE_EMBEDDING_BACKFILL_LIMIT)),
                ),
            ).fetchall()
        if not rows:
            return 0
        texts = [
            " ".join(
                value
                for value in (
                    str(row["file_name"] or "").strip(),
                    str(row["caption_ocr"] or "").strip(),
                )
                if value
            )
            for row in rows
        ]
        try:
            vectors, model = embed_texts(texts)
        except FileProcessingError as exc:
            now = datetime.now(timezone.utc).isoformat()
            with self._writer_lock, self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for row in rows:
                    conn.execute(
                        "UPDATE file_contents SET "
                        "processing_attempts=processing_attempts+1, "
                        "last_error_code=?, updated_at=? "
                        "WHERE rowid=? AND text_content_embedding_json IS NULL",
                        (exc.code, now, int(row["content_rowid"])),
                    )
                conn.commit()
            return 0

        now = datetime.now(timezone.utc).isoformat()
        processed = 0
        with self._writer_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for row, vector in zip(rows, vectors, strict=True):
                encoded = json.dumps(vector, separators=(",", ":"))
                is_image = str(row["mime_type"] or "").lower().startswith(
                    "image/"
                )
                processed += conn.execute(
                    "UPDATE file_contents SET text_content_embedding_json=?, "
                    "text_embedding_model=?, text_embedding_dimension=?, "
                    "image_embedding_json=CASE WHEN ? THEN ? "
                    "ELSE image_embedding_json END, "
                    "image_embedding_model=CASE WHEN ? THEN ? "
                    "ELSE image_embedding_model END, "
                    "image_embedding_dimension=CASE WHEN ? THEN ? "
                    "ELSE image_embedding_dimension END, "
                    "processing_attempts=0, last_error_code=NULL, updated_at=? "
                    "WHERE rowid=? AND text_content_embedding_json IS NULL",
                    (
                        encoded,
                        model,
                        len(vector),
                        is_image,
                        encoded,
                        is_image,
                        f"caption-text:{model}",
                        is_image,
                        len(vector),
                        now,
                        int(row["content_rowid"]),
                    ),
                ).rowcount
            conn.commit()
        return processed

    def _process_pending_parent_embeddings(
        self,
        *,
        provider: str,
        workspace_id: str,
        limit: int,
    ) -> int:
        """Embed a bounded set of unique thread/upload parent contexts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT MAX(thread_context) AS thread_context, "
                "MAX(upload_text) AS upload_text FROM file_shares "
                "WHERE project_id=? AND provider=? AND workspace_id=? "
                "AND tombstoned_at IS NULL AND parent_embedding_json IS NULL "
                "AND parent_embedding_attempts<? "
                "AND (thread_context IS NOT NULL OR upload_text IS NOT NULL) "
                "GROUP BY CASE WHEN thread_context IS NOT NULL "
                "THEN 'thread:' || thread_context ELSE 'upload:' || upload_text END "
                "ORDER BY MIN(shared_at), MIN(file_id) LIMIT ?",
                (
                    self.project_id,
                    provider,
                    workspace_id,
                    MAX_FILE_PROCESSING_ATTEMPTS,
                    max(1, min(limit, 10)),
                ),
            ).fetchall()
        processed = 0
        for row in rows:
            thread_context = str(row["thread_context"] or "")
            upload_text = str(row["upload_text"] or "")
            parent_text = thread_context or upload_text
            if thread_context:
                match_sql = "thread_context=?"
                match_params = (thread_context,)
            else:
                match_sql = "thread_context IS NULL AND upload_text=?"
                match_params = (upload_text,)
            try:
                vector, model = embed_text(parent_text)
            except FileProcessingError as exc:
                with self._writer_lock, self._connect() as conn:
                    conn.execute(
                        "UPDATE file_shares SET "
                        "parent_embedding_attempts=parent_embedding_attempts+1, "
                        "parent_embedding_error=?, updated_at=? "
                        "WHERE project_id=? AND provider=? AND workspace_id=? "
                        "AND tombstoned_at IS NULL AND "
                        + match_sql
                        + " AND parent_embedding_json IS NULL",
                        (
                            exc.code,
                            datetime.now(timezone.utc).isoformat(),
                            self.project_id,
                            provider,
                            workspace_id,
                            *match_params,
                        ),
                    )
                    conn.commit()
                continue
            encoded = json.dumps(vector, separators=(",", ":"))
            now = datetime.now(timezone.utc).isoformat()
            with self._writer_lock, self._connect() as conn:
                updated = conn.execute(
                    "UPDATE file_shares SET parent_embedding_json=?, "
                    "parent_embedding_model=?, parent_embedding_dimension=?, "
                    "parent_embedding_attempts=0, parent_embedding_error=NULL, "
                    "updated_at=? WHERE project_id=? AND provider=? "
                    "AND workspace_id=? AND tombstoned_at IS NULL AND "
                    + match_sql
                    + " AND parent_embedding_json IS NULL",
                    (
                        encoded,
                        model,
                        len(vector),
                        now,
                        self.project_id,
                        provider,
                        workspace_id,
                        *match_params,
                    ),
                ).rowcount
                conn.commit()
            processed += int(updated)
        return processed

    def _derived_content_by_hash(
        self,
        *,
        provider: str,
        workspace_id: str,
        file_id: str,
        content_sha256: str,
        mime_type: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM file_contents WHERE project_id=? AND provider=? "
                "AND workspace_id=? AND file_id!=? AND content_sha256=? "
                "AND mime_type=? AND processing_status='indexed' "
                "ORDER BY updated_at DESC LIMIT 1",
                (
                    self.project_id,
                    provider,
                    workspace_id,
                    file_id,
                    content_sha256,
                    mime_type,
                ),
            ).fetchone()
        if row is None:
            return None
        try:
            text_embedding = json.loads(
                row["text_content_embedding_json"] or "null"
            )
            image_embedding = json.loads(row["image_embedding_json"] or "null")
        except json.JSONDecodeError:
            return None
        return {
            "caption_ocr": row["caption_ocr"],
            "text_content_embedding": text_embedding,
            "image_embedding": image_embedding,
            "caption_model": row["caption_model"],
            "caption_prompt_version": row["caption_prompt_version"],
            "text_embedding_model": row["text_embedding_model"],
            "text_embedding_dimension": row["text_embedding_dimension"],
            "image_embedding_model": row["image_embedding_model"],
            "image_embedding_dimension": row["image_embedding_dimension"],
        }

    @staticmethod
    def _file_search_tokens(value: Any) -> set[str]:
        return search_feature_set(value)

    @staticmethod
    def _file_search_cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    @classmethod
    def _file_search_match_query(
        cls,
        query: str,
        *,
        column: str | None = None,
        trigram: bool = False,
        require_all: bool = False,
    ) -> str:
        segments = search_segments(query)
        if trigram:
            segments = [segment for segment in segments if len(segment) >= 3]
        elif not require_all:
            segments = [*segments, *search_index_terms(query)]
        if require_all:
            match_query = " AND ".join(
                f'"{segment.replace(chr(34), chr(34) * 2)}"'
                for segment in list(dict.fromkeys(segments))[
                    :MESSAGE_SEARCH_MAX_QUERY_TERMS
                ]
                if segment
            )
        else:
            match_query = cls._message_search_match_query(segments)
        if match_query and column:
            return f"{column} : ({match_query})"
        return match_query

    @staticmethod
    def _file_search_fts_lane(
        conn: Any,
        *,
        table: str,
        match_query: str,
        weights: tuple[float, float],
    ) -> list[int]:
        if not match_query:
            return []
        rows = conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? "
            f"ORDER BY bm25({table}, ?, ?) LIMIT ?",
            (
                match_query,
                *weights,
                FILE_SEARCH_MAX_INDEX_MATCHES,
            ),
        ).fetchall()
        return [int(row["rowid"]) for row in rows]

    @staticmethod
    def _file_search_dense_lane(
        conn: Any,
        *,
        table: str,
        embedding_column: str,
        dimension_column: str,
        query_embedding: list[float],
    ) -> dict[int, float] | None:
        if not query_embedding:
            return {}
        try:
            conn.execute("SELECT vec_version()").fetchone()
            rows = conn.execute(
                f"SELECT rowid, 1.0-vec_distance_cosine("
                f"{embedding_column}, ?) AS score FROM {table} "
                f"WHERE {embedding_column} IS NOT NULL "
                f"AND COALESCE({dimension_column}, "
                f"json_array_length({embedding_column}))=? "
                "ORDER BY score DESC LIMIT ?",
                (
                    json.dumps(query_embedding, separators=(",", ":")),
                    len(query_embedding),
                    FILE_SEARCH_MAX_INDEX_MATCHES,
                ),
            ).fetchall()
        except Exception:
            return None
        return {
            int(row["rowid"]): float(row["score"] or 0.0)
            for row in rows
        }

    def search_file_index(self, request: dict[str, Any]) -> dict[str, Any]:
        """Fuse lexical and dense retrieval by thread, then expand its files."""
        request = dict(request)
        access_token = str(request.pop("provider_access_token", "") or "")
        if str(request.get("project_id") or "") != self.project_id:
            raise ValueError("file index project mismatch")
        if str(request.get("store_generation") or "") != self.store_generation:
            raise ValueError("file index store generation mismatch")
        provider = str(request.get("provider") or "")
        workspace_id = str(request.get("workspace_id") or "")
        query = str(request.get("query") or "").strip()
        context_query = str(request.get("context_query") or "").strip()
        scope_revision = str(request.get("allowed_scope_revision") or "")
        window_started_at = str(request.get("window_started_at") or "")
        if not provider or not workspace_id or not scope_revision:
            raise ValueError("file search scope is required")
        conversations = self._allowed_file_sources(
            request.get("allowed_source_ids"),
            provider=provider,
            workspace_id=workspace_id,
        )
        if not conversations:
            return {
                "status": "complete",
                "coverage_complete": True,
                "retrieve_count": 0,
                "rerank_count": 0,
                "inspected_image_count": 0,
                "files": [],
                "store_generation": self.store_generation,
            }
        placeholders = ",".join("?" for _ in conversations)
        row_select = (
            "SELECT fc.*, fc.rowid AS content_rowid, "
            "fs.rowid AS share_rowid, fs.conversation_id, "
            "fs.provider_message_id, fs.uploader_id, fs.upload_text, "
            "fs.thread_context, fs.shared_at, fs.parent_embedding_json, "
            "COALESCE(m.parent_message_id, fs.provider_message_id) "
            "AS thread_root_id, "
            "COUNT(*) OVER (PARTITION BY fs.project_id, fs.provider, "
            "fs.workspace_id, fs.conversation_id, fs.provider_message_id) "
            "AS message_file_count, "
            "i.display_name AS uploader_name, c.title AS conversation_name "
            "FROM file_contents fc JOIN file_shares fs ON "
            "fc.project_id=fs.project_id AND fc.provider=fs.provider AND "
            "fc.workspace_id=fs.workspace_id AND fc.file_id=fs.file_id "
            "LEFT JOIN messages m ON m.project_id=fs.project_id AND "
            "m.provider=fs.provider AND m.workspace_id=fs.workspace_id AND "
            "m.conversation_id=fs.conversation_id AND "
            "m.provider_message_id=fs.provider_message_id "
            "LEFT JOIN identities i ON i.project_id=fs.project_id AND "
            "i.provider=fs.provider AND i.workspace_id=fs.workspace_id AND "
            "i.external_user_id=fs.uploader_id "
            "LEFT JOIN conversations c ON c.project_id=fs.project_id AND "
            "c.provider=fs.provider AND c.workspace_id=fs.workspace_id AND "
            "c.conversation_id=fs.conversation_id "
            "WHERE fc.project_id=? AND fc.provider=? AND fc.workspace_id=? "
            f"AND fs.conversation_id IN ({placeholders}) "
            "AND fc.processing_status!='deleted' "
            "AND fs.tombstoned_at IS NULL "
            + ("AND fs.shared_at>=? " if window_started_at else "")
        )
        row_params: list[Any] = [
            self.project_id,
            provider,
            workspace_id,
            *conversations,
            *([window_started_at] if window_started_at else []),
        ]
        with self._connect() as conn:
            coverage_complete = (
                conn.execute(
                    "SELECT 1 FROM file_index_runs WHERE provider=? "
                    "AND workspace_id=? AND store_generation=? "
                    "AND allowed_scope_revision=? AND status='complete' LIMIT 1",
                    (
                        provider,
                        workspace_id,
                        self.store_generation,
                        scope_revision,
                    ),
                ).fetchone()
                is not None
            )

        query_tokens = self._file_search_tokens(query)
        uploader_filter = str(request.get("uploader_id") or "")
        raw_types = request.get("file_types")
        file_types = {
            str(value).lower()
            for value in raw_types or []
            if str(value or "").strip()
        } if isinstance(raw_types, list) else set()
        query_embedding: list[float] = []
        context_query_embedding: list[float] = []
        if query and context_query:
            try:
                embeddings, _ = embed_texts([query, context_query])
                query_embedding, context_query_embedding = embeddings
            except FileProcessingError:
                pass
        elif query:
            try:
                query_embedding, _ = embed_text(query)
            except FileProcessingError:
                pass
        lexical_lanes: dict[str, tuple[str, list[int]]] = {}
        direct_dense: dict[int, float] | None = {}
        parent_dense: dict[int, float] | None = {}
        with self._connect() as conn:
            if query and self.file_search_index_available:
                lexical_specs = (
                    (
                        "content_words",
                        "content",
                        "file_content_search_fts",
                        self._file_search_match_query(query),
                        (4.0, 1.0),
                    ),
                    (
                        "content_trigram",
                        "content",
                        "file_content_search_trigram",
                        self._file_search_match_query(query, trigram=True),
                        (4.0, 1.0),
                    ),
                    (
                        "upload_words",
                        "share",
                        "file_share_search_fts",
                        self._file_search_match_query(
                            query, column="upload_text"
                        ),
                        (2.0, 0.0),
                    ),
                    (
                        "upload_trigram",
                        "share",
                        "file_share_search_trigram",
                        self._file_search_match_query(
                            query, column="upload_text", trigram=True
                        ),
                        (2.0, 0.0),
                    ),
                    (
                        "content_precise",
                        "content",
                        "file_content_search_fts",
                        self._file_search_match_query(
                            query, require_all=True
                        ),
                        (4.0, 1.0),
                    ),
                    (
                        "upload_precise",
                        "share",
                        "file_share_search_fts",
                        self._file_search_match_query(
                            query,
                            column="upload_text",
                            require_all=True,
                        ),
                        (2.0, 0.0),
                    ),
                )
                for name, row_kind, table, match_query, weights in lexical_specs:
                    lexical_lanes[name] = (
                        row_kind,
                        self._file_search_fts_lane(
                            conn,
                            table=table,
                            match_query=match_query,
                            weights=weights,
                        ),
                    )
                if context_query:
                    for name, table, trigram in (
                        ("thread_words", "file_share_search_fts", False),
                        (
                            "thread_trigram",
                            "file_share_search_trigram",
                            True,
                        ),
                    ):
                        lexical_lanes[name] = (
                            "share",
                            self._file_search_fts_lane(
                                conn,
                                table=table,
                                match_query=self._file_search_match_query(
                                    context_query,
                                    column="thread_context",
                                    trigram=trigram,
                                ),
                                weights=(0.0, 2.0),
                            ),
                        )
            direct_dense = self._file_search_dense_lane(
                conn,
                table="file_contents",
                embedding_column="text_content_embedding_json",
                dimension_column="text_embedding_dimension",
                query_embedding=query_embedding,
            )
            parent_dense = self._file_search_dense_lane(
                conn,
                table="file_shares",
                embedding_column="parent_embedding_json",
                dimension_column="parent_embedding_dimension",
                query_embedding=context_query_embedding,
            )
        content_matches = {
            rowid
            for row_kind, rowids in lexical_lanes.values()
            if row_kind == "content"
            for rowid in rowids
        }
        share_matches = {
            rowid
            for row_kind, rowids in lexical_lanes.values()
            if row_kind == "share"
            for rowid in rowids
        }
        with self._connect() as conn:
            if not query:
                rows = conn.execute(
                    row_select + "ORDER BY fs.shared_at DESC LIMIT ?",
                    (*row_params, FILE_SEARCH_RETRIEVE_LIMIT),
                ).fetchall()
            elif direct_dense is None or (
                context_query and parent_dense is None
            ):
                # sqlite-vec is optional. Keep the bounded Python fallback for
                # old pods until the pinned extension has rolled out.
                rows = conn.execute(
                    row_select + "ORDER BY fs.shared_at DESC LIMIT ?",
                    (*row_params, FILE_SEARCH_MAX_INDEX_MATCHES),
                ).fetchall()
            else:
                content_candidates = set(content_matches)
                content_candidates.update(
                    rowid
                    for rowid, score in direct_dense.items()
                    if score >= FILE_SEARCH_MIN_SEMANTIC_SCORE
                )
                share_candidates = set(share_matches)
                if context_query:
                    share_candidates.update(
                        rowid
                        for rowid, score in parent_dense.items()
                        if score
                        >= FILE_SEARCH_MIN_PARENT_SEMANTIC_SCORE
                    )
                clauses = []
                candidate_params: list[Any] = []
                if content_candidates:
                    clauses.append(
                        "fc.rowid IN ("
                        + ",".join("?" for _ in content_candidates)
                        + ")"
                    )
                    candidate_params.extend(sorted(content_candidates))
                if share_candidates:
                    clauses.append(
                        "fs.rowid IN ("
                        + ",".join("?" for _ in share_candidates)
                        + ")"
                    )
                    candidate_params.extend(sorted(share_candidates))
                rows = (
                    conn.execute(
                        row_select
                        + "AND ("
                        + " OR ".join(clauses)
                        + ") ORDER BY fs.shared_at DESC LIMIT ?",
                        (
                            *row_params,
                            *candidate_params,
                            FILE_SEARCH_MAX_INDEX_MATCHES,
                        ),
                    ).fetchall()
                    if clauses
                    else []
                )
        precise_content = set(
            lexical_lanes.get("content_precise", ("content", []))[1]
        )
        precise_shares = set(
            lexical_lanes.get("upload_precise", ("share", []))[1]
        )
        def make_item(row: Any) -> dict[str, Any] | None:
            if uploader_filter and str(row["uploader_id"] or "") != uploader_filter:
                return None
            if file_types:
                mime_type = str(row["mime_type"] or "").lower()
                suffix = Path(str(row["file_name"] or "")).suffix.lower().lstrip(".")
                if not (
                    suffix in file_types
                    or mime_type in file_types
                    or ("images" in file_types and mime_type.startswith("image/"))
                    or (
                        "pdfs" in file_types
                        and (
                            suffix == "pdf"
                            or mime_type == "application/pdf"
                        )
                    )
                ):
                    return None
            message_file_count = int(row["message_file_count"] or 1)
            content_rowid = int(row["content_rowid"])
            share_rowid = int(row["share_rowid"])
            semantic = float((direct_dense or {}).get(content_rowid, 0.0))
            parent_semantic = float(
                (parent_dense or {}).get(share_rowid, 0.0)
            )
            if query_embedding and direct_dense is None:
                try:
                    vector = json.loads(
                        row["text_content_embedding_json"] or "[]"
                    )
                except json.JSONDecodeError:
                    vector = []
                if isinstance(vector, list):
                    semantic = self._file_search_cosine(
                        query_embedding, vector
                    )
            if context_query_embedding and parent_dense is None:
                try:
                    parent_vector = json.loads(
                        row["parent_embedding_json"] or "[]"
                    )
                except json.JSONDecodeError:
                    parent_vector = []
                if isinstance(parent_vector, list):
                    parent_semantic = self._file_search_cosine(
                        context_query_embedding, parent_vector
                    )
            return {
                "file_id": str(row["file_id"]),
                "name": row["file_name"],
                "mimetype": row["mime_type"],
                "size": row["byte_size"],
                "uploaded_at": row["uploaded_at"],
                "shared_at": row["shared_at"],
                "uploader_id": row["uploader_id"],
                "uploader_name": row["uploader_name"],
                "channel_id": row["conversation_id"],
                "channel_name": row["conversation_name"],
                "message_id": row["provider_message_id"],
                "thread_root_id": str(row["thread_root_id"]),
                "message_file_count": message_file_count,
                "permalink": row["permalink"],
                "upload_text": row["upload_text"],
                "thread_context": row["thread_context"],
                "caption_ocr": row["caption_ocr"],
                "processing_status": row["processing_status"],
                "rerank_score": 0.0,
                "semantic_score": semantic,
                "parent_semantic_score": parent_semantic,
                "has_parent_embedding": bool(row["parent_embedding_json"]),
                "_content_rowid": content_rowid,
                "_share_rowid": share_rowid,
                "_precise_match": (
                    content_rowid in precise_content
                    or share_rowid in precise_shares
                ),
                "_lexical_match": (
                    content_rowid in content_matches
                    or share_rowid in share_matches
                ),
            }

        scoped_items = [
            item for row in rows if (item := make_item(row)) is not None
        ]
        candidates = [
            item
            for item in scoped_items
            if not query
            or item["_lexical_match"]
            or item["semantic_score"] >= FILE_SEARCH_MIN_SEMANTIC_SCORE
            or (
                context_query
                and item["parent_semantic_score"]
                >= FILE_SEARCH_MIN_PARENT_SEMANTIC_SCORE
            )
        ]
        if not query:
            retrieved = sorted(
                candidates,
                key=lambda item: str(item.get("shared_at") or ""),
                reverse=True,
            )[:FILE_SEARCH_RETRIEVE_LIMIT]
            reranked = retrieved[:FILE_SEARCH_RERANK_LIMIT]
            rerank_count = len(reranked)
        else:
            by_content: dict[int, list[dict[str, Any]]] = {}
            by_share: dict[int, list[dict[str, Any]]] = {}
            for item in candidates:
                by_content.setdefault(item["_content_rowid"], []).append(item)
                by_share.setdefault(item["_share_rowid"], []).append(item)

            lanes: list[tuple[list[dict[str, Any]], float]] = []
            for name, (row_kind, rowids) in lexical_lanes.items():
                if name.endswith("_precise"):
                    continue
                mapping = by_content if row_kind == "content" else by_share
                lane = [
                    item
                    for rowid in rowids
                    for item in sorted(
                        mapping.get(rowid, []),
                        key=lambda value: str(value.get("shared_at") or ""),
                        reverse=True,
                    )
                ][:FILE_SEARCH_RETRIEVE_LIMIT]
                if lane:
                    lanes.append(
                        (
                            lane,
                            FILE_SEARCH_CONTEXT_LANE_WEIGHT
                            if name.startswith("thread_")
                            else FILE_SEARCH_DIRECT_LEXICAL_LANE_WEIGHT,
                        )
                    )
            semantic_lane = sorted(
                (
                    item
                    for item in candidates
                    if item["semantic_score"]
                    >= FILE_SEARCH_MIN_SEMANTIC_SCORE
                ),
                key=lambda item: (
                    item["semantic_score"],
                    str(item.get("shared_at") or ""),
                ),
                reverse=True,
            )[:FILE_SEARCH_RETRIEVE_LIMIT]
            parent_lane = sorted(
                (
                    item
                    for item in candidates
                    if context_query
                    and item["parent_semantic_score"]
                    >= FILE_SEARCH_MIN_PARENT_SEMANTIC_SCORE
                ),
                key=lambda item: (
                    item["parent_semantic_score"],
                    str(item.get("shared_at") or ""),
                ),
                reverse=True,
            )[:FILE_SEARCH_RETRIEVE_LIMIT]
            if semantic_lane:
                lanes.append((semantic_lane, 1.0))
            if parent_lane:
                lanes.append(
                    (parent_lane, FILE_SEARCH_CONTEXT_DENSE_LANE_WEIGHT)
                )

            group_scores: dict[tuple[str, str], float] = {}
            item_scores: dict[int, float] = {}
            for lane, lane_weight in lanes:
                seen_groups: set[tuple[str, str]] = set()
                for rank, item in enumerate(lane, start=1):
                    score = lane_weight / (FILE_SEARCH_RRF_K + rank)
                    item_scores[id(item)] = (
                        item_scores.get(id(item), 0.0) + score
                    )
                    group = (
                        str(item["channel_id"]),
                        str(item["thread_root_id"]),
                    )
                    if group in seen_groups:
                        continue
                    seen_groups.add(group)
                    group_scores[group] = (
                        group_scores.get(group, 0.0) + score
                    )

            admitted = [
                item
                for item in candidates
                if item["_precise_match"]
                or item["semantic_score"]
                >= FILE_SEARCH_MIN_SEMANTIC_SCORE
                or (
                    context_query
                    and item["parent_semantic_score"]
                    >= FILE_SEARCH_MIN_PARENT_SEMANTIC_SCORE
                )
            ]
            admitted_groups = {
                (
                    str(item["channel_id"]),
                    str(item["thread_root_id"]),
                )
                for item in admitted
            }
            latest_by_group: dict[tuple[str, str], str] = {}
            for item in candidates:
                group = (
                    str(item["channel_id"]),
                    str(item["thread_root_id"]),
                )
                latest_by_group[group] = max(
                    latest_by_group.get(group, ""),
                    str(item.get("shared_at") or ""),
                )
            selected_groups = sorted(
                admitted_groups,
                key=lambda group: (
                    group_scores.get(group, 0.0),
                    latest_by_group.get(group, ""),
                ),
                reverse=True,
            )[:FILE_SEARCH_RERANK_LIMIT]
            if selected_groups:
                group_clauses = []
                group_params: list[Any] = []
                for conversation_id, thread_root_id in selected_groups:
                    group_clauses.append(
                        "(fs.conversation_id=? AND "
                        "COALESCE(m.parent_message_id, "
                        "fs.provider_message_id)=?)"
                    )
                    group_params.extend((conversation_id, thread_root_id))
                with self._connect() as conn:
                    expanded_rows = conn.execute(
                        row_select
                        + "AND ("
                        + " OR ".join(group_clauses)
                        + ") ORDER BY fs.shared_at DESC LIMIT ?",
                        (
                            *row_params,
                            *group_params,
                            FILE_SEARCH_MAX_INDEX_MATCHES,
                        ),
                    ).fetchall()
                scoped_items = [
                    item
                    for row in expanded_rows
                    if (item := make_item(row)) is not None
                ]
            group_order = {
                group: rank for rank, group in enumerate(selected_groups)
            }
            retrieved = sorted(
                candidates,
                key=lambda item: (
                    group_scores.get(
                        (
                            str(item["channel_id"]),
                            str(item["thread_root_id"]),
                        ),
                        0.0,
                    ),
                    str(item.get("shared_at") or ""),
                ),
                reverse=True,
            )[:FILE_SEARCH_RETRIEVE_LIMIT]
            reranked = [
                item
                for item in scoped_items
                if (
                    str(item["channel_id"]),
                    str(item["thread_root_id"]),
                )
                in admitted_groups
                and (
                    str(item["channel_id"]),
                    str(item["thread_root_id"]),
                )
                in group_order
            ]
            for item in reranked:
                group = (
                    str(item["channel_id"]),
                    str(item["thread_root_id"]),
                )
                item["rerank_score"] = (
                    group_scores.get(group, 0.0) * 100.0
                    + item_scores.get(id(item), 0.0)
                )
            reranked.sort(
                key=lambda item: (
                    -group_order[
                        (
                            str(item["channel_id"]),
                            str(item["thread_root_id"]),
                        )
                    ],
                    str(item.get("shared_at") or ""),
                    item["rerank_score"],
                ),
                reverse=True,
            )
            rerank_count = len(selected_groups)
        inspected = 0
        for item in reranked:
            item["image_inspected"] = False
        inspectable = [
            item
            for item in reranked
            if str(item.get("mimetype") or "").startswith("image/")
        ][:FILE_SEARCH_IMAGE_INSPECT_LIMIT]
        if access_token and inspectable:
            with ThreadPoolExecutor(max_workers=len(inspectable)) as pool:
                futures = {
                    pool.submit(
                        inspect_slack_image,
                        str(item["file_id"]),
                        access_token,
                        query,
                    ): item
                    for item in inspectable
                }
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        inspection = future.result()
                    except Exception:
                        continue
                    item["image_inspected"] = True
                    item["image_inspection"] = inspection
                    inspected += 1
        limit = max(1, min(int(request.get("limit") or 3), 20))
        output = []
        for item in reranked[:limit]:
            evidence = []
            for label, key in (
                ("upload", "upload_text"),
                ("thread", "thread_context"),
                ("image", "caption_ocr"),
                ("inspection", "image_inspection"),
            ):
                value = str(item.get(key) or "").strip()
                if value and query_tokens & self._file_search_tokens(value):
                    evidence.append(f"{label}: {value[:160]}")
            if query and not evidence:
                evidence.append("thread: matched discussion")
            description = str(
                item.get("image_inspection") or item.get("caption_ocr") or ""
            ).strip()
            output.append(
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "upload_text",
                        "thread_context",
                        "caption_ocr",
                        "message_file_count",
                        "semantic_score",
                        "parent_semantic_score",
                        "has_parent_embedding",
                        "image_inspection",
                        "thread_root_id",
                        "_content_rowid",
                        "_share_rowid",
                        "_precise_match",
                        "_lexical_match",
                    }
                }
                | {
                    "description": description[:2_000] or None,
                    "why_matched": " | ".join(evidence)[:320]
                    or (
                        "recent file in selected scope"
                        if not query
                        else "metadata match"
                    ),
                }
            )
        return {
            "status": "complete",
            "coverage_complete": coverage_complete,
            "retrieve_count": len(retrieved),
            "rerank_count": rerank_count,
            "inspected_image_count": inspected,
            "files": output,
            "store_generation": self.store_generation,
        }

    def file_graph_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Export a bounded, ACL-filtered Episode batch without persisting it centrally."""
        request = dict(request)
        request.pop("provider_access_token", None)
        if str(request.get("project_id") or "") != self.project_id:
            raise ValueError("file graph project mismatch")
        if str(request.get("store_generation") or "") != self.store_generation:
            raise ValueError("file graph store generation mismatch")
        provider = str(request.get("provider") or "")
        workspace_id = str(request.get("workspace_id") or "")
        conversations = self._allowed_file_sources(
            request.get("allowed_source_ids"),
            provider=provider,
            workspace_id=workspace_id,
        )
        if not provider or not workspace_id or not conversations:
            return {
                "status": "complete",
                "episodes": [],
                "store_generation": self.store_generation,
            }
        limit = max(1, min(int(request.get("limit") or 3), 10))
        window_started_at = str(request.get("window_started_at") or "")
        placeholders = ",".join("?" for _ in conversations)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fc.file_id, fc.content_version, fc.file_name, "
                "fc.mime_type, fc.permalink, fc.caption_ocr, "
                "fs.conversation_id, fs.provider_message_id, fs.uploader_id, "
                "fs.upload_text, fs.thread_context, fs.context_version, "
                "fs.shared_at, c.title AS conversation_name "
                "FROM file_contents fc JOIN file_shares fs ON "
                "fc.project_id=fs.project_id AND fc.provider=fs.provider AND "
                "fc.workspace_id=fs.workspace_id AND fc.file_id=fs.file_id "
                "LEFT JOIN conversations c ON c.project_id=fs.project_id AND "
                "c.provider=fs.provider AND c.workspace_id=fs.workspace_id AND "
                "c.conversation_id=fs.conversation_id "
                "LEFT JOIN file_graph_ingest_state gs ON "
                "gs.project_id=fs.project_id AND gs.provider=fs.provider AND "
                "gs.workspace_id=fs.workspace_id AND gs.file_id=fs.file_id AND "
                "gs.conversation_id=fs.conversation_id AND "
                "gs.provider_message_id=fs.provider_message_id "
                "WHERE fc.project_id=? AND fc.provider=? AND fc.workspace_id=? "
                "AND fs.conversation_id IN ("
                + placeholders
                + ") AND fc.processing_status NOT IN ('deleted', 'metadata_only') "
                "AND fs.tombstoned_at IS NULL "
                + ("AND fs.shared_at>=? " if window_started_at else "")
                + "AND (gs.file_id IS NULL OR "
                "gs.content_version<fc.content_version OR "
                "gs.context_version<fs.context_version) "
                "ORDER BY fs.shared_at, fs.conversation_id, "
                "fs.provider_message_id, fs.file_id LIMIT ?",
                (
                    self.project_id,
                    provider,
                    workspace_id,
                    *conversations,
                    *([window_started_at] if window_started_at else []),
                    limit,
                ),
            ).fetchall()
        episodes = []
        for row in rows:
            body_parts = [
                "Slack file-share evidence.",
                f"File name: {str(row['file_name'] or '')[:500]}",
                f"MIME type: {str(row['mime_type'] or '')[:200]}",
                f"Uploader Slack ID: {str(row['uploader_id'] or '')[:200]}",
                "Upload message:\n" + str(row["upload_text"] or "")[:4_000],
                "Thread context:\n" + str(row["thread_context"] or "")[:8_000],
                "File description/OCR:\n" + str(row["caption_ocr"] or "")[:8_000],
            ]
            file_id = str(row["file_id"])
            conversation_id = str(row["conversation_id"])
            message_id = str(row["provider_message_id"])
            episodes.append(
                {
                    "episode_id": ":".join(
                        (
                            provider,
                            workspace_id,
                            file_id,
                            conversation_id,
                            message_id,
                        )
                    ),
                    "name": f"slack-file:{workspace_id}:{file_id}:{conversation_id}:{message_id}",
                    "body": "\n\n".join(body_parts)[:20_000],
                    "source_description": (
                        f"{provider}:{workspace_id}:{conversation_id} #"
                        + str(row["conversation_name"] or conversation_id)[:500]
                        + " file"
                    ),
                    "reference_time": str(
                        row["shared_at"] or datetime.now(timezone.utc).isoformat()
                    ),
                    "source_id": f"{provider}:{workspace_id}:{conversation_id}",
                    "source_metadata": {
                        "file_id": file_id,
                        "uploader_id": row["uploader_id"],
                        "permalink": row["permalink"],
                        "mime_type": row["mime_type"],
                        "content_version": int(row["content_version"]),
                        "context_version": int(row["context_version"]),
                    },
                }
            )
        return {
            "status": "continuation" if len(episodes) == limit else "complete",
            "episodes": episodes,
            "store_generation": self.store_generation,
        }

    def ack_file_graph_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Advance only the independent file-graph watermark after graph commit."""
        if str(request.get("project_id") or "") != self.project_id:
            raise ValueError("file graph project mismatch")
        if str(request.get("store_generation") or "") != self.store_generation:
            raise ValueError("file graph store generation mismatch")
        provider = str(request.get("provider") or "")
        workspace_id = str(request.get("workspace_id") or "")
        acknowledgements = request.get("acknowledgements")
        if (
            not provider
            or not workspace_id
            or not isinstance(acknowledgements, list)
            or len(acknowledgements) > 10
        ):
            raise ValueError("file graph acknowledgements are invalid")
        now = datetime.now(timezone.utc).isoformat()
        applied = 0
        with self._writer_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in acknowledgements:
                if not isinstance(item, dict):
                    raise ValueError("file graph acknowledgement is invalid")
                required = (
                    str(item.get("file_id") or ""),
                    str(item.get("conversation_id") or ""),
                    str(item.get("provider_message_id") or ""),
                )
                versions = (
                    item.get("content_version"),
                    item.get("context_version"),
                )
                if (
                    not all(required)
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in versions
                    )
                ):
                    raise ValueError("file graph acknowledgement is invalid")
                conn.execute(
                    "INSERT INTO file_graph_ingest_state(project_id, provider, "
                    "workspace_id, file_id, conversation_id, provider_message_id, "
                    "content_version, context_version, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(project_id, provider, workspace_id, file_id, "
                    "conversation_id, provider_message_id) DO UPDATE SET "
                    "content_version=MAX(file_graph_ingest_state.content_version, "
                    "excluded.content_version), "
                    "context_version=MAX(file_graph_ingest_state.context_version, "
                    "excluded.context_version), ingested_at=excluded.ingested_at",
                    (
                        self.project_id,
                        provider,
                        workspace_id,
                        *required,
                        int(versions[0]),
                        int(versions[1]),
                        now,
                    ),
                )
                applied += 1
            conn.commit()
        return {
            "status": "complete",
            "acknowledged": applied,
            "store_generation": self.store_generation,
        }

    def _file_thread_context(self, conn: Any, row: Any) -> str | None:
        root = str(row["parent_message_id"] or row["provider_message_id"])
        context_rows = conn.execute(
            "SELECT text FROM messages WHERE project_id=? AND provider=? "
            "AND workspace_id=? AND conversation_id=? AND deleted_at IS NULL "
            "AND (provider_message_id=? OR parent_message_id=?) "
            "ORDER BY occurred_at LIMIT 20",
            (
                self.project_id,
                str(row["provider"]),
                str(row["workspace_id"]),
                str(row["conversation_id"]),
                root,
                root,
            ),
        ).fetchall()
        text = "\n".join(
            str(item["text"]).strip()
            for item in context_rows
            if str(item["text"] or "").strip()
        )
        return text[:8_000] or None

    def _apply_file_command(
        self,
        conn: Any,
        request: dict[str, Any],
        applied_at: str,
    ) -> str:
        operation = str(request.get("operation") or "")
        if operation not in {
            "upsert_share",
            "remove_share",
            "refresh_file",
            "delete_file",
        }:
            raise ValueError("unsupported file index operation")
        provider = str(request.get("provider") or "")
        workspace_id = str(request.get("workspace_id") or "")
        file_id = str(request.get("file_id") or "")
        source_version = request.get("source_version")
        if (
            not provider
            or not workspace_id
            or not file_id
            or isinstance(source_version, bool)
            or not isinstance(source_version, int)
            or source_version < 0
        ):
            raise ValueError("file index scope and integer version are required")
        occurred_at = str(request.get("occurred_at") or applied_at)
        file_scope = (self.project_id, provider, workspace_id, file_id)

        if operation == "delete_file":
            current = conn.execute(
                "SELECT source_version FROM file_tombstones WHERE project_id=? "
                "AND provider=? AND workspace_id=? AND file_id=?",
                file_scope,
            ).fetchone()
            if current is not None and int(current["source_version"]) > source_version:
                return "stale"
            conn.execute(
                "INSERT INTO file_tombstones(project_id, provider, workspace_id, "
                "file_id, source_version, tombstoned_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(project_id, provider, workspace_id, file_id) DO UPDATE SET "
                "source_version=excluded.source_version, "
                "tombstoned_at=excluded.tombstoned_at "
                "WHERE excluded.source_version >= file_tombstones.source_version",
                (*file_scope, source_version, occurred_at),
            )
            conn.execute(
                "UPDATE file_contents SET processing_status='deleted', updated_at=? "
                "WHERE project_id=? AND provider=? AND workspace_id=? AND file_id=? "
                "AND content_version<=?",
                (applied_at, *file_scope, source_version),
            )
            conn.execute(
                "UPDATE file_shares SET tombstone_version=?, tombstoned_at=?, "
                "updated_at=? WHERE project_id=? AND provider=? AND workspace_id=? "
                "AND file_id=? AND (tombstone_version IS NULL OR tombstone_version<=?)",
                (source_version, occurred_at, applied_at, *file_scope, source_version),
            )
            return "applied"

        if operation == "refresh_file":
            file_tombstone = conn.execute(
                "SELECT source_version FROM file_tombstones WHERE project_id=? "
                "AND provider=? AND workspace_id=? AND file_id=?",
                file_scope,
            ).fetchone()
            if (
                file_tombstone is not None
                and int(file_tombstone["source_version"]) >= source_version
            ):
                return "stale"
            content = conn.execute(
                "SELECT content_version FROM file_contents WHERE project_id=? "
                "AND provider=? AND workspace_id=? AND file_id=?",
                file_scope,
            ).fetchone()
            if content is None:
                return "stale"
            if int(content["content_version"]) > source_version:
                return "stale"
            conn.execute(
                "UPDATE file_contents SET content_version=?, "
                "processing_status='metadata_only', last_error_code=NULL, "
                "processing_attempts=0, updated_at=? "
                "WHERE project_id=? AND provider=? "
                "AND workspace_id=? AND file_id=?",
                (source_version, applied_at, *file_scope),
            )
            return "applied"

        conversation_id = str(request.get("conversation_id") or "")
        message_id = str(request.get("provider_message_id") or "")
        if not conversation_id or not message_id:
            raise ValueError("file index share identifiers are required")
        share_scope = (*file_scope, conversation_id, message_id)
        file_tombstone = conn.execute(
            "SELECT source_version FROM file_tombstones WHERE project_id=? "
            "AND provider=? AND workspace_id=? AND file_id=?",
            file_scope,
        ).fetchone()
        if (
            file_tombstone is not None
            and int(file_tombstone["source_version"]) >= source_version
        ):
            return "stale"

        if operation == "remove_share":
            if message_id == "*":
                conn.execute(
                    "UPDATE file_shares SET tombstone_version=?, tombstoned_at=?, "
                    "updated_at=? WHERE project_id=? AND provider=? "
                    "AND workspace_id=? AND file_id=? AND conversation_id=? "
                    "AND (tombstone_version IS NULL OR tombstone_version<=?)",
                    (
                        source_version,
                        occurred_at,
                        applied_at,
                        *file_scope,
                        conversation_id,
                        source_version,
                    ),
                )
                return "applied"
            conn.execute(
                "INSERT INTO file_shares(project_id, provider, workspace_id, file_id, "
                "conversation_id, provider_message_id, context_version, "
                "tombstone_version, tombstoned_at, inserted_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(project_id, provider, workspace_id, file_id, "
                "conversation_id, provider_message_id) DO UPDATE SET "
                "tombstone_version=excluded.tombstone_version, "
                "tombstoned_at=excluded.tombstoned_at, updated_at=excluded.updated_at "
                "WHERE file_shares.tombstone_version IS NULL OR "
                "excluded.tombstone_version >= file_shares.tombstone_version",
                (
                    *share_scope,
                    source_version,
                    source_version,
                    occurred_at,
                    applied_at,
                    applied_at,
                ),
            )
            return "applied"

        content_version = request.get("content_version", source_version)
        context_version = request.get("context_version", source_version)
        if (
            isinstance(content_version, bool)
            or not isinstance(content_version, int)
            or isinstance(context_version, bool)
            or not isinstance(context_version, int)
        ):
            raise ValueError("file content and context versions must be integers")

        existing_share = conn.execute(
            "SELECT tombstone_version, upload_text, thread_context, "
            "parent_embedding_json, parent_embedding_model, "
            "parent_embedding_dimension, parent_embedding_attempts, "
            "parent_embedding_error FROM file_shares WHERE project_id=? "
            "AND provider=? AND workspace_id=? AND file_id=? AND conversation_id=? "
            "AND provider_message_id=?",
            share_scope,
        ).fetchone()
        if (
            existing_share is not None
            and existing_share["tombstone_version"] is not None
            and int(existing_share["tombstone_version"]) >= context_version
        ):
            return "stale"

        processing_status = str(
            request.get("processing_status") or "metadata_only"
        )
        if processing_status not in {
            "indexed",
            "metadata_only",
            "unsupported",
            "unavailable",
            "deleted",
        }:
            raise ValueError("invalid file processing status")
        text_embedding = request.get("text_content_embedding")
        image_embedding = request.get("image_embedding")
        parent_embedding = request.get("parent_embedding")
        upload_text = str(request.get("upload_text") or "") or None
        thread_context = str(request.get("thread_context") or "") or None
        parent_embedding_json = (
            json.dumps(parent_embedding, separators=(",", ":"))
            if parent_embedding is not None
            else None
        )
        parent_embedding_model = (
            str(request.get("parent_embedding_model") or "") or None
        )
        parent_embedding_dimension = (
            int(request["parent_embedding_dimension"])
            if request.get("parent_embedding_dimension") is not None
            else None
        )
        parent_embedding_attempts = 0
        parent_embedding_error = None
        if (
            existing_share is not None
            and parent_embedding is None
            and existing_share["upload_text"] == upload_text
            and existing_share["thread_context"] == thread_context
        ):
            parent_embedding_json = existing_share["parent_embedding_json"]
            parent_embedding_model = existing_share["parent_embedding_model"]
            parent_embedding_dimension = existing_share[
                "parent_embedding_dimension"
            ]
            parent_embedding_attempts = int(
                existing_share["parent_embedding_attempts"] or 0
            )
            parent_embedding_error = existing_share["parent_embedding_error"]
        existing_content = conn.execute(
            "SELECT content_version FROM file_contents WHERE project_id=? "
            "AND provider=? AND workspace_id=? AND file_id=?",
            file_scope,
        ).fetchone()
        content_update = existing_content is None or any(
            key in request
            for key in (
                "processing_status",
                "content_sha256",
                "caption_ocr",
                "text_content_embedding",
                "image_embedding",
                "last_error_code",
            )
        )
        if content_update:
            conn.execute(
            "INSERT INTO file_contents(project_id, provider, workspace_id, file_id, "
            "content_version, content_sha256, file_name, mime_type, byte_size, "
            "uploaded_at, permalink, processing_status, caption_ocr, "
            "text_content_embedding_json, image_embedding_json, caption_model, "
            "caption_prompt_version, text_embedding_model, text_embedding_dimension, "
            "image_embedding_model, image_embedding_dimension, last_error_code, "
            "inserted_at, updated_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, provider, workspace_id, file_id) DO UPDATE SET "
            "content_version=excluded.content_version, "
            "content_sha256=excluded.content_sha256, file_name=excluded.file_name, "
            "mime_type=excluded.mime_type, byte_size=excluded.byte_size, "
            "uploaded_at=excluded.uploaded_at, permalink=excluded.permalink, "
            "processing_status=excluded.processing_status, "
            "caption_ocr=excluded.caption_ocr, "
            "text_content_embedding_json=excluded.text_content_embedding_json, "
            "image_embedding_json=excluded.image_embedding_json, "
            "caption_model=excluded.caption_model, "
            "caption_prompt_version=excluded.caption_prompt_version, "
            "text_embedding_model=excluded.text_embedding_model, "
            "text_embedding_dimension=excluded.text_embedding_dimension, "
            "image_embedding_model=excluded.image_embedding_model, "
            "image_embedding_dimension=excluded.image_embedding_dimension, "
            "last_error_code=excluded.last_error_code, updated_at=excluded.updated_at "
            "WHERE excluded.content_version >= file_contents.content_version",
                (
                    *file_scope,
                    content_version,
                    str(request.get("content_sha256") or "") or None,
                    str(request.get("file_name") or "") or None,
                    str(request.get("mime_type") or "") or None,
                    int(request["byte_size"])
                    if request.get("byte_size") is not None
                    else None,
                    str(request.get("uploaded_at") or "") or None,
                    str(request.get("permalink") or "") or None,
                    processing_status,
                    str(request.get("caption_ocr") or "") or None,
                    json.dumps(text_embedding, separators=(",", ":"))
                    if text_embedding is not None
                    else None,
                    json.dumps(image_embedding, separators=(",", ":"))
                    if image_embedding is not None
                    else None,
                    str(request.get("caption_model") or "") or None,
                    str(request.get("caption_prompt_version") or "") or None,
                    str(request.get("text_embedding_model") or "") or None,
                    int(request["text_embedding_dimension"])
                    if request.get("text_embedding_dimension") is not None
                    else None,
                    str(request.get("image_embedding_model") or "") or None,
                    int(request["image_embedding_dimension"])
                    if request.get("image_embedding_dimension") is not None
                    else None,
                    str(request.get("last_error_code") or "") or None,
                    applied_at,
                    applied_at,
                ),
            )
        conn.execute(
            "INSERT INTO file_shares(project_id, provider, workspace_id, file_id, "
            "conversation_id, provider_message_id, uploader_id, upload_text, "
            "thread_context, context_version, shared_at, parent_embedding_json, "
            "parent_embedding_model, parent_embedding_dimension, "
            "parent_embedding_attempts, parent_embedding_error, tombstone_version, "
            "tombstoned_at, inserted_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "NULL, NULL, ?, ?) "
            "ON CONFLICT(project_id, provider, workspace_id, file_id, "
            "conversation_id, provider_message_id) DO UPDATE SET "
            "uploader_id=excluded.uploader_id, "
            "parent_embedding_json=excluded.parent_embedding_json, "
            "parent_embedding_model=excluded.parent_embedding_model, "
            "parent_embedding_dimension=excluded.parent_embedding_dimension, "
            "parent_embedding_attempts=excluded.parent_embedding_attempts, "
            "parent_embedding_error=excluded.parent_embedding_error, "
            "upload_text=excluded.upload_text, "
            "thread_context=excluded.thread_context, "
            "context_version=excluded.context_version, shared_at=excluded.shared_at, "
            "tombstone_version=NULL, tombstoned_at=NULL, updated_at=excluded.updated_at "
            "WHERE excluded.context_version >= file_shares.context_version",
            (
                *share_scope,
                str(request.get("uploader_id") or "") or None,
                upload_text,
                thread_context,
                context_version,
                str(request.get("shared_at") or "") or None,
                parent_embedding_json,
                parent_embedding_model,
                parent_embedding_dimension,
                parent_embedding_attempts,
                parent_embedding_error,
                applied_at,
                applied_at,
            ),
        )
        return "applied"

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

    @staticmethod
    def _message_search_match_query(segments: list[str]) -> str:
        return " OR ".join(
            f'"{segment.replace(chr(34), chr(34) * 2)}"'
            for segment in list(dict.fromkeys(segments))[
                :MESSAGE_SEARCH_MAX_QUERY_TERMS
            ]
            if segment
        )

    def _message_search_candidate_rowids(
        self,
        conn: Any,
        *,
        query: str,
        start: str,
        end: str,
        providers: set[str],
        workspaces: set[str],
        conversations: set[str],
        allowed: set[tuple[str, str, str]] | None,
    ) -> list[int]:
        """Retrieve a relevance-bounded candidate set inside SQLite."""
        segments = search_segments(query)
        ranked: dict[int, float] = {}

        def collect_fts(table: str, match_query: str) -> None:
            if not match_query:
                return
            sql = (
                f"SELECT m.rowid AS message_rowid, bm25({table}) AS search_rank "
                f"FROM {table} JOIN messages m ON m.rowid={table}.rowid "
                f"WHERE {table} MATCH ? AND m.project_id=? "
                "AND m.occurred_at>=? AND m.occurred_at<=? "
                "AND m.deleted_at IS NULL"
            )
            params: list[Any] = [match_query, self.project_id, start, end]
            sql, params = self._query_filters(
                sql,
                params,
                providers=providers,
                workspaces=workspaces,
                conversations=conversations,
                prefix="m.",
                allowed=allowed,
            )
            sql += f" ORDER BY bm25({table}), m.occurred_at DESC LIMIT ?"
            params.append(MESSAGE_SEARCH_CANDIDATE_LIMIT)
            for row in conn.execute(sql, params).fetchall():
                rowid = int(row["message_rowid"])
                rank = float(row["search_rank"] or 0.0)
                ranked[rowid] = min(ranked.get(rowid, rank), rank)

        collect_fts(
            "message_search_fts",
            self._message_search_match_query(
                [*segments, *search_index_terms(query)]
            ),
        )
        collect_fts(
            "message_search_trigram",
            self._message_search_match_query(
                [segment for segment in segments if len(segment) >= 3]
            ),
        )

        # Single-character substring search cannot be represented by either
        # trigrams or the language-neutral bigram terms.
        single_character_segments = list(
            dict.fromkeys(segment for segment in segments if len(segment) < 2)
        )
        if single_character_segments:
            sql = (
                "SELECT m.rowid AS message_rowid FROM messages m "
                "WHERE m.project_id=? AND m.occurred_at>=? AND m.occurred_at<=? "
                "AND m.deleted_at IS NULL AND ("
                + " OR ".join(
                    "instr(ringo_search_normalize(m.text), ?) > 0"
                    for _segment in single_character_segments
                )
                + ")"
            )
            params = [self.project_id, start, end, *single_character_segments]
            sql, params = self._query_filters(
                sql,
                params,
                providers=providers,
                workspaces=workspaces,
                conversations=conversations,
                prefix="m.",
                allowed=allowed,
            )
            sql += " ORDER BY m.occurred_at DESC LIMIT ?"
            params.append(MESSAGE_SEARCH_CANDIDATE_LIMIT)
            for row in conn.execute(sql, params).fetchall():
                ranked.setdefault(int(row["message_rowid"]), 1_000.0)

        return [
            rowid
            for rowid, _rank in sorted(
                ranked.items(),
                key=lambda item: (item[1], item[0]),
            )[:MESSAGE_SEARCH_CANDIDATE_LIMIT]
        ]

    def _message_search_context_rows(
        self,
        conn: Any,
        *,
        candidate_rowids: list[int],
        start: str,
        end: str,
    ) -> list[Any]:
        """Fetch bounded whole-thread context for the indexed candidates."""
        if not candidate_rowids:
            return []
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS message_search_candidates("
            "message_rowid INTEGER PRIMARY KEY, priority INTEGER NOT NULL)"
        )
        conn.execute("DELETE FROM message_search_candidates")
        conn.executemany(
            "INSERT INTO message_search_candidates(message_rowid, priority) "
            "VALUES (?, ?)",
            [(rowid, priority) for priority, rowid in enumerate(candidate_rowids)],
        )
        sql = (
            "WITH candidate_roots AS (SELECT m.project_id, m.provider, "
            "m.workspace_id, m.conversation_id, COALESCE("
            "m.parent_message_id, m.provider_message_id) AS root_id, "
            "MIN(candidate.priority) AS priority "
            "FROM message_search_candidates candidate "
            "JOIN messages m ON m.rowid=candidate.message_rowid "
            "GROUP BY m.project_id, m.provider, m.workspace_id, "
            "m.conversation_id, COALESCE("
            "m.parent_message_id, m.provider_message_id)), "
            "scoped AS (SELECT m.*, c.title AS conversation_title, "
            "i.display_name AS sender_display_name, root.priority, "
            "ROW_NUMBER() OVER (PARTITION BY m.provider, m.workspace_id, "
            "m.conversation_id, root.root_id ORDER BY m.occurred_at) "
            "AS root_row_number FROM messages m JOIN candidate_roots root "
            "ON root.project_id=m.project_id AND root.provider=m.provider "
            "AND root.workspace_id=m.workspace_id "
            "AND root.conversation_id=m.conversation_id "
            "AND root.root_id=COALESCE("
            "m.parent_message_id, m.provider_message_id) "
            "LEFT JOIN conversations c ON c.project_id=m.project_id AND "
            "c.provider=m.provider AND c.workspace_id=m.workspace_id AND "
            "c.conversation_id=m.conversation_id LEFT JOIN identities i ON "
            "i.project_id=m.project_id AND i.provider=m.provider AND "
            "i.workspace_id=m.workspace_id AND i.external_user_id=m.sender_id "
            "WHERE m.project_id=? AND m.occurred_at>=? AND m.occurred_at<=? "
            "AND m.deleted_at IS NULL) SELECT * FROM scoped "
            "WHERE root_row_number<=? ORDER BY priority, occurred_at LIMIT ?"
        )
        return conn.execute(
            sql,
            (
                self.project_id,
                start,
                end,
                MESSAGE_SEARCH_MAX_ROWS_PER_ROOT,
                MESSAGE_SEARCH_CONTEXT_ROW_LIMIT,
            ),
        ).fetchall()

    def query(self, request: dict[str, Any]) -> dict[str, Any]:
        """Bounded project-local read after IE has resolved current ACL."""
        operation = str(request.get("operation") or "")
        if operation not in {
            "recent_activity",
            "fetch_history",
            "fetch_snapshot",
            "ingest_window",
            "search",
        }:
            raise ValueError("unsupported message store query operation")
        search_query = str(request.get("query") or "").strip()
        if operation == "search" and not search_query:
            raise ValueError("search query is required")
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
            coverage_complete = True
            coverage_reason: str | None = None

            def mark_incomplete(reason: str) -> None:
                nonlocal coverage_complete, coverage_reason
                coverage_complete = False
                coverage_reason = coverage_reason or reason

            unresolved_gap = conn.execute(
                "SELECT 1 FROM delivery_gaps WHERE repaired_at IS NULL LIMIT 1"
            ).fetchone()
            if unresolved_gap is not None:
                if operation != "search":
                    return {
                        "messages": [],
                        "coverage_complete": False,
                        "reason": "delivery_gap",
                    }
                mark_incomplete("delivery_gap")
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
                    if operation != "search":
                        return {
                            "messages": [],
                            "coverage_complete": False,
                            "reason": "coverage_missing",
                        }
                    mark_incomplete("coverage_missing")
                if allowed is None and conversations:
                    covered_ids = {item[2] for item in covered_sources}
                    if covered_ids != conversations:
                        if operation != "search":
                            return {
                                "messages": [],
                                "coverage_complete": False,
                                "reason": "coverage_missing",
                            }
                        mark_incomplete("coverage_missing")
                if not coverage_rows:
                    if operation != "search":
                        return {
                            "messages": [],
                            "coverage_complete": False,
                            "reason": "coverage_missing",
                        }
                    mark_incomplete("coverage_missing")
                if any(
                    row["state"] != "COLLECTING"
                    or not row["contiguous_since"]
                    or str(row["contiguous_since"]) > start
                    for row in coverage_rows
                ):
                    if operation != "search":
                        return {
                            "messages": [],
                            "coverage_complete": False,
                            "reason": "coverage_incomplete",
                        }
                    mark_incomplete("coverage_incomplete")

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
                "m.provider_message_id DESC"
            )
            if operation == "search":
                if not self.message_search_index_available:
                    return {
                        "hits": [],
                        "coverage_complete": False,
                        "reason": "search_index_unavailable",
                    }
                candidate_rowids = self._message_search_candidate_rowids(
                    conn,
                    query=search_query,
                    start=start,
                    end=end,
                    providers=providers,
                    workspaces=workspaces,
                    conversations=conversations,
                    allowed=allowed,
                )
                rows = self._message_search_context_rows(
                    conn,
                    candidate_rowids=candidate_rowids,
                    start=start,
                    end=end,
                )
                floors = [
                    str(row["contiguous_since"])
                    for row in coverage_rows
                    if row["contiguous_since"]
                ]
                result = {
                    "hits": _rank_message_rows(
                        rows,
                        query=search_query,
                        limit=limit,
                    ),
                    "coverage_complete": coverage_complete,
                    "last_sequence": max(
                        (
                            int(row["last_sequence"] or 0)
                            for row in conn.execute(
                                "SELECT last_sequence FROM coverage WHERE project_id=?",
                                (self.project_id,),
                            ).fetchall()
                        ),
                        default=0,
                    ),
                }
                if floors:
                    result["covered_since"] = max(floors)
                if coverage_reason:
                    result["reason"] = coverage_reason
                return result
            sql += " LIMIT ?"
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
            message_count = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            content_embeddings = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN text_content_embedding_json IS NOT NULL "
                "THEN 1 ELSE 0 END) AS embedded, "
                "SUM(CASE WHEN processing_status='indexed' "
                "AND caption_ocr IS NOT NULL AND caption_ocr!='' "
                "AND text_content_embedding_json IS NULL "
                "AND processing_attempts<? THEN 1 ELSE 0 END) AS pending, "
                "SUM(CASE WHEN processing_status='indexed' "
                "AND caption_ocr IS NOT NULL AND caption_ocr!='' "
                "AND text_content_embedding_json IS NULL "
                "AND processing_attempts>=? THEN 1 ELSE 0 END) AS failed "
                "FROM file_contents WHERE processing_status!='deleted'",
                (
                    MAX_FILE_PROCESSING_ATTEMPTS,
                    MAX_FILE_PROCESSING_ATTEMPTS,
                ),
            ).fetchone()
            parent_embeddings = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN parent_embedding_json IS NOT NULL "
                "THEN 1 ELSE 0 END) AS embedded, "
                "SUM(CASE WHEN parent_embedding_json IS NULL "
                "AND parent_embedding_attempts<? "
                "AND (thread_context IS NOT NULL OR upload_text IS NOT NULL) "
                "THEN 1 ELSE 0 END) AS pending, "
                "SUM(CASE WHEN parent_embedding_json IS NULL "
                "AND parent_embedding_attempts>=? "
                "AND (thread_context IS NOT NULL OR upload_text IS NOT NULL) "
                "THEN 1 ELSE 0 END) AS failed "
                "FROM file_shares WHERE tombstoned_at IS NULL",
                (
                    MAX_FILE_PROCESSING_ATTEMPTS,
                    MAX_FILE_PROCESSING_ATTEMPTS,
                ),
            ).fetchone()
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
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": list(PROTOCOL_CAPABILITIES),
            "file_index_store_generation": self.store_generation,
            "file_index_actions": {
                "apply": True,
                "reconcile": True,
                "search": True,
                "graph": True,
            },
            "message_count": int(message_count),
            "file_embedding_index": {
                "contents_total": int(content_embeddings["total"] or 0),
                "contents_embedded": int(content_embeddings["embedded"] or 0),
                "contents_pending": int(content_embeddings["pending"] or 0),
                "contents_failed": int(content_embeddings["failed"] or 0),
                "parents_total": int(parent_embeddings["total"] or 0),
                "parents_embedded": int(parent_embeddings["embedded"] or 0),
                "parents_pending": int(parent_embeddings["pending"] or 0),
                "parents_failed": int(parent_embeddings["failed"] or 0),
            },
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

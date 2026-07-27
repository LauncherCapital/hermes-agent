import base64
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gateway.event_ingress import write_project_marker
from hermes_cli.plugins import PluginManager


@pytest.fixture(autouse=True)
def _database_keyring(monkeypatch):
    monkeypatch.setenv(
        "RINGO_MESSAGE_STORE_DB_KEYS",
        json.dumps({"1": "11" * 32}),
    )
    monkeypatch.setenv("RINGO_MESSAGE_STORE_DB_KEY_VERSION", "1")


def _load_service() -> tuple[PluginManager, object]:
    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["ringo-message-store"]
    assert loaded.enabled
    assert loaded.module is not None
    return manager, loaded.module


def _message_store_health(manager: PluginManager) -> dict:
    return next(
        item
        for item in manager.invoke_hook("health_report")
        if item["name"] == "ringo_message_store"
    )


def _message_store_schema_version(module: object) -> int:
    return int(sys.modules[module.MessageStore.__module__].SCHEMA_VERSION)


@pytest.mark.asyncio
async def test_unclaimed_pool_instance_has_no_store_or_project_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager, _module = _load_service()

    health = _message_store_health(manager)

    assert health == {"name": "ringo_message_store", "status": "unclaimed"}
    assert not (tmp_path / "state/message_store.db").exists()
    assert not (tmp_path / "state/keys/message-store-v1.pem").exists()


@pytest.mark.asyncio
async def test_cold_claim_initializes_store_during_plugin_load(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)

    manager, module = _load_service()

    assert (tmp_path / "state/message_store.db").exists()
    assert (tmp_path / "state/keys/message-store-v1.pem").exists()
    health = _message_store_health(manager)
    assert health["project_id"] == project_id
    assert health["schema_version"] == _message_store_schema_version(module)


@pytest.mark.asyncio
async def test_claim_initializes_schema_key_and_idempotent_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager, module = _load_service()
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)

    claimed = await manager.invoke_hook_async("project_claimed", project_id=project_id)
    event = {
        "delivery_id": str(uuid.uuid4()),
        "sequence": 1,
        "project_id": project_id,
        "provider": "fixture",
        "workspace_id": "W1",
        "event_id": "E1",
        "event_type": "message.created",
        "conversation_id": "C1",
        "message_id": "M1",
        "occurred_at": "2026-07-21T00:00:00+00:00",
        "provider_version": "0001",
        "text": "fixture",
        "payload_hash": "payload-hash",
    }
    first = await manager.invoke_hook_async(
        "ingress_event", event=event, body_sha256="body-hash"
    )
    duplicate = await manager.invoke_hook_async(
        "ingress_event", event=event, body_sha256="body-hash"
    )

    assert claimed[0]["status"] == "ready"
    assert claimed[0]["key_registration"]["status"] == "pending"
    assert first[0]["status"] == "accepted"
    assert duplicate[0]["status"] == "duplicate"
    db_path = tmp_path / "state/message_store.db"
    private_path = tmp_path / "state/keys/message-store-v1.pem"
    assert db_path.exists()
    assert private_path.exists()
    assert db_path.read_bytes()[:16] != b"SQLite format 3\x00"
    with pytest.raises(sqlite3.DatabaseError):
        with sqlite3.connect(db_path) as plain:
            plain.execute("SELECT count(*) FROM sqlite_master").fetchone()
    store = module._store()
    assert store is not None
    with store._connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "deliveries",
            "delivery_cursor",
            "messages",
            "reactions",
            "conversations",
            "identities",
            "coverage",
            "conversation_memberships",
        } <= tables
        assert {
            "reconciliation_cycles",
            "reconciliation_seen",
        }.issubset(tables)
        assert {
            "file_contents",
            "file_shares",
            "file_tombstones",
            "file_index_runs",
            "file_graph_ingest_state",
        }.issubset(tables)
        assert (
            conn.execute("PRAGMA user_version").fetchone()[0]
            == _message_store_schema_version(module)
        )

    original_key = private_path.read_bytes()
    reopened = module.MessageStore(project_id, path=db_path)
    assert reopened.health()["status"] == "ready"
    assert private_path.read_bytes() == original_key
    health = reopened.health()
    assert health["last_sequence"] == 1
    assert {
        "schema_version",
        "protocol_version",
        "capabilities",
        "file_index_store_generation",
        "message_count",
        "key_version",
        "storage_encryption",
        "database_key_version",
        "encryption_integrity",
        "cipher_version",
        "journal_mode",
        "db_bytes",
        "wal_bytes",
        "lag_seconds",
        "unresolved_gaps",
        "coverage_states",
        "collection_states",
    } <= set(health)
    assert health["storage_encryption"] == "sqlcipher"
    assert health["database_key_version"] == 1
    assert health["encryption_integrity"] == "ok"
    assert health["protocol_version"] >= 1
    assert {
        "acl_metadata",
        "allowed_source_ids",
        "detailed_health",
        "event_batch",
        "file_index_v1",
        "ingest_window",
        "reconciliation_events",
        "stable_cursor",
    } <= set(health["capabilities"])
    assert health["message_count"] == 1


def test_schema_v5_upgrades_existing_v4_store_with_file_graph_cursor(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    with store._connect() as conn:
        conn.execute("DROP TABLE file_graph_ingest_state")
        conn.execute("ALTER TABLE file_contents DROP COLUMN processing_attempts")
        conn.execute("PRAGMA user_version=4")
        conn.commit()

    reopened = module.MessageStore(project_id, path=store.path)

    with reopened._connect() as conn:
        assert (
            conn.execute("PRAGMA user_version").fetchone()[0]
            == _message_store_schema_version(module)
        )
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='file_graph_ingest_state'"
        ).fetchone()
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(file_contents)")
        }
        assert "processing_attempts" in columns


def test_schema_v6_upgrades_existing_v5_file_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "upsert_share",
            "file_id": "F1",
            "conversation_id": "C1",
            "provider_message_id": "M1",
            "source_version": 1,
            "processing_status": "metadata_only",
        }
    )
    with store._connect() as conn:
        conn.execute("ALTER TABLE file_contents DROP COLUMN processing_attempts")
        conn.execute("PRAGMA user_version=5")
        conn.commit()

    reopened = module.MessageStore(project_id, path=store.path)

    with reopened._connect() as conn:
        content = conn.execute(
            "SELECT file_id, processing_attempts FROM file_contents"
        ).fetchone()
        assert (
            conn.execute("PRAGMA user_version").fetchone()[0]
            == _message_store_schema_version(module)
        )
    assert tuple(content) == ("F1", 0)


def test_retention_removes_expired_messages_reactions_and_deliveries(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    now = datetime.now(timezone.utc)
    expired_message_at = (now - timedelta(days=31)).isoformat()
    expired_delivery_at = (now - timedelta(days=8)).isoformat()
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO deliveries VALUES (?, ?, NULL, ?, ?, ?)",
            ("old-delivery", 1, "hash", expired_delivery_at, expired_delivery_at),
        )
        conn.execute(
            "INSERT INTO messages(project_id, provider, workspace_id, conversation_id, "
            "provider_message_id, occurred_at, inserted_at, updated_at) "
            "VALUES (?, 'fixture', 'W1', 'C1', 'M1', ?, ?, ?)",
            (project_id, expired_message_at, expired_message_at, expired_message_at),
        )
        conn.execute(
            "INSERT INTO reactions(project_id, provider, workspace_id, conversation_id, "
            "provider_message_id, reaction_name, actor_id, occurred_at) "
            "VALUES (?, 'fixture', 'W1', 'C1', 'M1', 'eyes', 'U1', ?)",
            (project_id, expired_message_at),
        )

    removed = store.run_retention(now=now)

    assert removed == {"deliveries": 1, "reactions": 1, "messages": 1}


def test_newer_sqlite_schema_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    db_path = tmp_path / "state/message_store.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version=999")

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["ringo-message-store"]
    assert loaded.enabled is False
    assert "newer than supported" in (loaded.error or "")


def test_plaintext_store_is_migrated_without_data_loss(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    db_path = tmp_path / "state/message_store.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE legacy_fixture(value TEXT NOT NULL)")
        conn.execute("INSERT INTO legacy_fixture VALUES ('preserved')")

    _manager, module = _load_service()
    store = module._store()
    assert store is not None

    assert db_path.read_bytes()[:16] != b"SQLite format 3\x00"
    assert not db_path.with_name(db_path.name + "-wal").exists()
    assert not db_path.with_name(db_path.name + "-shm").exists()
    with store._connect() as conn:
        assert conn.execute("SELECT value FROM legacy_fixture").fetchone()[0] == "preserved"
    assert store.health()["encryption_migration"] == "migrated"


def test_encrypted_store_fails_closed_without_matching_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    db_path = tmp_path / "state/message_store.db"
    module.MessageStore(project_id, path=db_path)

    monkeypatch.setenv("RINGO_MESSAGE_STORE_DB_KEYS", json.dumps({"2": "22" * 32}))
    monkeypatch.setenv("RINGO_MESSAGE_STORE_DB_KEY_VERSION", "2")
    with pytest.raises(RuntimeError, match="cannot be opened"):
        module.MessageStore(project_id, path=db_path)


def test_database_key_rotation_and_old_volume_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    db_path = tmp_path / "state/message_store.db"
    original = module.MessageStore(project_id, path=db_path)
    original.record_envelope(
        {"project_id": project_id, "delivery_id": "d1", "sequence": 1},
        "hash-1",
    )
    with original._connect() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    restored_path = tmp_path / "state/restored-message-store.db"
    shutil.copy2(db_path, restored_path)

    keyring = {"1": "11" * 32, "2": "22" * 32}
    monkeypatch.setenv("RINGO_MESSAGE_STORE_DB_KEYS", json.dumps(keyring))
    monkeypatch.setenv("RINGO_MESSAGE_STORE_DB_KEY_VERSION", "2")
    rotated = module.MessageStore(project_id, path=db_path)
    restored = module.MessageStore(project_id, path=restored_path)

    assert rotated.health()["encryption_migration"] == "rekeyed"
    assert restored.health()["encryption_migration"] == "rekeyed"
    assert rotated.health()["last_sequence"] == 1
    assert restored.health()["last_sequence"] == 1

    monkeypatch.setenv("RINGO_MESSAGE_STORE_DB_KEYS", json.dumps({"1": "11" * 32}))
    monkeypatch.setenv("RINGO_MESSAGE_STORE_DB_KEY_VERSION", "1")
    with pytest.raises(RuntimeError, match="cannot be opened"):
        module.MessageStore(project_id, path=db_path)


def test_private_key_recovery_copy_is_hybrid_encrypted_locally(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    recovery_private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    recovery_public_pem = recovery_private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    write_project_marker(
        project_id,
        recovery_public_keys={"platform-recovery-v1": recovery_public_pem},
    )
    _manager, module = _load_service()

    registration = module.build_recovery_registration(project_id)
    encoded_envelope = base64.b64decode(
        registration["wrapped_recovery_copy_b64"], validate=True
    )
    envelope = json.loads(encoded_envelope)
    dek = recovery_private.decrypt(
        base64.b64decode(envelope["wrapped_dek"]),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    aad_fields = {
        "schema_version": envelope["schema_version"],
        "project_id": envelope["project_id"],
        "key_version": envelope["key_version"],
        "recovery_key_id": envelope["recovery_key_id"],
    }
    aad = json.dumps(aad_fields, separators=(",", ":"), sort_keys=True).encode()
    recovered = AESGCM(dek).decrypt(
        base64.b64decode(envelope["nonce"]),
        base64.b64decode(envelope["ciphertext"]),
        aad,
    )

    private_pem = (tmp_path / "state/keys/message-store-v1.pem").read_bytes()
    assert recovered == private_pem
    assert b"PRIVATE KEY" not in encoded_envelope


def _encrypted_delivery(
    public_key,
    project_id,
    *,
    workspace_id="W1",
    sequence=1,
    key_version=1,
):
    delivery_id = str(uuid.uuid4())
    aad_fields = {
        "project_id": project_id,
        "provider": "fixture",
        "workspace_id": workspace_id,
        "delivery_id": delivery_id,
        "sequence": sequence,
        "schema_version": 1,
        "key_version": key_version,
    }
    aad = json.dumps(aad_fields, separators=(",", ":"), sort_keys=True).encode()
    event = {
        "text": "TOP SECRET FIXTURE",
        "conversation_id": "C1",
        "message_id": "M1",
        "occurred_at": "2026-07-21T00:00:00+00:00",
        "provider_version": "0001",
    }
    event_json = json.dumps(event, separators=(",", ":"), sort_keys=True).encode()
    plaintext = json.dumps(
        {
            "event": event,
            "plaintext_sha256": hashlib.sha256(event_json).hexdigest(),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    dek = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ciphertext = AESGCM(dek).encrypt(nonce, plaintext, aad)
    wrapped_dek = public_key.encrypt(
        dek,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        **aad_fields,
        "event_type": "message.created",
        "provider_event_id": "E1",
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        "encryption": {
            "algorithm": "AES-256-GCM",
            "key_wrap": "RSA-OAEP-SHA256",
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "wrapped_dek": base64.b64encode(wrapped_dek).decode(),
        },
    }


def test_encrypted_delivery_decrypts_and_mutated_aad_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    public_key = serialization.load_pem_public_key(
        (tmp_path / "state/keys/message-store-v1.pub.pem").read_bytes()
    )
    envelope = _encrypted_delivery(public_key, project_id)

    clear = module.decrypt_delivery_envelope(envelope)
    store = module.MessageStore(project_id)
    committed = store.record_envelope(clear, "signed-body-hash")
    replayed_after_lost_ack = store.record_envelope(clear, "signed-body-hash")
    mutated = dict(envelope)
    mutated["workspace_id"] = "W2"

    assert clear["text"] == "TOP SECRET FIXTURE"
    assert clear["project_id"] == project_id
    assert clear["payload_hash"] == envelope["ciphertext_sha256"]
    assert committed["status"] == "accepted"
    assert replayed_after_lost_ack["status"] == "duplicate"
    with pytest.raises(ValueError, match="authentication failed"):
        module.decrypt_delivery_envelope(mutated)


def test_gap_delivery_is_not_committed_until_replayed_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)

    gap = store.record_envelope(
        {"project_id": project_id, "delivery_id": "d2", "sequence": 2},
        "hash-2",
    )
    first = store.record_envelope(
        {"project_id": project_id, "delivery_id": "d1", "sequence": 1},
        "hash-1",
    )
    replay = store.record_envelope(
        {"project_id": project_id, "delivery_id": "d2", "sequence": 2},
        "hash-2",
    )

    assert gap == {"status": "gap_detected", "sequence": 2, "expected_sequence": 1}
    assert first["status"] == "accepted"
    assert replay["status"] == "accepted"
    assert store.health()["last_sequence"] == 2
    assert store.health()["unresolved_gaps"] == 0


@pytest.mark.asyncio
async def test_key_rotation_keeps_retired_version_decrypt_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    manager, module = _load_service()
    public_v1 = serialization.load_pem_public_key(
        (tmp_path / "state/keys/message-store-v1.pub.pem").read_bytes()
    )
    envelope_v1 = _encrypted_delivery(public_v1, project_id, key_version=1)

    write_project_marker(project_id, active_key_version=2)
    claimed = await manager.invoke_hook_async(
        "project_claimed", project_id=project_id, active_key_version=2
    )
    public_v2 = serialization.load_pem_public_key(
        (tmp_path / "state/keys/message-store-v2.pub.pem").read_bytes()
    )
    envelope_v2 = _encrypted_delivery(public_v2, project_id, key_version=2)

    assert claimed[0]["status"] == "ready"
    assert (tmp_path / "state/keys/message-store-v1.pem").exists()
    assert (tmp_path / "state/keys/message-store-v2.pem").exists()
    assert module.decrypt_delivery_envelope(envelope_v1)["key_version"] == 1
    assert module.decrypt_delivery_envelope(envelope_v2)["key_version"] == 2
    assert _message_store_health(manager)["key_version"] == 2


def test_normalized_events_share_one_apply_path_and_tombstone_wins(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)

    events = [
        {
            "event_type": "message.created",
            "conversation_id": "C1",
            "message_id": "M1",
            "sender_id": "U1",
            "text": "original",
            "occurred_at": "2026-07-21T00:00:01+00:00",
            "provider_version": "0002",
        },
        {
            "event_type": "message.deleted",
            "conversation_id": "C1",
            "message_id": "M1",
            "occurred_at": "2026-07-21T00:00:03+00:00",
            "provider_version": "0003",
        },
        {
            "event_type": "message.updated",
            "conversation_id": "C1",
            "message_id": "M1",
            "sender_id": "U1",
            "text": "stale resurrection",
            "occurred_at": "2026-07-21T00:00:02+00:00",
            "provider_version": "0002",
        },
        {
            "event_type": "reaction.added",
            "conversation_id": "C1",
            "message_id": "M1",
            "reaction_name": "eyes",
            "actor_id": "U2",
            "occurred_at": "2026-07-21T00:00:04+00:00",
        },
        {
            "event_type": "reaction.removed",
            "conversation_id": "C1",
            "message_id": "M1",
            "reaction_name": "eyes",
            "actor_id": "U2",
            "occurred_at": "2026-07-21T00:00:05+00:00",
        },
        {
            "event_type": "conversation.upsert",
            "conversation_id": "C1",
            "conversation_type": "channel",
            "title": "general",
            "is_private": True,
            "occurred_at": "2026-07-21T00:00:06+00:00",
        },
        {
            "event_type": "conversation.upsert",
            "conversation_id": "C1",
            "is_archived": True,
            "occurred_at": "2026-07-21T00:00:06.500000+00:00",
        },
        {
            "event_type": "identity.upsert",
            "external_user_id": "U1",
            "display_name": "Sunhee",
            "occurred_at": "2026-07-21T00:00:07+00:00",
        },
        {
            "event_type": "membership.changed",
            "conversation_id": "C1",
            "external_user_id": "U1",
            "is_member": True,
            "provider_version": "0008",
            "occurred_at": "2026-07-21T00:00:08+00:00",
        },
    ]
    for sequence, event in enumerate(events, start=1):
        store.record_envelope(
            {
                **event,
                "project_id": project_id,
                "provider": "slack",
                "workspace_id": "T1",
                "delivery_id": f"d{sequence}",
                "sequence": sequence,
            },
            f"hash-{sequence}",
        )

    with store._connect() as conn:
        message = conn.execute(
            "SELECT text, deleted_at, provider_version FROM messages"
        ).fetchone()
        reaction = conn.execute(
            "SELECT deleted_at FROM reactions WHERE reaction_name = 'eyes'"
        ).fetchone()
        conversation = conn.execute(
            "SELECT title, is_private, is_archived FROM conversations "
            "WHERE conversation_id = 'C1'"
        ).fetchone()
        identity = conn.execute(
            "SELECT display_name FROM identities WHERE external_user_id = 'U1'"
        ).fetchone()
        membership = conn.execute(
            "SELECT is_member FROM conversation_memberships"
        ).fetchone()

    assert message[0] is None
    assert message[1] == "2026-07-21T00:00:03+00:00"
    assert message[2] == "0003"
    assert reaction[0] == "2026-07-21T00:00:05+00:00"
    assert conversation[0] == "general"
    assert conversation[1:] == (1, 1)
    assert identity[0] == "Sunhee"
    assert membership[0] == 1


def test_file_index_content_share_cas_and_delete_tombstone(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)

    base = {
        "project_id": project_id,
        "store_generation": store.store_generation,
        "provider": "slack",
        "workspace_id": "T1",
        "operation": "upsert_share",
        "file_id": "F1",
        "conversation_id": "C1",
        "provider_message_id": "1700.1",
        "occurred_at": "2026-07-21T00:00:01+00:00",
        "source_version": 2,
        "content_version": 2,
        "context_version": 2,
        "file_name": "profile.png",
        "uploader_id": "U1",
        "uploaded_at": "2026-07-21T00:00:00+00:00",
        "content_sha256": "content-v2",
        "processing_status": "indexed",
        "upload_text": "Ringo rebrand profile",
        "thread_context": "new icon",
        "caption_ocr": "yellow app icon",
        "text_content_embedding": [0.1, 0.2],
        "image_embedding": [0.3, 0.4],
    }
    store.apply_file_command(base)
    store.apply_file_command(
        {
            **base,
            "source_version": 3,
            "content_version": 1,
            "context_version": 3,
            "upload_text": "edited upload text",
            "thread_context": "edited thread",
            "processing_status": "indexed",
            "caption_ocr": "stale caption",
            "content_sha256": "stale-content",
        }
    )
    store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "delete_file",
            "file_id": "F1",
            "occurred_at": "2026-07-21T00:00:04+00:00",
            "source_version": 4,
        }
    )
    stale = store.apply_file_command(
        {
            **base,
            "source_version": 3,
            "content_version": 3,
            "context_version": 3,
            "caption_ocr": "late stale caption",
        }
    )

    with store._connect() as conn:
        content = conn.execute(
            "SELECT * FROM file_contents WHERE workspace_id='T1' AND file_id='F1'"
        ).fetchone()
        share = conn.execute(
            "SELECT * FROM file_shares WHERE workspace_id='T1' AND file_id='F1'"
        ).fetchone()
        tombstone = conn.execute(
            "SELECT source_version FROM file_tombstones "
            "WHERE workspace_id='T1' AND file_id='F1'"
        ).fetchone()

    assert stale["status"] == "stale"
    assert share["upload_text"] == "edited upload text"
    assert share["thread_context"] == "edited thread"
    assert share["tombstoned_at"] == "2026-07-21T00:00:04+00:00"
    assert content["caption_ocr"] == "yellow app icon"
    assert content["content_sha256"] == "content-v2"
    assert content["processing_status"] == "deleted"
    assert tombstone["source_version"] == 4


def test_file_index_processes_raw_file_outside_transaction_without_storing_token(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store_module = sys.modules[module.MessageStore.__module__]
    seen = {}

    def fake_process(file_id, access_token, _reuse_by_sha=None):
        seen.update(file_id=file_id, access_token=access_token)
        return {
            "file_name": "profile.png",
            "mime_type": "image/png",
            "byte_size": 42,
            "content_sha256": "abc123",
            "processing_status": "indexed",
            "caption_ocr": "yellow Ringo profile icon",
            "text_content_embedding": [0.1, 0.2],
            "image_embedding": [0.1, 0.2],
            "caption_model": "vision-test",
            "caption_prompt_version": "file-index-v1",
            "text_embedding_model": "embedding-test",
            "text_embedding_dimension": 2,
            "image_embedding_model": "caption-text:embedding-test",
            "image_embedding_dimension": 2,
            "last_error_code": None,
        }

    monkeypatch.setattr(store_module, "process_slack_file", fake_process)
    result = store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "upsert_share",
            "file_id": "F1",
            "conversation_id": "C1",
            "provider_message_id": "M1",
            "source_version": 1,
            "content_version": 1,
            "context_version": 1,
            "file_name": "profile.png",
            "processing_status": "metadata_only",
            "provider_access_token": "xoxb-transient-secret",
        }
    )

    with store._connect() as conn:
        content = conn.execute(
            "SELECT * FROM file_contents WHERE file_id='F1'"
        ).fetchone()
        serialized_rows = "\n".join(
            "|".join("" if value is None else str(value) for value in row)
            for table in ("file_contents", "file_shares", "schema_meta")
            for row in conn.execute(f"SELECT * FROM {table}").fetchall()
        )

    assert result["indexed_contents"] == 1
    assert seen == {
        "file_id": "F1",
        "access_token": "xoxb-transient-secret",
    }
    assert content["processing_status"] == "indexed"
    assert content["content_sha256"] == "abc123"
    assert json.loads(content["image_embedding_json"]) == [0.1, 0.2]
    assert "xoxb-transient-secret" not in serialized_rows


def test_file_processing_deletes_transient_bytes_on_success_and_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    processing = sys.modules[module.MessageStore.apply_file_command.__globals__[
        "process_slack_file"
    ].__module__]
    downloaded = tmp_path / "state" / "tmp" / "file-index" / "fi-test"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(processing.FileProcessingError) as invalid_host:
        processing._download(
            url="https://example.com/private",
            access_token="xoxb-secret",
            expected_size=3,
        )
    assert invalid_host.value.code == "file_download_host_invalid"

    monkeypatch.setattr(
        processing,
        "_slack_file_info",
        lambda _file_id, _token, _reuse_by_sha=None: {
            "name": "profile.png",
            "mimetype": "image/png",
            "size": 3,
            "url_private_download": "https://files.slack.test/F1",
        },
    )

    def fake_download(**_kwargs):
        downloaded.write_bytes(b"raw")
        return downloaded, "abc", 3

    monkeypatch.setattr(processing, "_download", fake_download)
    monkeypatch.setattr(
        processing,
        "_caption_image",
        lambda _path: ("yellow profile icon", "vision-test"),
    )
    monkeypatch.setattr(
        processing,
        "embed_text",
        lambda _text: ([0.1, 0.2], "embedding-test"),
    )

    result = processing.process_slack_file("F1", "xoxb-secret")

    assert result["processing_status"] == "indexed"
    assert not downloaded.exists()

    monkeypatch.setattr(
        processing,
        "_caption_image",
        lambda _path: pytest.fail("same hash must reuse derived content"),
    )
    reused = processing.process_slack_file(
        "F1",
        "xoxb-secret",
        lambda digest, mime: (
            {
                "caption_ocr": "cached caption",
                "image_embedding": [0.3, 0.4],
            }
            if (digest, mime) == ("abc", "image/png")
            else None
        ),
    )
    assert reused["caption_ocr"] == "cached caption"
    assert not downloaded.exists()

    def fail_caption(_path):
        raise processing.FileProcessingError("image_caption_failed")

    monkeypatch.setattr(processing, "_caption_image", fail_caption)
    failed = processing.process_slack_file("F1", "xoxb-secret")

    assert failed["processing_status"] == "metadata_only"
    assert failed["last_error_code"] == "image_caption_failed"
    assert not downloaded.exists()

    def fail_file_info(_file_id, _token):
        raise processing.FileProcessingError("slack_file_info_failed")

    monkeypatch.setattr(processing, "_slack_file_info", fail_file_info)
    metadata_failed = processing.process_slack_file("F1", "xoxb-secret")
    assert metadata_failed == {
        "processing_status": "metadata_only",
        "last_error_code": "slack_file_info_failed",
    }


def test_image_caption_neuters_async_client_destructor_before_analysis(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    processing = sys.modules[module.MessageStore.apply_file_command.__globals__[
        "process_slack_file"
    ].__module__]
    from agent import auxiliary_client
    from tools import vision_tools

    calls = []

    def fake_neuter():
        calls.append("neuter")

    async def fake_vision(_path, _prompt, _model):
        calls.append("vision")
        return json.dumps({"success": True, "analysis": "yellow profile icon"})

    monkeypatch.setattr(auxiliary_client, "neuter_async_httpx_del", fake_neuter)
    monkeypatch.setattr(vision_tools, "vision_analyze_tool", fake_vision)

    result = processing._caption_image(tmp_path / "profile.png")

    assert result == ("yellow profile icon", "auxiliary-default")
    assert calls == ["neuter", "vision"]


def test_retryable_file_processing_stops_and_continues(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    for version, file_id in enumerate(("F1", "F2"), start=1):
        store.apply_file_command(
            {
                "project_id": project_id,
                "store_generation": store.store_generation,
                "provider": "slack",
                "workspace_id": "T1",
                "operation": "upsert_share",
                "file_id": file_id,
                "conversation_id": "C1",
                "provider_message_id": f"M{version}",
                "source_version": version,
                "processing_status": "metadata_only",
            }
        )
    store_module = sys.modules[module.MessageStore.__module__]

    def fake_process(file_id, _token, _reuse_by_sha=None):
        return {
            "processing_status": (
                "metadata_only" if file_id == "F1" else "indexed"
            ),
            "content_sha256": "f2" if file_id == "F2" else None,
            "last_error_code": (
                "embedding_failed" if file_id == "F1" else None
            ),
        }

    monkeypatch.setattr(
        store_module,
        "process_slack_file",
        fake_process,
    )

    processed = [
        store._process_pending_file_contents(
            provider="slack",
            workspace_id="T1",
            access_token="xoxb-transient",
            limit=1,
        )
        for _ in range(4)
    ]

    with store._connect() as conn:
        contents = conn.execute(
            "SELECT file_id, processing_status, processing_attempts, "
            "last_error_code "
            "FROM file_contents ORDER BY file_id"
        ).fetchall()
    assert processed == [0, 0, 1, 1]
    assert [tuple(row) for row in contents] == [
        ("F1", "unavailable", 3, "embedding_failed"),
        ("F2", "indexed", 0, None),
    ]


def test_file_change_refreshes_content_without_rewriting_share_context(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "upsert_share",
            "file_id": "F1",
            "conversation_id": "C1",
            "provider_message_id": "M1",
            "source_version": 1,
            "content_version": 1,
            "context_version": 1,
            "processing_status": "indexed",
            "content_sha256": "old",
            "upload_text": "original context",
        }
    )
    store_module = sys.modules[module.MessageStore.__module__]
    monkeypatch.setattr(
        store_module,
        "process_slack_file",
        lambda _file_id, _token, _reuse_by_sha=None: {
            "processing_status": "indexed",
            "content_sha256": "new",
            "file_name": "profile-v2.png",
            "mime_type": "image/png",
            "last_error_code": None,
        },
    )

    result = store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "refresh_file",
            "file_id": "F1",
            "source_version": 2,
            "provider_access_token": "xoxb-transient",
        }
    )

    with store._connect() as conn:
        content = conn.execute("SELECT * FROM file_contents").fetchone()
        share = conn.execute("SELECT * FROM file_shares").fetchone()
    assert result["indexed_contents"] == 1
    assert content["content_version"] == 2
    assert content["content_sha256"] == "new"
    assert share["upload_text"] == "original context"
    assert share["context_version"] == 1


def test_slow_file_processing_does_not_hold_message_writer_lock(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store_module = sys.modules[module.MessageStore.__module__]
    entered = threading.Event()
    release = threading.Event()

    def slow_process(_file_id, _token, _reuse_by_sha=None):
        entered.set()
        assert release.wait(timeout=3)
        return {
            "processing_status": "unsupported",
            "last_error_code": "fixture",
        }

    monkeypatch.setattr(store_module, "process_slack_file", slow_process)
    thread = threading.Thread(
        target=store.apply_file_command,
        args=(
            {
                "project_id": project_id,
                "store_generation": store.store_generation,
                "provider": "slack",
                "workspace_id": "T1",
                "operation": "upsert_share",
                "file_id": "F1",
                "conversation_id": "C1",
                "provider_message_id": "M1",
                "source_version": 1,
                "content_version": 1,
                "context_version": 1,
                "processing_status": "metadata_only",
                "provider_access_token": "xoxb-transient",
            },
        ),
    )
    thread.start()
    assert entered.wait(timeout=3)
    started = time.monotonic()
    store.record_envelope(
        {
            "project_id": project_id,
            "provider": "slack",
            "workspace_id": "T1",
            "delivery_id": "D1",
            "sequence": 1,
            "event_type": "message.created",
            "conversation_id": "C1",
            "message_id": "M2",
            "occurred_at": "2026-07-28T00:00:00+00:00",
            "provider_version": "1",
            "text": "message is not blocked",
        },
        "hash-D1",
    )
    elapsed = time.monotonic() - started
    release.set()
    thread.join(timeout=3)

    assert elapsed < 1
    assert not thread.is_alive()


def test_file_index_same_content_has_independent_shares_and_context_only_edit(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    base = {
        "project_id": project_id,
        "store_generation": store.store_generation,
        "provider": "slack",
        "workspace_id": "T1",
        "operation": "upsert_share",
        "file_id": "F1",
        "source_version": 2,
        "content_version": 2,
        "context_version": 2,
        "processing_status": "indexed",
        "content_sha256": "content-v2",
        "caption_ocr": "yellow app icon",
    }
    store.apply_file_command(
        {
            **base,
            "conversation_id": "C1",
            "provider_message_id": "M1",
            "upload_text": "first share",
        }
    )
    store.apply_file_command(
        {
            **base,
            "conversation_id": "C2",
            "provider_message_id": "M2",
            "upload_text": "second share",
        }
    )
    store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "upsert_share",
            "file_id": "F1",
            "conversation_id": "C1",
            "provider_message_id": "M1",
            "source_version": 3,
            "content_version": 3,
            "context_version": 3,
            "upload_text": "edited context only",
        }
    )

    with store._connect() as conn:
        contents = conn.execute("SELECT * FROM file_contents").fetchall()
        shares = conn.execute(
            "SELECT conversation_id, upload_text FROM file_shares "
            "ORDER BY conversation_id"
        ).fetchall()

    assert len(contents) == 1
    assert contents[0]["content_version"] == 2
    assert contents[0]["caption_ocr"] == "yellow app icon"
    assert [tuple(row) for row in shares] == [
        ("C1", "edited context only"),
        ("C2", "second share"),
    ]

    store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "remove_share",
            "file_id": "F1",
            "conversation_id": "C1",
            "provider_message_id": "*",
            "source_version": 4,
        }
    )
    with store._connect() as conn:
        channel_states = conn.execute(
            "SELECT conversation_id, tombstone_version FROM file_shares "
            "ORDER BY conversation_id"
        ).fetchall()
    assert [tuple(row) for row in channel_states] == [
        ("C1", 4),
        ("C2", None),
    ]


def test_file_reconcile_resumes_local_encrypted_message_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    for sequence, (message_id, file_id) in enumerate(
        (("1700.1", "F1"), ("1700.2", "F2")),
        start=1,
    ):
        store.record_envelope(
            {
                "project_id": project_id,
                "provider": "slack",
                "workspace_id": "T1",
                "conversation_id": "C1",
                "delivery_id": f"d-{sequence}",
                "sequence": sequence,
                "event_type": "message.created",
                "message_id": message_id,
                "sender_id": "U1",
                "text": f"upload {file_id}",
                "occurred_at": f"2026-07-21T00:00:0{sequence}+00:00",
                "provider_version": f"000{sequence}",
                "provider_payload": {
                    "files": [
                        {
                            "id": file_id,
                            "name": f"{file_id}.png",
                            "mimetype": "image/png",
                            "size": 123,
                            "timestamp": 1784592000 + sequence,
                            "user": "U1",
                            "permalink": f"https://slack.test/{file_id}",
                        }
                    ]
                },
            },
            f"hash-{sequence}",
        )
    request = {
        "project_id": project_id,
        "provider": "slack",
        "workspace_id": "T1",
        "store_generation": store.store_generation,
        "window_started_at": "2026-07-20T00:00:00+00:00",
        "window_ended_at": "2026-07-22T00:00:00+00:00",
        "allowed_source_ids": ["slack:T1:C1"],
        "allowed_scope_revision": "scope-1",
        "batch_limit": 1,
        "processing_signature": "metadata-v1",
    }

    first = store.reconcile_file_window(request)
    second = store.reconcile_file_window(request)
    third = store.reconcile_file_window(request)

    with store._connect() as conn:
        contents = conn.execute(
            "SELECT file_id, processing_status FROM file_contents ORDER BY file_id"
        ).fetchall()
        shares = conn.execute(
            "SELECT file_id, upload_text FROM file_shares ORDER BY file_id"
        ).fetchall()
        run = conn.execute("SELECT * FROM file_index_runs").fetchone()

    assert first["status"] == "continuation"
    assert second["status"] == "continuation"
    assert third["status"] == "complete"
    assert [tuple(row) for row in contents] == [
        ("F1", "metadata_only"),
        ("F2", "metadata_only"),
    ]
    assert [tuple(row) for row in shares] == [
        ("F1", "upload F1"),
        ("F2", "upload F2"),
    ]
    assert run["status"] == "complete"
    assert run["scanned_messages"] == 2
    assert run["discovered_shares"] == 2


def test_file_search_uses_fixed_acl_retrieve_rerank_inspect_limits(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store_module = sys.modules[module.MessageStore.__module__]
    monkeypatch.setattr(
        store_module,
        "embed_text",
        lambda _text: ([1.0, 0.0], "embedding-test"),
    )
    monkeypatch.setattr(
        store_module,
        "inspect_slack_image",
        lambda _file_id, _token, _query: "Ringo rebrand profile icon",
    )
    fixtures = [
        (
            "F0",
            "Screenshot 2026-05-15 at 11.00.31 AM.png",
            "C1",
            "링고 리브랜딩 프로필 사진 올립니다",
            "yellow Ringo profile logo",
        ),
        (
            "F1",
            "Screenshot 2026-05-15 at 11.00.31 AM.png",
            "C1",
            "ㅋㅋㅋ",
            "Ringo2 app installer drag to Applications",
        ),
        (
            "F2",
            "profile-option.png",
            "C1",
            "old unrelated profile draft",
            "blue abstract icon",
        ),
        (
            "F3",
            "brand-notes.png",
            "C1",
            "branding discussion",
            "text notes",
        ),
        (
            "F4",
            "ringo-profile-secret.png",
            "C2",
            "링고 리브랜딩 프로필 사진",
            "yellow Ringo profile logo",
        ),
    ]
    for index, (file_id, name, channel_id, upload_text, caption) in enumerate(
        fixtures
    ):
        store.apply_file_command(
            {
                "project_id": project_id,
                "store_generation": store.store_generation,
                "provider": "slack",
                "workspace_id": "T1",
                "operation": "upsert_share",
                "file_id": file_id,
                "conversation_id": channel_id,
                "provider_message_id": f"M{index}",
                "source_version": index + 1,
                "content_version": index + 1,
                "context_version": index + 1,
                "file_name": name,
                "mime_type": "image/png",
                "processing_status": "indexed",
                "caption_ocr": caption,
                "text_content_embedding": [1.0, 0.0],
                "image_embedding": [1.0, 0.0],
                "upload_text": upload_text,
                "shared_at": f"2026-07-21T00:00:0{index}+00:00",
            }
        )
    for index in range(10):
        store.apply_file_command(
            {
                "project_id": project_id,
                "store_generation": store.store_generation,
                "provider": "slack",
                "workspace_id": "T1",
                "operation": "upsert_share",
                "file_id": f"N{index}",
                "conversation_id": "C1",
                "provider_message_id": f"MN{index}",
                "source_version": 20 + index,
                "content_version": 20 + index,
                "context_version": 20 + index,
                "file_name": f"notes-{index}.txt",
                "mime_type": "text/plain",
                "processing_status": "indexed",
                "caption_ocr": "generic notes",
                "text_content_embedding": [1.0, 0.0],
                "upload_text": "unrelated notes",
                "shared_at": f"2026-07-20T00:00:{index:02d}+00:00",
            }
        )
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO file_index_runs(provider, workspace_id, "
            "store_generation, window_started_at, window_ended_at, "
            "allowed_scope_revision, status, processing_signature, updated_at, "
            "completed_at) VALUES (?, ?, ?, ?, ?, ?, 'complete', ?, ?, ?)",
            (
                "slack",
                "T1",
                store.store_generation,
                "2026-06-21T00:00:00+00:00",
                "2026-07-21T00:00:00+00:00",
                "scope-1",
                "content-v1",
                "2026-07-21T00:00:00+00:00",
                "2026-07-21T00:00:00+00:00",
            ),
        )
        conn.commit()

    result = store.search_file_index(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "query": "링고 리브랜딩 관련 프로필 사진 올린 거",
            "allowed_source_ids": ["slack:T1:C1"],
            "allowed_scope_revision": "scope-1",
            "limit": 20,
            "provider_access_token": "xoxb-transient",
        }
    )

    assert result["coverage_complete"] is True
    assert result["retrieve_count"] == 14
    assert result["rerank_count"] == 10
    assert result["inspected_image_count"] == 3
    assert len(result["files"]) == 3
    assert {item["channel_id"] for item in result["files"]} == {"C1"}
    assert sum(item["image_inspected"] for item in result["files"]) == 3
    assert result["files"][0]["file_id"] == "F0"
    assert "upload:" in result["files"][0]["why_matched"]


def test_file_graph_batch_has_independent_ack_cursor_and_no_vectors(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "upsert_share",
            "file_id": "F1",
            "conversation_id": "C1",
            "provider_message_id": "M1",
            "source_version": 2,
            "content_version": 2,
            "context_version": 2,
            "file_name": "profile.png",
            "mime_type": "image/png",
            "permalink": "https://slack.test/F1",
            "processing_status": "indexed",
            "caption_ocr": "Ringo rebrand profile icon",
            "text_content_embedding": [0.1, 0.2],
            "image_embedding": [0.3, 0.4],
            "upload_text": "Ringo rebrand",
            "thread_context": "use as the new profile photo",
            "shared_at": "2026-07-28T00:00:00+00:00",
        }
    )
    request = {
        "project_id": project_id,
        "store_generation": store.store_generation,
        "provider": "slack",
        "workspace_id": "T1",
        "allowed_source_ids": ["slack:T1:C1"],
        "limit": 3,
    }

    first = store.file_graph_batch(request)
    episode = first["episodes"][0]
    serialized = json.dumps(episode, ensure_ascii=False)
    assert "Ringo rebrand profile icon" in episode["body"]
    assert episode["source_description"].startswith("slack:T1:C1 ")
    assert "embedding" not in serialized
    assert "content_sha256" not in serialized

    metadata = episode["source_metadata"]
    store.ack_file_graph_batch(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "acknowledgements": [
                {
                    "file_id": "F1",
                    "conversation_id": "C1",
                    "provider_message_id": "M1",
                    "content_version": metadata["content_version"],
                    "context_version": metadata["context_version"],
                }
            ],
        }
    )
    second = store.file_graph_batch(request)

    assert second["episodes"] == []


def test_complete_reconcile_expires_old_share_and_unreferenced_content(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store.apply_file_command(
        {
            "project_id": project_id,
            "store_generation": store.store_generation,
            "provider": "slack",
            "workspace_id": "T1",
            "operation": "upsert_share",
            "file_id": "F-old",
            "conversation_id": "C1",
            "provider_message_id": "M-old",
            "source_version": 1,
            "content_version": 1,
            "context_version": 1,
            "processing_status": "indexed",
            "shared_at": "2026-05-01T00:00:00+00:00",
        }
    )

    result = store.reconcile_file_window(
        {
            "project_id": project_id,
            "provider": "slack",
            "workspace_id": "T1",
            "store_generation": store.store_generation,
            "window_started_at": "2026-06-28T00:00:00+00:00",
            "window_ended_at": "2026-07-28T00:00:00+00:00",
            "allowed_source_ids": ["slack:T1:C1"],
            "allowed_scope_revision": "scope-1",
            "batch_limit": 10,
            "processing_signature": "content-v1",
        }
    )

    with store._connect() as conn:
        share = conn.execute("SELECT * FROM file_shares").fetchone()
        content_count = conn.execute(
            "SELECT COUNT(*) FROM file_contents"
        ).fetchone()[0]
    assert result["status"] == "complete"
    assert share["tombstoned_at"] is not None
    assert content_count == 0


def test_normalized_event_batch_applies_atomically_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)

    result = store.record_envelope(
        {
            "project_id": project_id,
            "provider": "slack",
            "workspace_id": "T1",
            "delivery_id": "batch-1",
            "sequence": 1,
            "event_type": "events.batch",
            "events": [
                {
                    "event_type": "message.created",
                    "conversation_id": "C1",
                    "message_id": "M1",
                    "sender_id": "U1",
                    "text": "original",
                    "occurred_at": "2026-07-21T00:00:01+00:00",
                    "provider_version": "0001",
                    "provider_payload": {
                        "reactions": [
                            {
                                "name": "white_check_mark",
                                "count": 3,
                                "users": ["U1", "U2"],
                            }
                        ]
                    },
                },
                {
                    "event_type": "message.updated",
                    "conversation_id": "C1",
                    "message_id": "M1",
                    "sender_id": "U1",
                    "text": "edited",
                    "occurred_at": "2026-07-21T00:00:02+00:00",
                    "provider_version": "0002",
                    "provider_payload": {
                        "reactions": [
                            {
                                "name": "white_check_mark",
                                "count": 3,
                                "users": ["U1", "U2"],
                            }
                        ]
                    },
                },
                {
                    "event_type": "reaction.added",
                    "conversation_id": "C1",
                    "message_id": "M1",
                    "reaction_name": "white_check_mark",
                    "actor_id": "U2",
                    "occurred_at": "2026-07-21T00:00:03+00:00",
                },
                {
                    "event_type": "coverage.completed",
                    "conversation_id": "C1",
                    "contiguous_since": "2026-07-20T00:00:00+00:00",
                    "occurred_at": "2026-07-21T00:00:04+00:00",
                },
            ],
        },
        "batch-hash",
    )

    with store._connect() as conn:
        message = conn.execute(
            "SELECT text, provider_version FROM messages WHERE provider_message_id='M1'"
        ).fetchone()
        reaction = conn.execute(
            "SELECT reaction_name, actor_id FROM reactions"
        ).fetchone()
        deliveries = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
    queried = store.query(
        {
            "operation": "fetch_history",
            "start": "2026-07-21T00:00:00+00:00",
            "end": "2026-07-21T01:00:00+00:00",
            "providers": ["slack"],
            "workspace_ids": ["T1"],
            "conversation_ids": ["C1"],
            "limit": 10,
        }
    )

    assert result["status"] == "accepted"
    assert tuple(message) == ("edited", "0002")
    assert tuple(reaction) == ("white_check_mark", "U2")
    assert deliveries == 1
    assert queried["messages"][0]["reactions"] == [
        {
            "name": "white_check_mark",
            "count": 3,
            "users": ["U1", "U2"],
        }
    ]


def test_normalized_event_batch_rolls_back_every_child_on_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)

    with pytest.raises(ValueError, match="unsupported normalized event type"):
        store.record_envelope(
            {
                "project_id": project_id,
                "provider": "slack",
                "workspace_id": "T1",
                "delivery_id": "batch-invalid",
                "sequence": 1,
                "event_type": "events.batch",
                "events": [
                    {
                        "event_type": "message.created",
                        "conversation_id": "C1",
                        "message_id": "M1",
                        "text": "must roll back",
                        "occurred_at": "2026-07-21T00:00:01+00:00",
                        "provider_version": "0001",
                    },
                    {"event_type": "unsupported.fixture"},
                ],
            },
            "batch-invalid-hash",
        )

    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM delivery_cursor"
        ).fetchone()[0] == 0


def test_snapshot_query_reads_exact_acked_ids_without_claiming_contiguous_coverage(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store.record_envelope(
        {
            "project_id": project_id,
            "provider": "slack",
            "workspace_id": "T1",
            "delivery_id": "snapshot-batch",
            "sequence": 1,
            "event_type": "events.batch",
            "events": [
                {
                    "event_type": "message.created",
                    "conversation_id": "C1",
                    "message_id": "M1",
                    "text": "selected",
                    "occurred_at": "2026-07-21T00:00:01+00:00",
                    "provider_version": "0001",
                },
                {
                    "event_type": "message.created",
                    "conversation_id": "C2",
                    "message_id": "M2",
                    "text": "not selected",
                    "occurred_at": "2026-07-21T00:00:02+00:00",
                    "provider_version": "0002",
                },
            ],
        },
        "snapshot-batch-hash",
    )
    base = {
        "operation": "fetch_snapshot",
        "start": "2026-07-21T00:00:00+00:00",
        "end": "2026-07-21T01:00:00+00:00",
        "allowed_source_ids": ["slack:T1:C1"],
        "provider_message_keys": [
            {"conversation_id": "C1", "provider_message_id": "M1"}
        ],
        "limit": 10,
    }

    exact = store.query(base)
    missing = store.query(
        {
            **base,
            "provider_message_keys": [
                {"conversation_id": "C1", "provider_message_id": "missing"}
            ],
        }
    )

    assert exact["coverage_complete"] is True
    assert exact["reason"] == "snapshot_exact"
    assert [item["provider_message_id"] for item in exact["messages"]] == ["M1"]
    assert missing["coverage_complete"] is False
    assert missing["reason"] == "snapshot_missing"
    with pytest.raises(ValueError, match="outside conversation scope"):
        store.query(
            {
                **base,
                "provider_message_keys": [
                    {"conversation_id": "C2", "provider_message_id": "M2"}
                ],
            }
        )


def test_reaction_tombstone_rejects_stale_add(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)

    for sequence, (event_type, occurred_at) in enumerate(
        [
            ("reaction.removed", "2026-07-21T00:00:05+00:00"),
            ("reaction.added", "2026-07-21T00:00:04+00:00"),
        ],
        start=1,
    ):
        store.record_envelope(
            {
                "event_type": event_type,
                "conversation_id": "C1",
                "message_id": "M1",
                "reaction_name": "eyes",
                "actor_id": "U2",
                "occurred_at": occurred_at,
                "project_id": project_id,
                "provider": "slack",
                "workspace_id": "T1",
                "delivery_id": f"d{sequence}",
                "sequence": sequence,
            },
            f"hash-{sequence}",
        )

    with store._connect() as conn:
        reaction = conn.execute(
            "SELECT occurred_at, deleted_at FROM reactions WHERE reaction_name = 'eyes'"
        ).fetchone()
    assert tuple(reaction) == (
        "2026-07-21T00:00:05+00:00",
        "2026-07-21T00:00:05+00:00",
    )


def test_workspace_purge_removes_only_target_partition(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    sequence = 0
    for workspace_id in ("T1", "T2"):
        sequence += 1
        store.record_envelope(
            {
                "project_id": project_id,
                "provider": "slack",
                "workspace_id": workspace_id,
                "conversation_id": "C1",
                "delivery_id": f"message-{workspace_id}",
                "sequence": sequence,
                "event_type": "message.created",
                "message_id": "1.0",
                "sender_id": "U1",
                "text": workspace_id,
                "occurred_at": "2026-07-21T00:00:00+00:00",
                "provider_version": "0001",
            },
            f"hash-{workspace_id}",
        )
    sequence += 1
    result = store.record_envelope(
        {
            "project_id": project_id,
            "provider": "slack",
            "workspace_id": "T1",
            "delivery_id": "purge-T1",
            "sequence": sequence,
            "event_type": "workspace.purge",
            "occurred_at": "2026-07-21T01:00:00+00:00",
        },
        "hash-purge-T1",
    )

    with store._connect() as conn:
        rows = conn.execute(
            "SELECT workspace_id, text FROM messages ORDER BY workspace_id"
        ).fetchall()
        coverage = conn.execute(
            "SELECT workspace_id FROM coverage ORDER BY workspace_id"
        ).fetchall()
    assert result["status"] == "accepted"
    assert [tuple(row) for row in rows] == [("T2", "T2")]
    assert [tuple(row) for row in coverage] == [("T2",)]


def test_reconciliation_repairs_delete_and_reaction_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    cycle_id = "cycle-1"
    events = [
        {
            "event_type": "message.created",
            "conversation_id": "C1",
            "message_id": "Mkeep",
            "text": "keep",
            "occurred_at": "2026-07-21T00:00:01+00:00",
            "provider_version": "0001",
        },
        {
            "event_type": "message.created",
            "conversation_id": "C1",
            "message_id": "Mdeleted",
            "text": "deleted at provider",
            "occurred_at": "2026-07-21T00:00:02+00:00",
            "provider_version": "0002",
        },
        {
            "event_type": "reaction.added",
            "conversation_id": "C1",
            "message_id": "Mkeep",
            "reaction_name": "eyes",
            "actor_id": "U2",
            "occurred_at": "2026-07-21T00:00:03+00:00",
        },
        {
            "event_type": "reconciliation.started",
            "conversation_id": "C1",
            "reconciliation_cycle_id": cycle_id,
            "floor_at": "2026-07-21T00:00:00+00:00",
            "ceiling_at": "2026-07-21T00:01:00+00:00",
            "occurred_at": "2026-07-21T00:01:00+00:00",
        },
        {
            "event_type": "message.reconciled",
            "conversation_id": "C1",
            "message_id": "Mkeep",
            "text": "keep",
            "occurred_at": "2026-07-21T00:00:01+00:00",
            "provider_version": "0001",
            "reconciliation_cycle_id": cycle_id,
            "reconciled_at": "2026-07-21T00:01:00+00:00",
            "provider_payload": {"reactions": []},
        },
        {
            "event_type": "reconciliation.completed",
            "conversation_id": "C1",
            "reconciliation_cycle_id": cycle_id,
            "provider_version": "9999",
            "occurred_at": "2026-07-21T00:01:01+00:00",
            "completed_thread_ts": [],
        },
    ]
    for sequence, event in enumerate(events, start=1):
        store.record_envelope(
            {
                **event,
                "project_id": project_id,
                "provider": "slack",
                "workspace_id": "T1",
                "delivery_id": f"r{sequence}",
                "sequence": sequence,
            },
            f"hash-r{sequence}",
        )

    with store._connect() as conn:
        messages = dict(
            conn.execute(
                "SELECT provider_message_id, deleted_at FROM messages"
            ).fetchall()
        )
        reaction = conn.execute(
            "SELECT deleted_at FROM reactions WHERE provider_message_id = 'Mkeep'"
        ).fetchone()
        cycles = conn.execute("SELECT COUNT(*) FROM reconciliation_cycles").fetchone()[0]
        seen = conn.execute("SELECT COUNT(*) FROM reconciliation_seen").fetchone()[0]
    assert messages["Mkeep"] is None
    assert messages["Mdeleted"] is not None
    assert reaction[0] == "2026-07-21T00:01:00+00:00"
    assert cycles == 0
    assert seen == 0


def test_bounded_query_enforces_exact_acl_tuples_and_complete_coverage(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    sequence = 0

    def apply(event, *, workspace_id, conversation_id):
        nonlocal sequence
        sequence += 1
        store.record_envelope(
            {
                **event,
                "project_id": project_id,
                "provider": "slack",
                "workspace_id": workspace_id,
                "conversation_id": conversation_id,
                "delivery_id": f"query-{sequence}",
                "sequence": sequence,
            },
            f"hash-query-{sequence}",
        )

    for workspace_id, conversation_id, message_id in (
        ("T1", "C1", "allowed-one"),
        ("T2", "C2", "allowed-two"),
        ("T1", "C2", "cross-product-leak"),
    ):
        apply(
            {
                "event_type": "message.created",
                "message_id": message_id,
                "sender_id": "U1",
                "text": message_id,
                "occurred_at": "2026-07-21T00:10:00+00:00",
                "provider_version": "0001",
                "provider_payload": {"type": "message", "ts": "1.0"},
            },
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        apply(
            {
                "event_type": "coverage.completed",
                "contiguous_since": "2026-07-20T00:00:00+00:00",
                "occurred_at": "2026-07-21T00:20:00+00:00",
            },
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )

    result = store.query(
        {
            "operation": "recent_activity",
            "start": "2026-07-21T00:00:00+00:00",
            "end": "2026-07-21T01:00:00+00:00",
            "allowed_source_ids": ["slack:T1:C1", "slack:T2:C2"],
            "limit": 10,
        }
    )

    assert result["coverage_complete"] is True
    assert {row["provider_message_id"] for row in result["messages"]} == {
        "allowed-one",
        "allowed-two",
    }


def test_message_edit_preserves_history_time_and_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)

    def apply(sequence, event):
        store.record_envelope(
            {
                **event,
                "project_id": project_id,
                "provider": "slack",
                "workspace_id": "T1",
                "conversation_id": "C1",
                "delivery_id": f"edit-time-{sequence}",
                "sequence": sequence,
            },
            f"edit-time-hash-{sequence}",
        )

    apply(
        1,
        {
            "event_type": "message.created",
            "message_id": "M1",
            "text": "before",
            "occurred_at": "2026-07-21T00:10:00+00:00",
            "message_occurred_at": "2026-07-21T00:10:00+00:00",
            "provider_version": "0001",
        },
    )
    apply(
        2,
        {
            "event_type": "message.created",
            "message_id": "M2",
            "text": "newer",
            "occurred_at": "2026-07-21T00:20:00+00:00",
            "message_occurred_at": "2026-07-21T00:20:00+00:00",
            "provider_version": "0001",
        },
    )
    apply(
        3,
        {
            "event_type": "message.updated",
            "message_id": "M1",
            "text": "after",
            "occurred_at": "2026-07-21T00:30:00+00:00",
            "message_occurred_at": "2026-07-21T00:10:00+00:00",
            "edited_at": "2026-07-21T00:30:00+00:00",
            "provider_version": "0002",
        },
    )
    apply(
        4,
        {
            "event_type": "coverage.completed",
            "contiguous_since": "2026-07-21T00:00:00+00:00",
            "occurred_at": "2026-07-21T00:40:00+00:00",
        },
    )

    recent = store.query(
        {
            "operation": "recent_activity",
            "start": "2026-07-21T00:25:00+00:00",
            "end": "2026-07-21T01:00:00+00:00",
            "allowed_source_ids": ["slack:T1:C1"],
        }
    )
    history = store.query(
        {
            "operation": "fetch_history",
            "start": "2026-07-21T00:00:00+00:00",
            "end": "2026-07-21T01:00:00+00:00",
            "allowed_source_ids": ["slack:T1:C1"],
        }
    )

    assert recent["coverage_complete"] is True
    assert recent["messages"] == []
    assert [row["provider_message_id"] for row in history["messages"]] == [
        "M2",
        "M1",
    ]
    assert history["messages"][1]["text"] == "after"
    assert history["messages"][1]["occurred_at"] == "2026-07-21T00:10:00+00:00"
    assert history["messages"][1]["edited_at"] == "2026-07-21T00:30:00+00:00"


def test_bounded_query_falls_back_when_coverage_does_not_reach_requested_start(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    store.record_envelope(
        {
            "project_id": project_id,
            "provider": "slack",
            "workspace_id": "T1",
            "conversation_id": "C1",
            "delivery_id": "coverage-late",
            "sequence": 1,
            "event_type": "coverage.completed",
            "contiguous_since": "2026-07-21T00:30:00+00:00",
            "occurred_at": "2026-07-21T01:00:00+00:00",
        },
        "coverage-late-hash",
    )

    result = store.query(
        {
            "operation": "fetch_history",
            "start": "2026-07-21T00:00:00+00:00",
            "end": "2026-07-21T01:00:00+00:00",
            "allowed_source_ids": ["slack:T1:C1"],
        }
    )

    assert result == {
        "messages": [],
        "coverage_complete": False,
        "reason": "coverage_incomplete",
    }


def test_ingest_window_uses_stable_changed_at_cursor(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    with store._connect() as conn:
        for index, changed_at in enumerate(
            (
                "2026-07-21T00:03:00+00:00",
                "2026-07-21T00:02:00+00:00",
                "2026-07-21T00:01:00+00:00",
            ),
            start=1,
        ):
            conn.execute(
                "INSERT INTO messages(project_id, provider, workspace_id, "
                "conversation_id, provider_message_id, sender_id, text, occurred_at, "
                "inserted_at, updated_at) VALUES (?, 'slack', 'T1', 'C1', ?, 'U1', "
                "?, ?, ?, ?)",
                (project_id, f"M{index}", f"message-{index}", changed_at, changed_at, changed_at),
            )
        conn.execute(
            "INSERT INTO coverage(project_id, provider, workspace_id, conversation_id, "
            "contiguous_since, last_sequence, last_event_at, state) VALUES "
            "(?, 'slack', 'T1', 'C1', '2026-07-20T00:00:00+00:00', 3, "
            "'2026-07-21T00:03:00+00:00', 'COLLECTING')",
            (project_id,),
        )

    request = {
        "operation": "ingest_window",
        "start": "2026-07-21T00:00:00+00:00",
        "end": "2026-07-21T00:04:00+00:00",
        "providers": ["slack"],
        "workspace_ids": ["T1"],
        "limit": 2,
    }
    first = store.query(request)
    second = store.query({**request, "cursor": first["next_cursor"]})

    assert [row["provider_message_id"] for row in first["messages"]] == ["M1", "M2"]
    assert first["next_cursor"]["changed_at"] == "2026-07-21T00:02:00+00:00"
    assert [row["provider_message_id"] for row in second["messages"]] == ["M3"]
    assert "next_cursor" not in second


def test_reaction_advances_message_change_feed_time(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    base = {
        "project_id": project_id,
        "provider": "slack",
        "workspace_id": "T1",
        "conversation_id": "C1",
    }
    store.record_envelope(
        {
            **base,
            "delivery_id": "message",
            "sequence": 1,
            "event_type": "message.created",
            "message_id": "M1",
            "sender_id": "U1",
            "text": "hello",
            "occurred_at": "2026-07-20T00:00:00+00:00",
            "provider_version": "0001",
        },
        "hash-message",
    )
    with store._connect() as conn:
        before = conn.execute(
            "SELECT updated_at FROM messages WHERE provider_message_id='M1'"
        ).fetchone()[0]
    store.record_envelope(
        {
            **base,
            "delivery_id": "reaction",
            "sequence": 2,
            "event_type": "reaction.added",
            "message_id": "M1",
            "reaction_name": "eyes",
            "actor_id": "U2",
            # Provider time predates local ingestion; the local change clock
            # must still advance rather than being held back by this timestamp.
            "occurred_at": "2026-07-20T00:01:00+00:00",
        },
        "hash-reaction",
    )
    with store._connect() as conn:
        after = conn.execute(
            "SELECT updated_at FROM messages WHERE provider_message_id='M1'"
        ).fetchone()[0]
        conn.execute(
            "UPDATE coverage SET contiguous_since='2026-07-19T00:00:00+00:00' "
            "WHERE project_id=? AND workspace_id='T1' AND conversation_id='C1'",
            (project_id,),
        )

    result = store.query(
        {
            "operation": "ingest_window",
            "start": before,
            "end": after,
            "providers": ["slack"],
            "workspace_ids": ["T1"],
        }
    )
    assert after >= before
    assert result["messages"][0]["reactions"] == [
        {"name": "eyes", "count": 1, "users": ["U2"]}
    ]


def test_project_local_query_p95_is_below_200ms_on_pilot_fixture(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    _manager, module = _load_service()
    store = module.MessageStore(project_id)
    now = datetime.now(timezone.utc)
    rows = []
    for index in range(3000):
        occurred_at = (now - timedelta(seconds=index)).isoformat()
        rows.append(
            (
                project_id,
                "slack",
                "T1",
                f"C{index % 30:02d}",
                f"{now.timestamp() - index:.6f}",
                "U1",
                "fixture",
                "{}",
                occurred_at,
                occurred_at,
                occurred_at,
            )
        )
    with store._connect() as conn:
        conn.executemany(
            "INSERT INTO messages(project_id, provider, workspace_id, "
            "conversation_id, provider_message_id, sender_id, text, "
            "provider_payload_json, occurred_at, inserted_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        for index in range(30):
            conn.execute(
                "INSERT INTO coverage(project_id, provider, workspace_id, "
                "conversation_id, contiguous_since, last_sequence, "
                "last_event_at, state) VALUES (?, 'slack', 'T1', ?, ?, 1, ?, "
                "'COLLECTING')",
                (
                    project_id,
                    f"C{index:02d}",
                    (now - timedelta(days=7)).isoformat(),
                    now.isoformat(),
                ),
            )
    request = {
        "operation": "recent_activity",
        "start": (now - timedelta(hours=24)).isoformat(),
        "end": now.isoformat(),
        "providers": ["slack"],
        "workspace_ids": ["T1"],
        "allowed_source_ids": [f"slack:T1:C{index:02d}" for index in range(30)],
        "limit": 240,
        "per_conversation": 8,
    }

    store.query(request)  # warm SQLite page cache
    durations = []
    for _ in range(25):
        # The full suite runs test files in parallel on shared CI runners. Use
        # process CPU time so scheduler contention does not masquerade as a
        # local SQLCipher query regression.
        started = time.process_time()
        result = store.query(request)
        durations.append((time.process_time() - started) * 1000)

    p95 = sorted(durations)[int(len(durations) * 0.95) - 1]
    assert result["coverage_complete"] is True
    assert len(result["messages"]) == 240
    assert p95 < 200, f"local query CPU p95 was {p95:.1f}ms"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are platform-specific")
@pytest.mark.asyncio
async def test_private_key_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager, _module = _load_service()
    project_id = str(uuid.uuid4())
    write_project_marker(project_id)
    await manager.invoke_hook_async("project_claimed", project_id=project_id)
    mode = (tmp_path / "state/keys/message-store-v1.pem").stat().st_mode & 0o777
    assert mode == 0o600

"""Authenticated encryption for canonical entity ``SKILL.md`` files."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


SCHEMA_VERSION = 1
KEYRING_ENV = "RINGO_MESSAGE_STORE_DB_KEYS"
ACTIVE_KEY_VERSION_ENV = "RINGO_MESSAGE_STORE_DB_KEY_VERSION"


class EncryptedSkillStoreError(RuntimeError):
    """The entity store cannot be opened without weakening confidentiality."""


def _keyring() -> tuple[str, dict[str, bytes]]:
    try:
        raw = json.loads(os.environ[KEYRING_ENV])
        active = str(int(os.environ[ACTIVE_KEY_VERSION_ENV]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EncryptedSkillStoreError("entity skill encryption keyring unavailable") from exc
    if not isinstance(raw, dict) or active not in raw:
        raise EncryptedSkillStoreError("active entity skill encryption key unavailable")
    keys: dict[str, bytes] = {}
    try:
        for version, encoded in raw.items():
            key = bytes.fromhex(str(encoded))
            if len(key) != 32:
                raise ValueError
            keys[str(int(version))] = key
    except (TypeError, ValueError) as exc:
        raise EncryptedSkillStoreError("entity skill encryption keyring invalid") from exc
    return active, keys


def _aad(
    project_id: str,
    key_version: str,
    kind: str,
    entity_id: str,
) -> bytes:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "project_id": project_id,
            "key_version": int(key_version),
            "kind": kind,
            "entity_id": entity_id,
            "purpose": "ringo-entity-skill",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class EncryptedSkillStore:
    """Store ciphertext at the canonical virtual ``SKILL.md`` path."""

    def __init__(self, skills_root: Path):
        self.skills_root = Path(skills_root).resolve()

    def path(self, kind: str, entity_id: str) -> Path:
        path = (self.skills_root / kind / entity_id / "SKILL.md").resolve()
        if not path.is_relative_to(self.skills_root):
            raise EncryptedSkillStoreError("entity skill path escapes root")
        return path

    def get(self, project_id: str, kind: str, entity_id: str) -> str | None:
        path = self.path(kind, entity_id)
        active, keys = _keyring()
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            raise EncryptedSkillStoreError("entity SKILL.md is not encrypted") from exc
        try:
            version = str(int(envelope["key_version"]))
            if (
                envelope.get("schema_version") != SCHEMA_VERSION
                or envelope.get("algorithm") != "AES-256-GCM"
                or str(envelope.get("project_id") or "") != project_id
                or str(envelope.get("kind") or "") != kind
                or str(envelope.get("entity_id") or "") != entity_id
                or version not in keys
            ):
                raise ValueError
            nonce = base64.b64decode(envelope["nonce"], validate=True)
            ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
            content = AESGCM(keys[version]).decrypt(
                nonce,
                ciphertext,
                _aad(project_id, version, kind, entity_id),
            ).decode("utf-8")
        except (
            InvalidTag,
            KeyError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise EncryptedSkillStoreError(
                "encrypted entity SKILL.md authentication failed"
            ) from exc
        if version != active:
            self.put(project_id, kind, entity_id, content)
        return content

    def put(
        self,
        project_id: str,
        kind: str,
        entity_id: str,
        content: str,
    ) -> None:
        active, keys = _keyring()
        nonce = os.urandom(12)
        ciphertext = AESGCM(keys[active]).encrypt(
            nonce,
            content.encode("utf-8"),
            _aad(project_id, active, kind, entity_id),
        )
        envelope = json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "project_id": project_id,
                "kind": kind,
                "entity_id": entity_id,
                "key_version": int(active),
                "algorithm": "AES-256-GCM",
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.path(kind, entity_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".ringo-tmp")
        try:
            with tmp.open("wb") as handle:
                os.chmod(tmp, 0o600)
                handle.write(envelope)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            os.chmod(path, 0o600)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

"""SQLCipher connection, plaintext migration, and key rotation."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sqlcipher3 import dbapi2 as sqlcipher


KEYRING_ENV = "RINGO_MESSAGE_STORE_DB_KEYS"
ACTIVE_KEY_VERSION_ENV = "RINGO_MESSAGE_STORE_DB_KEY_VERSION"
_PLAINTEXT_HEADER = b"SQLite format 3\x00"
_RAW_KEY_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class MessageStoreEncryptionError(RuntimeError):
    """The encrypted store cannot be opened without weakening encryption."""


@dataclass(frozen=True)
class DatabaseKeyring:
    keys: dict[int, str]
    active_version: int

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] | None = None
    ) -> "DatabaseKeyring":
        source = environ if environ is not None else os.environ
        raw = str(source.get(KEYRING_ENV) or "").strip()
        active_raw = str(source.get(ACTIVE_KEY_VERSION_ENV) or "").strip()
        if not raw or not active_raw:
            raise MessageStoreEncryptionError(
                f"{KEYRING_ENV} and {ACTIVE_KEY_VERSION_ENV} are required"
            )
        try:
            parsed = json.loads(raw)
            active_version = int(active_raw)
        except (TypeError, ValueError) as exc:
            raise MessageStoreEncryptionError(
                "message store database key configuration is invalid"
            ) from exc
        if not isinstance(parsed, dict) or active_version < 1:
            raise MessageStoreEncryptionError(
                "message store database key configuration is invalid"
            )
        keys: dict[int, str] = {}
        for version_raw, key_raw in parsed.items():
            try:
                version = int(version_raw)
            except (TypeError, ValueError) as exc:
                raise MessageStoreEncryptionError(
                    "message store database key versions must be positive integers"
                ) from exc
            key = str(key_raw).strip().lower()
            if version < 1 or not _RAW_KEY_RE.fullmatch(key):
                raise MessageStoreEncryptionError(
                    "message store database keys must be 32-byte hexadecimal values"
                )
            keys[version] = key
        if active_version not in keys:
            raise MessageStoreEncryptionError(
                "active message store database key version is unavailable"
            )
        return cls(keys=keys, active_version=active_version)


class EncryptedDatabase:
    def __init__(self, path: Path, *, environ: Mapping[str, str] | None = None):
        self.path = path
        self.keyring = DatabaseKeyring.from_environ(environ)
        self.active_key_version = self.keyring.active_version
        self.opened_key_version = self.active_key_version
        self.migration_status = "not_needed"
        self.integrity_status = "unknown"
        self.cipher_version = "unknown"

    @property
    def active_key(self) -> str:
        return self.keyring.keys[self.active_key_version]

    def prepare(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._is_plaintext_database():
            self._migrate_plaintext()
            self.migration_status = "migrated"
        elif not self.path.exists() or self.path.stat().st_size == 0:
            with self._connect_with_key(self.active_key):
                pass
            self.migration_status = "created"
        else:
            opened_version = self._find_working_key_version()
            self.opened_key_version = opened_version
            if opened_version != self.active_key_version:
                self._rekey(self.keyring.keys[opened_version], self.active_key)
                self.migration_status = "rekeyed"
                self.opened_key_version = self.active_key_version
        self._verify_integrity()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def connect(self):
        return self._connect_with_key(self.active_key)

    def _connect_with_key(self, key: str):
        conn = sqlcipher.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlcipher.Row
        try:
            self._apply_key(conn, key)
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except Exception:
            conn.close()
            raise

    @staticmethod
    def _apply_key(conn: Any, key: str) -> None:
        # key is exactly 64 hex chars, validated before interpolation.
        conn.execute(f'PRAGMA key = "x\'{key}\'"')
        conn.execute("PRAGMA cipher_memory_security=ON")

    def _find_working_key_version(self) -> int:
        ordered = [self.active_key_version] + sorted(
            (version for version in self.keyring.keys if version != self.active_key_version),
            reverse=True,
        )
        for version in ordered:
            try:
                with self._connect_with_key(self.keyring.keys[version]):
                    return version
            except sqlcipher.DatabaseError:
                continue
        raise MessageStoreEncryptionError(
            "message store database cannot be opened by any configured key version"
        )

    def _is_plaintext_database(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size < len(_PLAINTEXT_HEADER):
            return False
        with self.path.open("rb") as handle:
            return handle.read(len(_PLAINTEXT_HEADER)) == _PLAINTEXT_HEADER

    def _migrate_plaintext(self) -> None:
        # Fold any plaintext WAL into the main file before exporting.
        with sqlite3.connect(str(self.path), timeout=5.0) as plain:
            plain.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            user_version = int(plain.execute("PRAGMA user_version").fetchone()[0])

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".sqlcipher", dir=self.path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        source = sqlcipher.connect(str(self.path), timeout=5.0)
        try:
            source.execute("SELECT count(*) FROM sqlite_master").fetchone()
            source.execute(
                f'ATTACH DATABASE ? AS encrypted KEY "x\'{self.active_key}\'"',
                (str(temporary),),
            )
            source.execute("SELECT sqlcipher_export('encrypted')")
            source.execute(f"PRAGMA encrypted.user_version={user_version}")
            source.execute("DETACH DATABASE encrypted")
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            source.close()

        try:
            probe = EncryptedDatabase(temporary, environ={
                KEYRING_ENV: json.dumps({str(self.active_key_version): self.active_key}),
                ACTIVE_KEY_VERSION_ENV: str(self.active_key_version),
            })
            probe._verify_integrity()
            os.replace(temporary, self.path)
            for suffix in ("-wal", "-shm"):
                sidecar = self.path.with_name(self.path.name + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            self._fsync_parent()
        finally:
            if temporary.exists():
                temporary.unlink()

    def _rekey(self, old_key: str, new_key: str) -> None:
        with self._connect_with_key(old_key) as conn:
            # SQLCipher rekey rewrites database pages. Move out of WAL first so
            # no old-key WAL frames survive the rewrite and fail HMAC checks.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            conn.execute("PRAGMA journal_mode=DELETE").fetchone()
            conn.execute(f'PRAGMA rekey = "x\'{new_key}\'"')
        try:
            with self._connect_with_key(new_key):
                pass
        except Exception as exc:
            raise MessageStoreEncryptionError(
                "message store database key rotation verification failed"
            ) from exc

    def _verify_integrity(self) -> None:
        try:
            with self._connect_with_key(self.active_key) as conn:
                cipher_row = conn.execute("PRAGMA cipher_version").fetchone()
                self.cipher_version = str(cipher_row[0]) if cipher_row else "unknown"
                cipher_errors = conn.execute("PRAGMA cipher_integrity_check").fetchall()
                integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if cipher_errors or not integrity or str(integrity[0]).lower() != "ok":
                raise MessageStoreEncryptionError(
                    "message store database integrity check failed"
                )
            self.integrity_status = "ok"
        except MessageStoreEncryptionError:
            self.integrity_status = "failed"
            raise
        except Exception as exc:
            self.integrity_status = "failed"
            raise MessageStoreEncryptionError(
                "message store database integrity check failed"
            ) from exc

    def _fsync_parent(self) -> None:
        try:
            descriptor = os.open(str(self.path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

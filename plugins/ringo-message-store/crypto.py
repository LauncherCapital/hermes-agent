"""Project-local encryption-key lifecycle."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from hermes_constants import get_hermes_home


KEY_VERSION = 1


def keys_dir() -> Path:
    path = get_hermes_home() / "state" / "keys"
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def private_key_path(version: int = KEY_VERSION) -> Path:
    return keys_dir() / f"message-store-v{int(version)}.pem"


def public_key_path(version: int = KEY_VERSION) -> Path:
    return keys_dir() / f"message-store-v{int(version)}.pub.pem"


def metadata_path(version: int = KEY_VERSION) -> Path:
    return keys_dir() / f"message-store-v{int(version)}.json"


def ensure_project_encryption_key(project_id: str, version: int = KEY_VERSION) -> dict:
    """Create the key only after a durable project claim marker exists."""
    from gateway.event_ingress import read_project_marker

    marker = read_project_marker()
    if marker is None or marker.get("project_id") != str(project_id):
        raise RuntimeError("project Volume is not claimed")

    version = int(version)
    if version < 1:
        raise ValueError("key version must be positive")
    private_path = private_key_path(version)
    public_path = public_key_path(version)
    meta_path = metadata_path(version)
    if private_path.exists() and public_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError("project key metadata is missing or invalid") from exc
        if (
            str(metadata.get("project_id") or "") != str(project_id)
            or int(metadata.get("version") or 0) != version
        ):
            raise RuntimeError("project key metadata does not match the Volume claim")
        return {
            "version": version,
            "private_path": str(private_path),
            "public_key": public_path.read_text(encoding="utf-8"),
        }

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _atomic_write(private_path, private_pem, 0o600)
    _atomic_write(public_path, public_pem, 0o644)
    _atomic_write(
        meta_path,
        json.dumps(
            {"project_id": str(project_id), "version": version},
            sort_keys=True,
        ).encode("utf-8"),
        0o600,
    )
    return {
        "version": version,
        "private_path": str(private_path),
        "public_key": public_pem.decode("utf-8"),
    }


def build_recovery_registration(
    project_id: str,
    *,
    version: int | None = None,
    recovery_key_id: str | None = None,
) -> dict | None:
    """Encrypt a recoverable private-key copy locally for central registration."""
    from gateway.event_ingress import read_project_marker

    marker = read_project_marker()
    if marker is None or marker.get("project_id") != str(project_id):
        raise RuntimeError("project Volume is not claimed")
    recovery_keys = marker.get("recovery_public_keys") or {}
    if not isinstance(recovery_keys, dict) or not recovery_keys:
        return None
    key_id = recovery_key_id or marker.get("active_recovery_key_id")
    if not key_id:
        key_id = sorted(str(item) for item in recovery_keys)[-1]
    recovery_pem = recovery_keys.get(key_id)
    if not recovery_pem:
        raise RuntimeError("active recovery public key is unavailable")
    try:
        recovery_key = serialization.load_pem_public_key(
            str(recovery_pem).encode("utf-8")
        )
    except Exception as exc:
        raise RuntimeError("recovery public key is invalid") from exc
    if not isinstance(recovery_key, rsa.RSAPublicKey) or recovery_key.key_size < 3072:
        raise RuntimeError("recovery public key must be RSA-3072 or stronger")

    active_version = int(version or marker.get("active_key_version") or KEY_VERSION)
    project_key = ensure_project_encryption_key(project_id, active_version)
    aad_fields = {
        "schema_version": 1,
        "project_id": str(project_id),
        "key_version": active_version,
        "recovery_key_id": str(key_id),
    }
    aad = json.dumps(aad_fields, separators=(",", ":"), sort_keys=True).encode("utf-8")
    dek = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    private_pem = Path(project_key["private_path"]).read_bytes()
    ciphertext = AESGCM(dek).encrypt(nonce, private_pem, aad)
    wrapped_dek = recovery_key.encrypt(
        dek,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    recovery_envelope = json.dumps(
        {
            **aad_fields,
            "algorithm": "AES-256-GCM",
            "key_wrap": "RSA-OAEP-SHA256",
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "wrapped_dek": base64.b64encode(wrapped_dek).decode("ascii"),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "version": active_version,
        "public_key_pem": project_key["public_key"],
        "wrapped_recovery_copy_b64": base64.b64encode(recovery_envelope).decode("ascii"),
        "recovery_key_id": str(key_id),
        "activate": True,
    }


def decrypt_delivery_envelope(envelope: dict) -> dict:
    """Decrypt one IE delivery and bind its clear event to authenticated routing AAD."""
    project_id = str(envelope.get("project_id") or "")
    version = int(envelope.get("key_version") or 0)
    aad_fields = {
        "project_id": project_id,
        "provider": str(envelope.get("provider") or ""),
        "workspace_id": str(envelope.get("workspace_id") or ""),
        "delivery_id": str(envelope.get("delivery_id") or ""),
        "sequence": int(envelope.get("sequence") or 0),
        "schema_version": int(envelope.get("schema_version") or 0),
        "key_version": version,
    }
    if (
        not project_id
        or not aad_fields["provider"]
        or not aad_fields["workspace_id"]
        or not aad_fields["delivery_id"]
        or aad_fields["sequence"] < 1
        or aad_fields["schema_version"] != 1
        or version < 1
    ):
        raise ValueError("encrypted delivery metadata is invalid")
    encryption = envelope.get("encryption")
    if not isinstance(encryption, dict):
        raise ValueError("encrypted delivery payload is missing")
    if (
        encryption.get("algorithm") != "AES-256-GCM"
        or encryption.get("key_wrap") != "RSA-OAEP-SHA256"
    ):
        raise ValueError("encrypted delivery algorithm is unsupported")
    try:
        nonce = base64.b64decode(encryption["nonce"], validate=True)
        ciphertext = base64.b64decode(encryption["ciphertext"], validate=True)
        wrapped_dek = base64.b64decode(encryption["wrapped_dek"], validate=True)
    except Exception as exc:
        raise ValueError("encrypted delivery encoding is invalid") from exc
    ciphertext_hash = hashlib.sha256(ciphertext).hexdigest()
    if ciphertext_hash != str(envelope.get("ciphertext_sha256") or ""):
        raise ValueError("encrypted delivery ciphertext hash does not match")

    from gateway.event_ingress import read_project_marker

    marker = read_project_marker()
    if marker is None or marker.get("project_id") != project_id:
        raise ValueError("encrypted delivery targets another project")
    key_path = private_key_path(version)
    meta_path = metadata_path(version)
    if not key_path.exists() or not meta_path.exists():
        raise ValueError("encrypted delivery key version is unavailable")
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            metadata.get("project_id") != project_id
            or int(metadata.get("version") or 0) != version
        ):
            raise ValueError("project key metadata mismatch")
        private_key = serialization.load_pem_private_key(
            key_path.read_bytes(), password=None
        )
        dek = private_key.decrypt(
            wrapped_dek,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        aad = json.dumps(
            aad_fields, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        plaintext = AESGCM(dek).decrypt(nonce, ciphertext, aad)
        decoded = json.loads(plaintext)
    except Exception as exc:
        raise ValueError("encrypted delivery authentication failed") from exc
    event = decoded.get("event") if isinstance(decoded, dict) else None
    if not isinstance(event, dict):
        raise ValueError("encrypted delivery event is invalid")
    event_json = json.dumps(event, separators=(",", ":"), sort_keys=True).encode()
    if hashlib.sha256(event_json).hexdigest() != decoded.get("plaintext_sha256"):
        raise ValueError("encrypted delivery plaintext hash does not match")
    clear = dict(event)
    for field, value in aad_fields.items():
        if field in clear and str(clear[field]) != str(value):
            raise ValueError(f"encrypted event conflicts with authenticated {field}")
        clear[field] = value
    clear["event_type"] = str(envelope.get("event_type") or clear.get("event_type") or "")
    clear["event_id"] = str(
        envelope.get("provider_event_id") or clear.get("event_id") or ""
    )
    clear["payload_hash"] = ciphertext_hash
    return clear


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    os.chmod(path, mode)

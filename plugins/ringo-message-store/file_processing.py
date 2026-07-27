"""Bounded, local-only Slack file processing for the encrypted file index."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 100_000
STALE_TEMP_SECONDS = 6 * 60 * 60
CAPTION_PROMPT_VERSION = "file-index-v1"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".rst",
    ".text",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
_RETRYABLE_ERROR_CODES = {
    "embedding_failed",
    "file_download_failed",
    "image_caption_failed",
    "image_caption_invalid",
    "provider_rate_limited",
    "provider_unavailable",
    "slack_file_info_failed",
}


class FileProcessingError(RuntimeError):
    """A content processing failure represented by a non-secret error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _temp_dir() -> Path:
    path = get_hermes_home() / "state" / "tmp" / "file-index"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def cleanup_stale_temp_files(*, now: float | None = None) -> int:
    """Remove abandoned raw downloads left by a killed process."""
    cutoff = (time.time() if now is None else now) - STALE_TEMP_SECONDS
    removed = 0
    for path in _temp_dir().glob("fi-*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _slack_file_info(file_id: str, access_token: str) -> dict[str, Any]:
    try:
        response = httpx.get(
            "https://slack.com/api/files.info",
            params={"file": file_id},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15.0,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        code = (
            "provider_rate_limited"
            if status == 429
            else "provider_unavailable"
            if status >= 500
            else "slack_file_unavailable"
        )
        raise FileProcessingError(code) from None
    except Exception as exc:
        logger.warning("Slack file metadata request failed: %s", type(exc).__name__)
        raise FileProcessingError("slack_file_info_failed") from None
    if not isinstance(payload, dict) or not payload.get("ok"):
        error = str(payload.get("error") or "") if isinstance(payload, dict) else ""
        raise FileProcessingError(
            "provider_rate_limited"
            if error == "ratelimited"
            else "slack_file_unavailable"
        )
    file_data = payload.get("file")
    if not isinstance(file_data, dict):
        raise FileProcessingError("slack_file_info_invalid")
    return file_data


def _download(
    *,
    url: str,
    access_token: str,
    expected_size: int | None,
) -> tuple[Path, str, int]:
    if expected_size is not None and expected_size > MAX_DOWNLOAD_BYTES:
        raise FileProcessingError("file_too_large")
    parsed = urlparse(url)
    host = str(parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not host
        or not (
            host == "slack.com"
            or host.endswith(".slack.com")
            or host == "slack-files.com"
            or host.endswith(".slack-files.com")
        )
    ):
        raise FileProcessingError("file_download_host_invalid")
    temp = tempfile.NamedTemporaryFile(
        prefix="fi-",
        dir=_temp_dir(),
        delete=False,
    )
    path = Path(temp.name)
    digest = hashlib.sha256()
    total = 0
    try:
        with temp:
            with httpx.stream(
                "GET",
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=False,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(64 * 1024):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise FileProcessingError("file_too_large")
                    digest.update(chunk)
                    temp.write(chunk)
        path.chmod(0o600)
        return path, digest.hexdigest(), total
    except FileProcessingError:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        logger.warning("Slack file download failed: %s", type(exc).__name__)
        raise FileProcessingError("file_download_failed") from None


def _is_image(mime_type: str) -> bool:
    return mime_type.lower().startswith("image/")


def _is_text(mime_type: str, file_name: str) -> bool:
    return mime_type.lower().startswith("text/") or Path(file_name).suffix.lower() in _TEXT_SUFFIXES


def _extract_text(path: Path) -> str:
    raw = path.read_bytes()[: MAX_TEXT_CHARS * 4]
    return raw.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS].strip()


def _caption_image(path: Path) -> tuple[str, str]:
    from tools.vision_tools import vision_analyze_tool

    model = os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None
    prompt = (
        "Describe this workplace file precisely for search. Include all visible "
        "text (OCR), names, product or project terms, logos, colors, layout, and "
        "what the image is for. Return concise plain text only."
    )
    raw = asyncio.run(vision_analyze_tool(str(path), prompt, model))
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise FileProcessingError("image_caption_invalid") from None
    if not result.get("success") or not str(result.get("analysis") or "").strip():
        raise FileProcessingError("image_caption_failed")
    return str(result["analysis"]).strip()[:MAX_TEXT_CHARS], model or "auxiliary-default"


def embed_text(text: str) -> tuple[list[float], str]:
    if not text.strip():
        raise FileProcessingError("empty_embedding_input")
    model = (
        os.getenv("FILE_INDEX_EMBEDDING_MODEL", "").strip()
        or DEFAULT_EMBEDDING_MODEL
    )
    try:
        from agent.auxiliary_client import resolve_provider_client

        client, _ = resolve_provider_client("openrouter", async_mode=False)
        if client is None:
            raise RuntimeError("embedding client unavailable")
        response = client.embeddings.create(model=model, input=[text[:MAX_TEXT_CHARS]])
        vector = [float(value) for value in response.data[0].embedding]
    except Exception as exc:
        logger.warning("File embedding failed: %s", type(exc).__name__)
        raise FileProcessingError("embedding_failed") from None
    if not vector:
        raise FileProcessingError("embedding_empty")
    return vector, model


def process_slack_file(
    file_id: str,
    access_token: str,
    reuse_by_sha: Callable[[str, str], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Download one file transiently and return encrypted-store-ready derivatives."""
    cleanup_stale_temp_files()
    try:
        file_data = _slack_file_info(file_id, access_token)
    except FileProcessingError as exc:
        return {
            "processing_status": (
                "metadata_only"
                if exc.code in _RETRYABLE_ERROR_CODES
                else "unavailable"
            ),
            "last_error_code": exc.code,
        }
    file_name = str(file_data.get("name") or file_data.get("title") or "")
    mime_type = str(file_data.get("mimetype") or "")
    size_value = file_data.get("size")
    byte_size = (
        int(size_value)
        if isinstance(size_value, (int, float)) and not isinstance(size_value, bool)
        else None
    )
    download_url = str(
        file_data.get("url_private_download")
        or file_data.get("url_private")
        or ""
    )
    metadata = {
        "file_name": file_name or None,
        "mime_type": mime_type or None,
        "byte_size": byte_size,
        "uploaded_at": str(
            file_data.get("timestamp") or file_data.get("created") or ""
        )
        or None,
        "permalink": str(file_data.get("permalink") or "") or None,
    }
    if not download_url:
        return {
            **metadata,
            "processing_status": "unavailable",
            "last_error_code": "file_download_url_missing",
        }

    path: Path | None = None
    try:
        path, digest, actual_size = _download(
            url=download_url,
            access_token=access_token,
            expected_size=byte_size,
        )
        base = {
            **metadata,
            "byte_size": actual_size,
            "content_sha256": digest,
        }
        reused = reuse_by_sha(digest, mime_type) if reuse_by_sha else None
        if reused:
            return {
                **base,
                **reused,
                "processing_status": "indexed",
                "last_error_code": None,
            }
        if _is_image(mime_type):
            caption, caption_model = _caption_image(path)
            vector, embedding_model = embed_text(caption)
            return {
                **base,
                "processing_status": "indexed",
                "caption_ocr": caption,
                "text_content_embedding": vector,
                "image_embedding": vector,
                "caption_model": caption_model,
                "caption_prompt_version": CAPTION_PROMPT_VERSION,
                "text_embedding_model": embedding_model,
                "text_embedding_dimension": len(vector),
                "image_embedding_model": f"caption-text:{embedding_model}",
                "image_embedding_dimension": len(vector),
                "last_error_code": None,
            }
        if _is_text(mime_type, file_name):
            text = _extract_text(path)
            if not text:
                return {
                    **base,
                    "processing_status": "unsupported",
                    "last_error_code": "empty_text_file",
                }
            vector, embedding_model = embed_text(text)
            return {
                **base,
                "processing_status": "indexed",
                "caption_ocr": text,
                "text_content_embedding": vector,
                "text_embedding_model": embedding_model,
                "text_embedding_dimension": len(vector),
                "last_error_code": None,
            }
        return {
            **base,
            "processing_status": "unsupported",
            "last_error_code": "unsupported_mime_type",
        }
    except FileProcessingError as exc:
        return {
            **metadata,
            "processing_status": (
                "unsupported"
                if exc.code in {"file_too_large"}
                else "metadata_only"
                if exc.code in _RETRYABLE_ERROR_CODES
                else "unavailable"
            ),
            "last_error_code": exc.code,
        }
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def inspect_slack_image(
    file_id: str,
    access_token: str,
    query: str,
) -> str:
    """Re-check access and inspect one shortlisted image against the query."""
    cleanup_stale_temp_files()
    file_data = _slack_file_info(file_id, access_token)
    mime_type = str(file_data.get("mimetype") or "")
    if not _is_image(mime_type):
        raise FileProcessingError("not_an_image")
    size_value = file_data.get("size")
    byte_size = (
        int(size_value)
        if isinstance(size_value, (int, float)) and not isinstance(size_value, bool)
        else None
    )
    download_url = str(
        file_data.get("url_private_download")
        or file_data.get("url_private")
        or ""
    )
    if not download_url:
        raise FileProcessingError("file_download_url_missing")
    path: Path | None = None
    try:
        path, _digest, _actual_size = _download(
            url=download_url,
            access_token=access_token,
            expected_size=byte_size,
        )
        from tools.vision_tools import vision_analyze_tool

        # Keep the user query out of the shared vision tool's debug/log surface.
        # The caller compares this fresh, generic inspection with the query.
        prompt = (
            "Describe the visible workplace-search evidence in this image, "
            "including OCR text, names, logos, colors, layout, and purpose. "
            "Return concise plain text and do not speculate."
        )
        model = os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None
        raw = asyncio.run(vision_analyze_tool(str(path), prompt, model))
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            raise FileProcessingError("image_inspection_invalid") from None
        analysis = str(result.get("analysis") or "").strip()
        if not result.get("success") or not analysis:
            raise FileProcessingError("image_inspection_failed")
        return analysis[:MAX_TEXT_CHARS]
    finally:
        if path is not None:
            path.unlink(missing_ok=True)

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .config import (
    COMPUTE_TYPES,
    DEVICES,
    OUTPUT_FORMATS,
    ComputeType,
    Device,
    OutputFormat,
    Settings,
)
from .runner import RunResult, WhisperXRunner

logger = logging.getLogger("whisperx.service")


@dataclass
class TranscribeParams:
    file_bytes: bytes
    filename: str

    model: str | None = None
    language: str | None = None
    output_format: OutputFormat | None = None
    diarize: bool | None = None
    batch_size: int | None = None
    device: Device | None = None
    compute_type: ComputeType | None = None
    diarize_model: str | None = None

    hf_token: str | None = None

    # from JSON config field
    config: dict[str, Any] | None = None


@dataclass
class TranscribeResult:
    content: bytes
    media_type: str


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _norm_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _cfg_get_str(cfg: dict[str, Any], key: str) -> str | None:
    if key not in cfg:
        return None
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail={"error": "invalid config json", "message": f"`{key}` must be string"})
    return _norm_str(value)


def _cfg_get_bool(cfg: dict[str, Any], key: str) -> bool | None:
    if key not in cfg:
        return None
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise HTTPException(status_code=400, detail={"error": "invalid config json", "message": f"`{key}` must be boolean"})
    return value


def _cfg_get_int(cfg: dict[str, Any], key: str) -> int | None:
    if key not in cfg:
        return None
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise HTTPException(status_code=400, detail={"error": "invalid config json", "message": f"`{key}` must be integer"})
    return value


def _cfg_get_literal(cfg: dict[str, Any], key: str, allowed: set[str]) -> str | None:
    value = _cfg_get_str(cfg, key)
    if value is None:
        return None
    if value not in allowed:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid config json", "message": f"`{key}` must be one of: {sorted(allowed)}"},
        )
    return value


def _resolve_hf_token(
    hf_token_form: str | None,
    hf_token_header: str | None,
    authorization: str | None,
) -> str | None:
    if hf_token_form:
        return hf_token_form
    if hf_token_header:
        return hf_token_header
    if authorization:
        prefix = "bearer "
        if authorization.lower().startswith(prefix):
            token = authorization[len(prefix):].strip()
            return token if token else None
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return env_token if env_token else None


def _whisperx_cmd() -> list[str]:
    whisperx = shutil.which("whisperx")
    if whisperx:
        return [whisperx]
    return ["python", "-m", "whisperx"]


def _find_output_file(output_dir: Path, output_format: str) -> Path:
    matches = sorted(output_dir.glob(f"*.{output_format}"))
    if not matches:
        raise FileNotFoundError(f"No *.{output_format} in {output_dir}")
    if len(matches) == 1:
        return matches[0]
    return max(matches, key=lambda p: p.stat().st_size)


def _media_type_for(fmt: str) -> str:
    if fmt == "vtt":
        return "text/vtt; charset=utf-8"
    if fmt == "json":
        return "application/json; charset=utf-8"
    return "text/plain; charset=utf-8"


def _sanitize_tail(tail: str) -> str:
    """Маскуємо системні шляхи у виводі помилок."""
    import re
    tail = re.sub(r"(/tmp/whisperx_api_)\w+", r"\1***", tail)
    tail = re.sub(r"(/home/\w+)", "/home/***", tail)
    return tail


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------


class TranscriptionService:
    """Основна бізнес-логіка транскрипції з GPU-блокуванням."""

    def __init__(self, settings: Settings, runner: WhisperXRunner) -> None:
        self._settings = settings
        self._runner = runner
        self._gpu_lock = threading.Lock()

    def transcribe(
        self,
        params: TranscribeParams,
        hf_token_header: str | None,
        authorization: str | None,
    ) -> TranscribeResult:
        cfg = params.config or {}
        s = self._settings

        # --- resolve parameters (form > config json > env default) ---
        cfg_model = _cfg_get_str(cfg, "model")
        cfg_language = _cfg_get_str(cfg, "language")
        cfg_output_format = _cfg_get_literal(cfg, "output_format", OUTPUT_FORMATS)
        cfg_diarize = _cfg_get_bool(cfg, "diarize")
        cfg_batch_size = _cfg_get_int(cfg, "batch_size")
        cfg_device = _cfg_get_literal(cfg, "device", DEVICES)
        cfg_compute_type = _cfg_get_literal(cfg, "compute_type", COMPUTE_TYPES)
        cfg_hf_token = _cfg_get_str(cfg, "hf_token")
        cfg_diarize_model = _cfg_get_str(cfg, "diarize_model")

        resolved_model = _norm_str(params.model) or cfg_model or s.model
        resolved_language = _norm_str(params.language) or cfg_language or _norm_str(s.language)
        resolved_output_format: str = params.output_format or cfg_output_format or s.output_format
        resolved_batch_size = params.batch_size if params.batch_size is not None else (cfg_batch_size if cfg_batch_size is not None else s.batch_size)
        resolved_device: str = params.device or cfg_device or s.device
        resolved_compute_type: str = params.compute_type or cfg_compute_type or s.compute_type
        resolved_diarize_model = _norm_str(params.diarize_model) or cfg_diarize_model or s.diarize_model

        hf_token_input = _norm_str(params.hf_token) or cfg_hf_token
        resolved_hf_token = _resolve_hf_token(hf_token_input, hf_token_header, authorization)

        resolved_diarize = params.diarize if params.diarize is not None else (cfg_diarize if cfg_diarize is not None else False)
        if resolved_diarize and not resolved_hf_token:
            raise HTTPException(
                status_code=400,
                detail="`diarize=true` требует HF token (form `hf_token`, header `X-HF-Token` або env `HF_TOKEN`).",
            )

        # --- save uploaded file ---
        with tempfile.TemporaryDirectory(prefix="whisperx_api_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            output_dir = tmp_dir / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            safe_name = os.path.basename(params.filename)
            if not safe_name.strip("."):
                safe_name = "audio"
            input_path = tmp_dir / safe_name

            input_path.write_bytes(params.file_bytes)

            # --- build CLI command ---
            cmd: list[str] = [
                *_whisperx_cmd(),
                "--device", resolved_device,
                "--compute_type", resolved_compute_type,
                "--model", resolved_model,
                "--output_format", resolved_output_format,
                "--output_dir", str(output_dir),
                "--batch_size", str(resolved_batch_size),
            ]
            if resolved_language:
                cmd.extend(["--language", resolved_language])
            if resolved_diarize:
                cmd.append("--diarize")
                cmd.extend(["--diarize_model", resolved_diarize_model])
            if resolved_hf_token:
                cmd.extend(["--hf_token", resolved_hf_token])
            cmd.append(str(input_path))

            env = os.environ.copy()
            if resolved_hf_token:
                env.setdefault("HF_TOKEN", resolved_hf_token)

            # --- acquire GPU lock with timeout ---
            acquired = self._gpu_lock.acquire(timeout=self._settings.gpu_lock_timeout_sec)
            if not acquired:
                logger.warning("GPU lock timeout after %ds", self._settings.gpu_lock_timeout_sec)
                raise HTTPException(
                    status_code=503,
                    detail="Server busy — GPU is occupied. Try again later.",
                )

            try:
                result = self._run(cmd, env, resolved_hf_token)
            finally:
                self._gpu_lock.release()

            # --- handle result ---
            if result.returncode != 0:
                logger.error("WhisperX failed (rc=%d)", result.returncode)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "whisperx failed",
                        "returncode": result.returncode,
                        "stdout_tail": _sanitize_tail(result.stdout_tail),
                        "stderr_tail": _sanitize_tail(result.stderr_tail),
                    },
                )

            try:
                out_path = _find_output_file(output_dir, resolved_output_format)
            except Exception as e:
                logger.error("Output file not found: %s", e)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "output file not found",
                        "message": str(e),
                        "stdout_tail": _sanitize_tail(result.stdout_tail),
                        "stderr_tail": _sanitize_tail(result.stderr_tail),
                    },
                ) from e

            return TranscribeResult(
                content=out_path.read_bytes(),
                media_type=_media_type_for(resolved_output_format),
            )

    def _run(self, cmd: list[str], env: dict[str, str], token: str | None) -> RunResult:
        try:
            return self._runner.run(cmd, env, token)
        except Exception as e:
            logger.exception("Failed to start whisperx")
            raise HTTPException(
                status_code=500,
                detail={"error": "failed to start whisperx", "message": str(e)},
            ) from e

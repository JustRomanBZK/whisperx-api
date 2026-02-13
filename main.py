from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

# --- API Key ---
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    print("FATAL: API_KEY environment variable is not set. Exiting.")
    sys.exit(1)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        key = request.headers.get("X-API-Key")
        if not key or key != API_KEY:
            return Response(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)


app = FastAPI(
    title="WhisperX API",
    version="0.2.0",
    description="HTTP-обёртка над WhisperX CLI. Один запрос за раз (GPU lock).",
)
app.add_middleware(ApiKeyMiddleware)

GPU_LOCK = threading.Lock()

OutputFormat = Literal["srt", "vtt", "txt", "json"]
ComputeType = Literal["float16", "float32", "int8"]
Device = Literal["cuda", "cpu"]

TAIL_MAX_BYTES = 4000


def _env_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


DEFAULT_DEVICE: Device = _env_default("WHISPERX_DEVICE", "cuda")  # type: ignore[assignment]
DEFAULT_COMPUTE_TYPE: ComputeType = _env_default("WHISPERX_COMPUTE_TYPE", "float16")  # type: ignore[assignment]
DEFAULT_MODEL = _env_default("WHISPERX_MODEL", "tiny")
DEFAULT_LANGUAGE = os.environ.get("WHISPERX_LANGUAGE")  # если не задан — автоопределение языка
DEFAULT_OUTPUT_FORMAT: OutputFormat = _env_default("WHISPERX_OUTPUT_FORMAT", "srt")  # type: ignore[assignment]
DEFAULT_BATCH_SIZE = int(_env_default("WHISPERX_BATCH_SIZE", "1"))


def _append_tail(buf: bytearray, chunk: bytes, max_len: int = TAIL_MAX_BYTES) -> None:
    buf.extend(chunk)
    if len(buf) > max_len:
        del buf[:-max_len]


def _norm_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _parse_config_json(config_json: str | None) -> dict[str, Any]:
    config_json = _norm_str(config_json)
    if not config_json:
        return {}
    try:
        obj = json.loads(config_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid config json", "message": str(e)}) from e
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail={"error": "invalid config json", "message": "config must be a JSON object"})
    return obj


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
            token = authorization[len(prefix) :].strip()
            return token if token else None
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return env_token if env_token else None


def _whisperx_cmd() -> list[str]:
    whisperx = shutil.which("whisperx")
    if whisperx:
        return [whisperx]
    return ["python", "-m", "whisperx"]


def _find_output_file(output_dir: Path, output_format: OutputFormat) -> Path:
    matches = sorted(output_dir.glob(f"*.{output_format}"))
    if not matches:
        raise FileNotFoundError(f"No *.{output_format} in {output_dir}")
    if len(matches) == 1:
        return matches[0]
    return max(matches, key=lambda p: p.stat().st_size)


def _pump_stream(
    pipe,
    target,
    tail: bytearray,
    token_bytes: bytes | None,
) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            if token_bytes:
                chunk = chunk.replace(token_bytes, b"***")
            target.write(chunk)
            target.flush()
            _append_tail(tail, chunk)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _run_whisperx_streaming(
    cmd: list[str],
    env: dict[str, str],
    token: str | None,
) -> tuple[int, str, str]:
    token_bytes = token.encode("utf-8") if token else None
    stdout_tail = bytearray()
    stderr_tail = bytearray()

    run_env = env.copy()
    run_env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    t_out = threading.Thread(
        target=_pump_stream,
        args=(proc.stdout, sys.stdout.buffer, stdout_tail, token_bytes),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_pump_stream,
        args=(proc.stderr, sys.stderr.buffer, stderr_tail, token_bytes),
        daemon=True,
    )
    t_out.start()
    t_err.start()

    returncode = proc.wait()
    t_out.join()
    t_err.join()

    return (
        returncode,
        stdout_tail.decode("utf-8", errors="replace"),
        stderr_tail.decode("utf-8", errors="replace"),
    )


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.post("/transcribe", response_class=Response)
def transcribe(
    file: Annotated[UploadFile, File(description="Аудио файл (wav/mp3/m4a/...)")],
    # параметры можно передавать как отдельные поля form-data ...
    model: Annotated[str | None, Form(description="Whisper model name (optional)", examples=["tiny"])] = None,
    language: Annotated[str | None, Form(description="Language code (optional). Если не передавать — автоопределение.", examples=["uk", "ru", "en"])] = None,
    output_format: Annotated[OutputFormat | None, Form(description="Output format (optional)")] = None,
    diarize: Annotated[bool | None, Form(description="Enable diarization (optional, needs HF token)")] = None,
    batch_size: Annotated[int | None, Form(description="Batch size (optional)")] = None,
    device: Annotated[Device | None, Form(description="Device (optional)")] = None,
    compute_type: Annotated[ComputeType | None, Form(description="Compute type (optional)")] = None,
    # ... или одним JSON-объектом в поле config (string).
    # Это нужно, т.к. при multipart upload нельзя одновременно отправить JSON body через requests `json=...`.
    config: Annotated[str | None, Form(description="JSON object with options (optional)")] = None,
    hf_token: Annotated[str | None, Form(description="HuggingFace token (optional)")] = None,
    x_hf_token: Annotated[str | None, Header(alias="X-HF-Token")] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> Response:
    cfg = _parse_config_json(config)

    cfg_model = _cfg_get_str(cfg, "model")
    cfg_language = _cfg_get_str(cfg, "language")
    cfg_output_format = _cfg_get_literal(cfg, "output_format", {"srt", "vtt", "txt", "json"})
    cfg_diarize = _cfg_get_bool(cfg, "diarize")
    cfg_batch_size = _cfg_get_int(cfg, "batch_size")
    cfg_device = _cfg_get_literal(cfg, "device", {"cuda", "cpu"})
    cfg_compute_type = _cfg_get_literal(cfg, "compute_type", {"float16", "float32", "int8"})
    cfg_hf_token = _cfg_get_str(cfg, "hf_token")

    resolved_model = _norm_str(model) or cfg_model or DEFAULT_MODEL
    resolved_language = _norm_str(language) or cfg_language or _norm_str(DEFAULT_LANGUAGE)
    resolved_output_format: OutputFormat = (output_format or cfg_output_format or DEFAULT_OUTPUT_FORMAT)  # type: ignore[assignment]
    resolved_batch_size = batch_size if batch_size is not None else (cfg_batch_size if cfg_batch_size is not None else DEFAULT_BATCH_SIZE)
    resolved_device: Device = (device or cfg_device or DEFAULT_DEVICE)  # type: ignore[assignment]
    resolved_compute_type: ComputeType = (compute_type or cfg_compute_type or DEFAULT_COMPUTE_TYPE)  # type: ignore[assignment]

    hf_token_input = _norm_str(hf_token) or cfg_hf_token
    resolved_hf_token = _resolve_hf_token(hf_token_input, x_hf_token, authorization)

    resolved_diarize = diarize if diarize is not None else (cfg_diarize if cfg_diarize is not None else False)
    if resolved_diarize and not resolved_hf_token:
        raise HTTPException(
            status_code=400,
            detail="`diarize=true` требует HF token (form `hf_token`, header `X-HF-Token` или env `HF_TOKEN`).",
        )

    with tempfile.TemporaryDirectory(prefix="whisperx_api_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_dir = tmp_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = os.path.basename(file.filename or "audio")
        if not safe_name.strip("."):
            safe_name = "audio"
        input_path = tmp_dir / safe_name

        try:
            with input_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            try:
                file.file.close()
            except Exception:
                pass

        cmd: list[str] = [
            *_whisperx_cmd(),
            "--device",
            resolved_device,
            "--compute_type",
            resolved_compute_type,
            "--model",
            resolved_model,
            "--output_format",
            resolved_output_format,
            "--output_dir",
            str(output_dir),
            "--batch_size",
            str(resolved_batch_size),
        ]
        if resolved_language:
            cmd.extend(["--language", resolved_language])
        if resolved_diarize:
            cmd.append("--diarize")
        if resolved_hf_token:
            cmd.extend(["--hf_token", resolved_hf_token])
        cmd.append(str(input_path))

        env = os.environ.copy()
        if resolved_hf_token:
            env.setdefault("HF_TOKEN", resolved_hf_token)

        with GPU_LOCK:
            try:
                returncode, stdout_tail, stderr_tail = _run_whisperx_streaming(cmd, env, resolved_hf_token)
            except Exception as e:
                raise HTTPException(status_code=500, detail={"error": "failed to start whisperx", "message": str(e)}) from e

        if returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "whisperx failed",
                    "returncode": returncode,
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                },
            )

        try:
            out_path = _find_output_file(output_dir, resolved_output_format)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "output file not found",
                    "message": str(e),
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                },
            ) from e

        data = out_path.read_bytes()

        media_type = "text/plain; charset=utf-8"
        if resolved_output_format == "vtt":
            media_type = "text/vtt; charset=utf-8"
        elif resolved_output_format == "json":
            media_type = "application/json; charset=utf-8"

        return Response(content=data, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=_env_default("HOST", "0.0.0.0"),
        port=int(_env_default("PORT", "8000")),
        workers=1,
    )

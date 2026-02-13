from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response

from .config import ComputeType, Device, OutputFormat, Settings
from .service import TranscribeParams, TranscriptionService

logger = logging.getLogger("whisperx.routes")


def _parse_config_json(config_json: str | None) -> dict:
    if not config_json or not config_json.strip():
        return {}
    try:
        obj = json.loads(config_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid config json", "message": str(e)}) from e
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail={"error": "invalid config json", "message": "config must be a JSON object"})
    return obj


def register_routes(app: FastAPI, service: TranscriptionService, settings: Settings) -> None:

    @app.get("/health", response_class=PlainTextResponse)
    def health() -> str:
        return "ok"

    @app.post("/transcribe", response_class=Response)
    def transcribe(
        file: Annotated[UploadFile, File(description="Аудіо файл (wav/mp3/m4a/...)")],
        model: Annotated[str | None, Form(description="Whisper model name (optional)", examples=["tiny"])] = None,
        language: Annotated[str | None, Form(description="Language code (optional). Якщо не передати — автовизначення.", examples=["uk", "ru", "en"])] = None,
        output_format: Annotated[OutputFormat | None, Form(description="Output format (optional)")] = None,
        diarize: Annotated[bool | None, Form(description="Enable diarization (optional, needs HF token)")] = None,
        batch_size: Annotated[int | None, Form(description="Batch size (optional)")] = None,
        device: Annotated[Device | None, Form(description="Device (optional)")] = None,
        compute_type: Annotated[ComputeType | None, Form(description="Compute type (optional)")] = None,
        diarize_model: Annotated[str | None, Form(description="Diarization model (optional)")] = None,
        config: Annotated[str | None, Form(description="JSON object with options (optional)")] = None,
        hf_token: Annotated[str | None, Form(description="HuggingFace token (optional)")] = None,
        x_hf_token: Annotated[str | None, Header(alias="X-HF-Token")] = None,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ) -> Response:
        # --- file size check ---
        max_bytes = settings.max_file_size_mb * 1024 * 1024
        try:
            file_bytes = file.file.read()
        finally:
            try:
                file.file.close()
            except Exception:
                pass

        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.max_file_size_mb} MB.",
            )

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file.")

        logger.info(
            "Transcribe request: file=%s size=%d model=%s lang=%s fmt=%s diarize=%s",
            file.filename, len(file_bytes), model, language, output_format, diarize,
        )

        cfg = _parse_config_json(config)

        params = TranscribeParams(
            file_bytes=file_bytes,
            filename=file.filename or "audio",
            model=model,
            language=language,
            output_format=output_format,
            diarize=diarize,
            batch_size=batch_size,
            device=device,
            compute_type=compute_type,
            diarize_model=diarize_model,
            hf_token=hf_token,
            config=cfg,
        )

        result = service.transcribe(params, hf_token_header=x_hf_token, authorization=authorization)

        return Response(content=result.content, media_type=result.media_type)

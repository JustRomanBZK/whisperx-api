from __future__ import annotations

import logging
import sys

from fastapi import FastAPI

from .config import Settings
from .middleware import ApiKeyMiddleware
from .routes import register_routes
from .runner import WhisperXRunner
from .service import TranscriptionService


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    _setup_logging()

    if settings is None:
        settings = Settings.from_env()

    runner = WhisperXRunner(timeout=settings.process_timeout_sec)
    service = TranscriptionService(settings, runner)

    app = FastAPI(
        title="WhisperX API",
        version="0.3.0",
        description="HTTP-обгортка над WhisperX CLI. Один запит за раз (GPU lock).",
    )
    app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)
    register_routes(app, service, settings)

    logging.getLogger("whisperx").info(
        "WhisperX API started — device=%s model=%s compute=%s diarize_model=%s",
        settings.device, settings.model, settings.compute_type, settings.diarize_model,
    )

    return app

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

OutputFormat = Literal["srt", "vtt", "txt", "json"]
ComputeType = Literal["float16", "float32", "int8"]
Device = Literal["cuda", "cpu"]

OUTPUT_FORMATS: set[str] = {"srt", "vtt", "txt", "json"}
COMPUTE_TYPES: set[str] = {"float16", "float32", "int8"}
DEVICES: set[str] = {"cuda", "cpu"}

TAIL_MAX_BYTES = 4000


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


@dataclass(frozen=True)
class Settings:
    api_key: str

    # whisperx defaults
    device: Device
    compute_type: ComputeType
    model: str
    language: str | None
    output_format: OutputFormat
    batch_size: int
    diarize_model: str

    # server
    host: str
    port: int

    # limits
    max_file_size_mb: int
    process_timeout_sec: int
    gpu_lock_timeout_sec: int

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.environ.get("API_KEY", "")
        if not api_key:
            print("FATAL: API_KEY environment variable is not set. Exiting.")
            sys.exit(1)

        return cls(
            api_key=api_key,
            device=_env("WHISPERX_DEVICE", "cuda"),  # type: ignore[arg-type]
            compute_type=_env("WHISPERX_COMPUTE_TYPE", "float16"),  # type: ignore[arg-type]
            model=_env("WHISPERX_MODEL", "tiny"),
            language=os.environ.get("WHISPERX_LANGUAGE") or None,
            output_format=_env("WHISPERX_OUTPUT_FORMAT", "srt"),  # type: ignore[arg-type]
            batch_size=int(_env("WHISPERX_BATCH_SIZE", "1")),
            diarize_model=_env(
                "WHISPERX_DIARIZE_MODEL",
                "pyannote/speaker-diarization-community-1",
            ),
            host=_env("HOST", "0.0.0.0"),
            port=int(_env("PORT", "8000")),
            max_file_size_mb=int(_env("MAX_FILE_SIZE_MB", "500")),
            process_timeout_sec=int(_env("PROCESS_TIMEOUT", "600")),
            gpu_lock_timeout_sec=int(_env("GPU_LOCK_TIMEOUT", "660")),
        )

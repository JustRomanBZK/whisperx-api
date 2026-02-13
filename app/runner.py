from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass

from .config import TAIL_MAX_BYTES

logger = logging.getLogger("whisperx.runner")


@dataclass
class RunResult:
    returncode: int
    stdout_tail: str
    stderr_tail: str


def _append_tail(buf: bytearray, chunk: bytes, max_len: int = TAIL_MAX_BYTES) -> None:
    buf.extend(chunk)
    if len(buf) > max_len:
        del buf[:-max_len]


def _pump_stream(
    pipe,  # noqa: ANN001
    target,  # noqa: ANN001
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


class WhisperXRunner:
    """Запуск WhisperX subprocess із streaming виводом та таймаутом."""

    def __init__(self, timeout: int) -> None:
        self._timeout = timeout

    def run(
        self,
        cmd: list[str],
        env: dict[str, str],
        token: str | None,
    ) -> RunResult:
        token_bytes = token.encode("utf-8") if token else None
        stdout_tail = bytearray()
        stderr_tail = bytearray()

        run_env = env.copy()
        run_env.setdefault("PYTHONUNBUFFERED", "1")

        logger.info("Starting whisperx: %s", " ".join(cmd[:4]) + " ...")
        proc = subprocess.Popen(
            cmd,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        assert proc.stdout is not None  # noqa: S101
        assert proc.stderr is not None  # noqa: S101

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

        try:
            returncode = proc.wait(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            logger.error("WhisperX process timed out after %ds, killing", self._timeout)
            proc.kill()
            proc.wait(timeout=10)
            returncode = -1

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        return RunResult(
            returncode=returncode,
            stdout_tail=stdout_tail.decode("utf-8", errors="replace"),
            stderr_tail=stderr_tail.decode("utf-8", errors="replace"),
        )

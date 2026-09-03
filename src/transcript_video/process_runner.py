from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class FFmpegProgress:
    elapsed_seconds: float | None = None
    speed: str | None = None
    state: str | None = None


_progress_callback: ContextVar[Callable[[FFmpegProgress], None] | None] = ContextVar(
    "ffmpeg_progress_callback", default=None
)


@contextmanager
def ffmpeg_progress_handler(callback: Callable[[FFmpegProgress], None]):
    """Attach FFmpeg progress to the current pipeline/TUI worker context."""
    token = _progress_callback.set(callback)
    try:
        yield
    finally:
        _progress_callback.reset(token)


def run_process(
    args: Sequence[str | Path], *, cwd: Path | None = None, timeout: float | None = None
) -> ProcessResult:
    command = tuple(str(item) for item in args)
    logger.debug("Running command: %s", subprocess.list2cmdline(command))
    try:
        completed = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise ProcessExecutionError(f"Command timed out after {timeout}s.") from exc
    result = ProcessResult(command, completed.returncode, completed.stdout, completed.stderr)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "No diagnostic output."
        raise ProcessExecutionError(f"Command failed ({completed.returncode}): {detail}")
    return result


def parse_ffmpeg_progress(values: dict[str, str]) -> FFmpegProgress:
    elapsed: float | None = None
    try:
        if "out_time_us" in values:
            elapsed = int(values["out_time_us"]) / 1_000_000
        elif "out_time_ms" in values:
            # FFmpeg historically names this field ms while reporting microseconds.
            elapsed = int(values["out_time_ms"]) / 1_000_000
        elif "out_time" in values:
            hours, minutes, seconds = values["out_time"].split(":")
            elapsed = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        logger.debug("Ignoring malformed FFmpeg progress: %r", values)
    return FFmpegProgress(elapsed, values.get("speed"), values.get("progress"))


def run_ffmpeg(
    args: Sequence[str | Path],
    *,
    on_progress: Callable[[FFmpegProgress], None] | None = None,
) -> ProcessResult:
    on_progress = on_progress or _progress_callback.get()
    command = [str(item) for item in args]
    if "-progress" not in command:
        command[1:1] = ["-progress", "pipe:1", "-nostats"]
    logger.debug("Running FFmpeg: %s", subprocess.list2cmdline(command))
    values: dict[str, str] = {}
    stdout_lines: list[str] = []
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as error_stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=error_stream,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                stdout_lines.append(raw_line)
                key, separator, value = raw_line.strip().partition("=")
                if not separator:
                    continue
                values[key] = value
                if key == "progress":
                    if on_progress:
                        on_progress(parse_ffmpeg_progress(values))
                    values = {}
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        error_stream.seek(0)
        stderr = error_stream.read()
    result = ProcessResult(tuple(command), process.returncode, "".join(stdout_lines), stderr)
    if process.returncode:
        raise ProcessExecutionError(
            f"FFmpeg failed ({process.returncode}): {stderr.strip() or 'No diagnostic output.'}"
        )
    return result


def probe_media(ffprobe: str | Path, media: Path) -> dict[str, object]:
    result = run_process(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", media]
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProcessExecutionError("ffprobe returned invalid JSON.") from exc

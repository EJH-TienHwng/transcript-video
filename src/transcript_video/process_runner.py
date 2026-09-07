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

from .context import current_context

logger = logging.getLogger(__name__)


class ProcessExecutionError(RuntimeError):
    """Concise public error; the full subprocess diagnostics remain in DEBUG logs."""

    def __init__(
        self,
        message: str | None = None,
        *,
        command: Sequence[str] = (),
        returncode: int | None = None,
        stdout: str | bytes | None = "",
        stderr: str | bytes | None = "",
        tool: str | None = None,
        timeout: float | None = None,
    ):
        self.command = tuple(command)
        self.returncode = returncode
        self.stdout = (
            stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout or ""
        )
        self.stderr = (
            stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr or ""
        )
        self.tool = tool or (Path(self.command[0]).stem if self.command else None)
        self.timeout = timeout
        self.stage = current_context.get().stage
        label = self.tool or "Command"
        super().__init__(
            message
            or (
                f"{label} timed out after {timeout}s"
                if timeout is not None
                else f"{label} exited with code {returncode}"
            )
        )


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
    args: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    tool: str | None = None,
) -> ProcessResult:
    command = tuple(str(item) for item in args)
    logger.debug("Running command: %s", subprocess.list2cmdline(command))
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        logger.debug(
            "Command timed out: command=%r timeout=%s stdout=%r stderr=%r",
            command,
            timeout,
            exc.stdout,
            exc.stderr,
        )
        raise ProcessExecutionError(
            command=command, stdout=exc.stdout, stderr=exc.stderr, tool=tool, timeout=timeout
        ) from exc
    except OSError as exc:
        logger.debug("Could not start command=%r", command, exc_info=True)
        raise ProcessExecutionError(
            command=command,
            stderr=str(exc),
            tool=tool,
            message=f"{tool or Path(command[0]).stem} could not start: {exc.strerror or exc}",
        ) from exc
    result = ProcessResult(command, completed.returncode, completed.stdout, completed.stderr)
    if completed.returncode:
        logger.debug(
            "Command failed: command=%r returncode=%s stdout=%s stderr=%s",
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
        raise ProcessExecutionError(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            tool=tool,
        )
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
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=error_stream,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            logger.debug("Could not start FFmpeg command=%r", command, exc_info=True)
            raise ProcessExecutionError(
                f"FFmpeg could not start: {exc.strerror or exc}",
                command=command,
                stderr=str(exc),
                tool="FFmpeg",
            ) from exc
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
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        error_stream.seek(0)
        stderr = error_stream.read()
    result = ProcessResult(tuple(command), process.returncode, "".join(stdout_lines), stderr)
    if process.returncode:
        logger.debug(
            "FFmpeg failed: command=%r returncode=%s stdout=%s stderr=%s",
            command,
            process.returncode,
            result.stdout,
            stderr,
        )
        raise ProcessExecutionError(
            command=command,
            returncode=process.returncode,
            stdout=result.stdout,
            stderr=stderr,
            tool="FFmpeg",
        )
    return result


def probe_media(
    ffprobe: str | Path, media: Path, *, timeout: float | None = None
) -> dict[str, object]:
    result = run_process(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", media],
        timeout=timeout,
        tool="ffprobe",
    )
    try:
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise ValueError("ffprobe metadata must be a JSON object")
        return data
    except ValueError as exc:
        logger.debug(
            "Invalid ffprobe JSON: command=%r stdout=%s stderr=%s",
            result.args,
            result.stdout,
            result.stderr,
        )
        raise ProcessExecutionError(
            "ffprobe returned invalid JSON.",
            command=result.args,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            tool="ffprobe",
        ) from exc

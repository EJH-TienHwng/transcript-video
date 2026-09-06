from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from ..context import current_context

_previous_levels: dict[str, int] = {}


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.pipeline_context = current_context.get().fields()
        return True


class TextLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")
        )
        context = " ".join(f"{key}={value}" for key, value in record.pipeline_context.items())
        text = f"{timestamp} {record.levelname:<7} {context} {record.name} - {record.getMessage()}"
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "schema_version": 1,
            "timestamp": datetime.fromtimestamp(record.created)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            **record.pipeline_context,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class RunLogs:
    text: Path
    jsonl: Path
    global_log: Path


def close_logging() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if getattr(handler, "transcript_video_owned", False):
            root.removeHandler(handler)
            handler.close()
    for name, level in _previous_levels.items():
        logging.getLogger(name).setLevel(level)
    _previous_levels.clear()


def configure_logging(
    console: Console,
    verbosity: int,
    log_file: Path | None,
    *,
    run_id: str | None = None,
    command: str = "process",
    write_files: bool = True,
    console_enabled: bool = True,
) -> RunLogs | None:
    """Own only our handlers; external/library handlers survive reconfiguration.

    --log-file retains its rotating-log destination semantics. Per-run DEBUG files
    live in its sibling runs/ folder. No files are opened for dry-run.
    """
    close_logging()
    root = logging.getLogger()
    _previous_levels[""] = root.level
    root.setLevel(logging.DEBUG)

    def attach(handler, formatter, level=logging.DEBUG):
        handler.transcript_video_owned = True
        handler.setLevel(level)
        handler.addFilter(ContextFilter())
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if console_enabled:
        handler = RichHandler(
            console=console,
            show_path=verbosity >= 2,
            rich_tracebacks=verbosity >= 2,
            tracebacks_show_locals=False,
            markup=False,
        )
        # Exceptions are rendered by the command owner, once, including under -vv.
        handler.addFilter(lambda record: not getattr(record, "diagnostic_only", False))
        attach(
            handler,
            logging.Formatter("%(message)s"),
            logging.DEBUG
            if verbosity >= 2
            else logging.INFO
            if verbosity == 1
            else logging.WARNING,
        )
    for name in ("transformers", "urllib3", "huggingface_hub", "filelock", "httpx", "httpcore"):
        _previous_levels[name] = logging.getLogger(name).level
        logging.getLogger(name).setLevel(logging.INFO if verbosity >= 2 else logging.WARNING)
    if not write_files or log_file is None:
        return None
    global_path = log_file.expanduser().resolve()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    attach(
        RotatingFileHandler(global_path, maxBytes=10 * 1024**2, backupCount=5, encoding="utf-8"),
        TextLogFormatter(),
    )
    if run_id is None:
        return None
    folder = global_path.parent / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_{command}"
    logs = RunLogs(folder / f"{stem}.log", folder / f"{stem}.jsonl", global_path)
    attach(logging.FileHandler(logs.text, encoding="utf-8"), TextLogFormatter())
    attach(logging.FileHandler(logs.jsonl, encoding="utf-8"), JsonLogFormatter())
    return logs

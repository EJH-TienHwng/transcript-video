from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(console: Console, verbosity: int, log_file: Path | None) -> Path | None:
    """Configure package logging once for a CLI invocation."""
    console_level = (
        logging.WARNING if verbosity < 0 else logging.DEBUG if verbosity else logging.INFO
    )
    log_path = log_file.expanduser().resolve() if log_file is not None else None

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    rich_handler = RichHandler(
        console=console,
        level=console_level,
        show_path=verbosity >= 2,
        rich_tracebacks=verbosity >= 2,
        markup=False,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rich_handler)
    if log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(file_handler)
    return log_path

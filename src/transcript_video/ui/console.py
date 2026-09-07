from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

from rich.console import Console

from .theme import THEME


@dataclass(slots=True)
class ConsolePair:
    out: Console
    err: Console


def make_consoles(*, no_color: bool = False) -> ConsolePair:
    color_disabled = no_color or "NO_COLOR" in os.environ
    return ConsolePair(
        Console(theme=THEME, no_color=color_disabled, force_terminal=None),
        Console(theme=THEME, no_color=color_disabled, force_terminal=None, file=sys.stderr),
    )


@contextmanager
def help_output(no_color: bool = False):
    """Typer's eager help uses a separate console; scope its presentation settings."""
    from typer import rich_utils

    previous = rich_utils.COLOR_SYSTEM, rich_utils.FORCE_TERMINAL
    if no_color or "NO_COLOR" in os.environ or not sys.stdout.isatty():
        rich_utils.COLOR_SYSTEM = None
        rich_utils.FORCE_TERMINAL = False
    try:
        yield
    finally:
        rich_utils.COLOR_SYSTEM, rich_utils.FORCE_TERMINAL = previous

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.theme import Theme

THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "yellow",
        "error": "bold red",
        "path": "bright_blue",
        "stage": "bold magenta",
    }
)


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

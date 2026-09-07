"""Shared semantic palette for Rich, Questionary, and Textual."""

from rich.theme import Theme

PALETTE = {
    "accent": "#00aaaa",
    "success": "#00aa00",
    "warning": "#aaaa00",
    "error": "#ff5555",
    "info": "#00aaaa",
    "muted": "#888888",
    "path": "#5555ff",
    "stage": "#aa55aa",
    "active": "#00aaaa",
    "pending": "#888888",
    "review": "#aaaa00",
}
THEME = Theme(PALETTE)
SYMBOLS = {
    "start": "●",
    "progress": "●",
    "complete": "✓",
    "reused": "↻",
    "review": "⚠",
    "warning": "⚠",
    "failure": "✗",
    "artifact": "→",
    "pending": "○",
}


def questionary_style(no_color: bool = False):
    from questionary import Style
    from questionary.constants import DEFAULT_STYLE

    if no_color:
        return Style([(name, "") for name, _ in DEFAULT_STYLE.style_rules])

    mapping = {
        "qmark": "accent",
        "question": "active",
        "answer": "success",
        "pointer": "active",
        "highlighted": "active",
        "selected": "success",
        "instruction": "muted",
        "search_success": "success",
        "search_none": "error",
    }
    return Style(
        [(key, "" if no_color else "fg:" + PALETTE[value]) for key, value in mapping.items()]
    )


def terminal_symbols(encoding: str) -> dict[str, str]:
    try:
        "".join(SYMBOLS.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return {
            "start": ">",
            "progress": ">",
            "complete": "OK",
            "reused": "REUSED",
            "review": "!",
            "warning": "!",
            "failure": "X",
            "artifact": "->",
            "pending": "-",
        }
    return SYMBOLS

"""Questionary-based course creation wizard."""

from __future__ import annotations

import logging
import math
import re
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import questionary
    from questionary import Choice
except ImportError as exc:
    raise RuntimeError(
        "Course wizard requires 'questionary'. Install dependencies with: uv sync"
    ) from exc

from ..artifacts import is_partial_artifact
from ..config import VIDEO_EXTENSIONS, find_project_root
from ..hardware import get_ffprobe_exe
from ..process_runner import probe_media
from ..ui.console import make_consoles
from ..ui.progress import format_duration
from ..ui.theme import questionary_style, terminal_symbols
from .config import (
    RenderConfig,
    TocConfig,
    parse_course_config,
    save_course_config,
    validate_session_video,
)
from .timeline import build_timeline

console = make_consoles().out
PROMPT_STYLE = questionary_style(bool(console.no_color))
logger = logging.getLogger(__name__)


@contextmanager
def wizard_ui(ui_console):
    # A linear wizard owns the terminal for this scope; restore it for the next invocation.
    global console, PROMPT_STYLE
    previous = console, PROMPT_STYLE
    console, PROMPT_STYLE = ui_console, questionary_style(bool(ui_console.no_color))
    try:
        yield
    finally:
        console, PROMPT_STYLE = previous


def _ask(kind: str, message: str, **kwargs):
    try:
        answer = getattr(questionary, kind)(message, style=PROMPT_STYLE, **kwargs).unsafe_ask()
    except (KeyboardInterrupt, EOFError):
        logger.info("Wizard cancelled")
        raise
    if answer is None:
        logger.info("Wizard cancelled")
        raise KeyboardInterrupt
    return answer


def _project_root() -> Path:
    """Find the repository root from the current working directory."""
    return find_project_root()


def _relative_or_absolute(path: Path, root: Path) -> str:
    """Store project-local paths as relative paths for portable JSON configs."""
    path = path.expanduser().resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _slugify_filename(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "course"


def _default_title_from_video(path: Path) -> str:
    title = path.stem

    # Remove common pipeline output suffixes.
    suffixes = (
        "_en-dub_en-sub",
        "_vi-dub_en-sub",
        "_en-sub",
        "_final",
    )
    lowered = title.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            title = title[: -len(suffix)]
            break

    title = re.sub(r"[_-]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _scan_videos(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not is_partial_artifact(path)
    )


def _ask_text(message: str, default: str | None = None) -> str:
    return _ask("text", message, default=default or "").strip()


def _ask_int(message: str, default: int, minimum: int = 1) -> int:
    while True:
        answer = _ask_text(message, str(default))
        try:
            value = int(answer)
        except ValueError:
            console.print("The value must be an integer.", style="error")
            continue

        if not math.isfinite(value) or value < minimum:
            console.print(f"The value must be at least {minimum}.", style="error")
            continue
        return value


def _ask_float(message: str, default: float, minimum: float = 0.0) -> float:
    while True:
        answer = _ask_text(message, str(default))
        try:
            value = float(answer)
        except ValueError:
            console.print("The value must be a number.", style="error")
            continue

        if not math.isfinite(value) or value < minimum:
            console.print(f"The value must be at least {minimum}.", style="error")
            continue
        return value


def _step(index: int) -> None:
    symbols = terminal_symbols(console.encoding)
    names = ("Sessions", "Appearance", "Output", "Review")
    text = Text()
    for position, name in enumerate(names, 1):
        if position > 1:
            text.append(" — ", style="muted")
        kind = "complete" if position < index else "start" if position == index else "pending"
        style = "success" if position < index else "active" if position == index else "pending"
        text.append(f"{symbols[kind]} {name}", style=style)
    console.print(Panel(text, title=f"COURSE SETUP · {index}/4", border_style="accent"))


def _metadata(path: Path, cache: dict) -> dict:
    path = path.resolve()
    if path not in cache:
        try:
            data = probe_media(get_ffprobe_exe(), path, timeout=10)
            stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
            )
            duration = float(data.get("format", {}).get("duration", 0))
            cache[path] = {
                "duration": duration if math.isfinite(duration) and duration > 0 else None,
                "width": stream.get("width"),
                "height": stream.get("height"),
                "codec": stream.get("codec_name"),
            }
        except Exception:
            logger.debug("Video metadata unavailable: %s", path, exc_info=True)
            cache[path] = {}
    return cache[path]


def _metadata_label(metadata: dict) -> str:
    parts = []
    if metadata.get("duration"):
        parts.append(format_duration(metadata["duration"]))
    if metadata.get("width") and metadata.get("height"):
        parts.append(f"{metadata['width']}x{metadata['height']}")
    if metadata.get("codec"):
        parts.append(str(metadata["codec"]).upper())
    return " · ".join(parts) or "metadata unavailable"


def _manual_video(root: Path, selected: list[Path]) -> Path:
    while True:
        path = Path(_ask_text("Video path:")).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        try:
            return validate_session_video(str(path), root, selected)
        except ValueError as exc:
            console.print(Text(str(exc), style="error"))


def _collect_sessions(
    root: Path, output_dir: Path, cache: dict | None = None, existing: list[dict] = ()
) -> list[dict]:
    cache = cache if cache is not None else {}
    used = [(root / item["video"]).resolve() for item in existing]
    videos = [video for video in _scan_videos(output_dir) if video not in used]
    console.print(Text(f"Found {len(videos)} videos · {output_dir}", style="muted"))
    console.print("Space toggles · Enter confirms · Review changes the final order", style="info")
    with console.status("Scanning video metadata…", spinner_style="accent"):
        choices = [
            Choice(f"{path.name} · {_metadata_label(_metadata(path, cache))}", value=path)
            for path in videos
        ]
    choices.append(Choice("Enter a video path manually…", value="__manual__"))
    selected = []
    while not selected:
        answer = _ask("checkbox", "Select course videos:", choices=choices)
        # Questionary returns checklist/display order; reordering is explicit on the review screen.
        selected = list(dict.fromkeys(path for path in answer if isinstance(path, Path)))
        try:
            for path in selected:
                validate_session_video(str(path), root, used)
        except ValueError as exc:
            console.print(Text(str(exc), style="error"))
            selected = []
            continue
        if "__manual__" in answer:
            while True:
                path = _manual_video(root, used + selected)
                selected.append(path)
                _metadata(path, cache)
                if not _ask("confirm", "Add another manual video?", default=False):
                    break
        if not selected:
            console.print("Select at least one video.", style="warning")
    logger.info("Wizard selected %d videos", len(selected))
    return [
        {
            "number": index,
            "title": _default_title_from_video(path) or f"Session {index}",
            "video": _relative_or_absolute(path, root),
        }
        for index, path in enumerate(
            selected, max((item["number"] for item in existing), default=0) + 1
        )
    ]


def _image_label(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        size = f"{image.width}x{image.height}"
        image.verify()
    return size


def _choose_theme(root: Path) -> str | None:
    default = root / "assets/bosch_theme.png"
    choices = []
    if default.is_file():
        try:
            choices.append(
                Choice(
                    f"Bosch default · {_image_label(default)} · assets/bosch_theme.png",
                    value=default,
                )
            )
        except (OSError, ValueError):
            console.print("Default theme is unreadable; choose another image.", style="warning")
    choices += [Choice("Custom image…", value="__manual__"), Choice("None", value="__none__")]
    answer = _ask("select", "Theme:", choices=choices)
    if answer == "__none__":
        return None
    if answer == "__manual__":
        while True:
            path = Path(_ask_text("Theme image path:")).expanduser()
            path = (path if path.is_absolute() else root / path).resolve()
            try:
                label = _image_label(path)
            except (OSError, ValueError):
                console.print(
                    "Choose an existing, readable image supported by Pillow.", style="error"
                )
                continue
            console.print(Text(f"{path.name} · {label}", style="path"))
            return _relative_or_absolute(path, root)
    return _relative_or_absolute(answer, root)


def _review_sessions(sessions: list[dict]) -> None:
    table = Table(
        title=f"Session order · {len(sessions)} sessions",
        header_style="accent",
        border_style="muted",
        expand=True,
    )
    for name, width in (
        ("ORDER", 5),
        ("NUMBER", 6),
        ("TITLE", None),
        ("VIDEO", None),
        ("STATUS", 6),
    ):
        table.add_column(name, width=width, overflow="fold")
    for position, session in enumerate(sessions, 1):
        path = Path(session["video"])
        video = Text(path.name, style="path")
        if str(path.parent) != ".":
            video.append("\n" + str(path.parent), style="muted")
        table.add_row(
            str(position),
            f"{session['number']:02d}",
            Text(session["title"]),
            video,
            Text("Ready", style="success"),
        )
    console.print(table)


def _edit_sessions(
    sessions: list[dict],
    root: Path | None = None,
    output_dir: Path | None = None,
    cache: dict | None = None,
) -> list[dict]:
    root = root or _project_root()
    output_dir = output_dir or root / "data/output"
    while True:
        _review_sessions(sessions)
        action = _ask(
            "select",
            "Review session list:",
            choices=[
                "Continue",
                "Add session",
                "Edit title/number",
                "Move up",
                "Move down",
                "Remove",
                "Cancel",
            ],
        )
        if action == "Continue":
            return sessions
        if action == "Cancel":
            raise KeyboardInterrupt
        if action == "Add session":
            sessions.extend(_collect_sessions(root, output_dir, cache, sessions))
            continue
        selected = _ask(
            "select",
            "Choose a session:",
            choices=[Choice(f"{i + 1:02d}. {s['title']}", value=i) for i, s in enumerate(sessions)],
        )
        item = sessions[selected]
        if action == "Edit title/number":
            item["title"] = _ask_text("Session title:", item["title"]) or item["title"]
            while True:
                number = _ask_int("Session number:", item["number"])
                if any(
                    i != selected and other["number"] == number for i, other in enumerate(sessions)
                ):
                    console.print("Session number is already in use.", style="error")
                else:
                    item["number"] = number
                    break
        elif action == "Remove":
            if len(sessions) == 1:
                console.print("A course must contain at least one session.", style="warning")
            elif (
                _ask("select", f'Remove "{item["title"]}"?', choices=["Keep", "Remove"]) == "Remove"
            ):
                sessions.pop(selected)
        else:
            destination = selected + (-1 if action == "Move up" else 1)
            if 0 <= destination < len(sessions):
                sessions[selected], sessions[destination] = (
                    sessions[destination],
                    sessions[selected],
                )
                console.print(
                    Text(
                        f'{terminal_symbols(console.encoding)["complete"]} Moved "{item["title"]}" {selected + 1:02d} -> {destination + 1:02d}',
                        style="success",
                    )
                )


def _appearance(config: dict, root: Path, custom: bool):
    _step(2)
    config["theme_image"] = _choose_theme(root)
    if custom:
        toc = config["toc"]
        toc["enabled"] = _ask("confirm", "Enable table of contents?", default=toc["enabled"])
        if toc["enabled"]:
            toc["heading"] = _ask_text("TOC heading:", toc["heading"]) or "TABLE OF CONTENTS"
            toc["items_per_page"] = _ask_int("Sessions per TOC page:", toc["items_per_page"])
            toc["page_duration"] = _ask_float("TOC page duration:", toc["page_duration"], 0.1)
        config["card_duration"] = _ask_float("Session card duration:", config["card_duration"], 0.1)
    console.print(
        Text(
            f"TOC: {'enabled' if config['toc']['enabled'] else 'disabled'} · Session cards: {config['card_duration']:g}s",
            style="info",
        )
    )


def _output(config: dict, config_name: str, root: Path, custom: bool) -> str:
    _step(3)
    name = Path(_ask_text("JSON config filename:", config_name)).name or config_name
    if not name.lower().endswith(".json"):
        name += ".json"
    output = Path(_ask_text("Final compiled video filename:", Path(config["output"]).name)).name
    if not output.lower().endswith(".mp4"):
        output += ".mp4"
    config["output"] = f"data/compilation/{output}"
    if custom:
        render = config["render"]
        config["add_chapters"] = _ask(
            "confirm", "Add MP4 chapters?", default=config["add_chapters"]
        )
        for key in ("width", "height", "fps", "audio_sample_rate"):
            while True:
                value = _ask_int(key.replace("_", " ").title() + ":", render[key])
                if key in {"width", "height"} and value % 2:
                    console.print(
                        "Width and height must be even for video encoding.", style="error"
                    )
                else:
                    render[key] = value
                    break
        for key in ("video_bitrate", "audio_bitrate"):
            render[key] = _ask_text(key.replace("_", " ").title() + ":", render[key]) or render[key]
        render["video_encoder"] = _ask(
            "select",
            "Video encoder:",
            choices=["auto", "h264_nvenc", "libx264"],
            default=render["video_encoder"],
        )
        if _ask("confirm", "Use a custom font?", default=bool(render["font_path"])):
            while True:
                path = Path(_ask_text("Font path:", render["font_path"] or "")).expanduser()
                path = (path if path.is_absolute() else root / path).resolve()
                if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}:
                    render["font_path"] = _relative_or_absolute(path, root)
                    break
                console.print("Choose an existing .ttf or .otf font.", style="error")
        else:
            render["font_path"] = None
    return name


def _summary(config: dict, config_path: Path, root: Path, cache: dict) -> dict:
    parsed = parse_course_config(config, root)
    durations = [_metadata(session.video, cache).get("duration") for session in parsed.sessions]
    source = sum(durations) if all(durations) else None
    estimated = build_timeline(parsed, durations)[-1].content_end if source is not None else None
    return {
        "Title": config["title"],
        "Sessions": len(config["sessions"]),
        "Source duration": format_duration(source) if source is not None else "Unavailable",
        "Estimated total": format_duration(estimated) if estimated is not None else "Unavailable",
        "Theme": config["theme_image"] or "None",
        "TOC": "Enabled" if config["toc"]["enabled"] else "Disabled",
        "Cards": f"{config['card_duration']:g} s",
        "Chapters": "Enabled" if config["add_chapters"] else "Disabled",
        "Rendering": f"{parsed.render.width}x{parsed.render.height} @ {parsed.render.fps} fps",
        "Encoder": parsed.render.video_encoder,
        "Config": _relative_or_absolute(config_path, root),
        "Output": config["output"],
    }


def create_course_config_interactive(
    root: Path | None = None, output_dir: Path | None = None
) -> Path | None:
    root = (root or _project_root()).expanduser().resolve()
    output_dir = (output_dir or root / "data/output").expanduser().resolve()
    logger.info("Wizard started: %s", root)
    cache = {}
    _step(1)
    console.print(
        "Arrow keys navigate · Space selects videos · Enter confirms · Ctrl+C cancels",
        style="muted",
    )
    title = _ask_text("Course title:", "Training Course") or "Training Course"
    sessions = _edit_sessions(_collect_sessions(root, output_dir, cache), root, output_dir, cache)
    custom = _ask("select", "Setup mode:", choices=["Recommended", "Custom"]) == "Custom"
    slug = _slugify_filename(title)
    config = {
        "title": title,
        "sessions": sessions,
        "output": f"data/compilation/{slug}.mp4",
        "work_dir": "data/compilation",
        "theme_image": None,
        "card_duration": 5.0,
        "add_chapters": True,
        "toc": asdict(TocConfig()),
        "render": asdict(RenderConfig()),
    }
    _appearance(config, root, custom)
    name = _output(config, f"{slug}.json", root, custom)
    while True:
        _step(4)
        config_path = root / "configs/courses" / name
        try:
            summary = _summary(config, config_path, root, cache)
        except (ValueError, FileNotFoundError) as exc:
            console.print(Text(str(exc), style="error"))
            name = _output(config, name, root, True)
            continue
        _review_sessions(config["sessions"])
        table = Table.grid(padding=(0, 2))
        for key, value in summary.items():
            table.add_row(
                Text(key, style="muted"),
                Text(
                    str(value),
                    style="path"
                    if key in {"Config", "Output"}
                    else "bold accent"
                    if key == "Title"
                    else "success"
                    if key == "Estimated total"
                    else "info",
                ),
            )
        console.print(Panel(table, title="Course summary", border_style="accent"))
        action = _ask(
            "select", "Final review:", choices=["Create configuration", "Edit settings", "Cancel"]
        )
        if action == "Cancel":
            logger.info("Wizard cancelled")
            console.print("Configuration creation cancelled.", style="warning")
            return None
        if action == "Edit settings":
            section = _ask(
                "select", "Edit:", choices=["Course & Sessions", "Appearance", "Output & Rendering"]
            )
            if section == "Course & Sessions":
                _step(1)
                config["title"] = _ask_text("Course title:", config["title"]) or config["title"]
                config["sessions"] = _edit_sessions(config["sessions"], root, output_dir, cache)
            elif section == "Appearance":
                _appearance(config, root, True)
            else:
                name = _output(config, name, root, True)
            continue
        if config_path.exists() and not _ask(
            "confirm", f"Overwrite {config_path.name}?", default=False
        ):
            continue
        save_course_config(config, config_path, root)
        logger.info("Wizard config written: %s", config_path)
        console.print(
            Panel(
                Text(
                    f"Config: {summary['Config']}\nOutput: {summary['Output']}\nSessions: {len(config['sessions'])}"
                ),
                title="Course configuration saved",
                border_style="success",
            )
        )
        for label, command in (("Build", "build"), ("Edit in TUI", "tui")):
            console.print(label, style="accent")
            console.print(
                Text(
                    f'uv run transcript-video course {command} --config "{summary["Config"]}"',
                    style="path",
                )
            )
        return config_path

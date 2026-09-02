from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import find_project_root


@dataclass
class SessionConfig:
    """One session in the final compiled training video."""

    title: str
    video: Path
    number: int | None = None


@dataclass
class TocConfig:
    enabled: bool = True
    items_per_page: int = 8
    page_duration: float = 5.0
    heading: str = "TABLE OF CONTENTS"


@dataclass
class RenderConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 48000
    font_path: Path | None = None


@dataclass
class CourseConfig:
    title: str
    output: Path
    theme_image: Path | None
    sessions: list[SessionConfig]
    card_duration: float = 5.0
    toc: TocConfig = field(default_factory=TocConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    work_dir: Path = Path("data/compilation")
    add_chapters: bool = True


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A path value must be a non-empty string or null.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _read_bool(data: dict[str, Any], key: str, default: bool, name: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def load_course_config(config_path: Path) -> CourseConfig:
    """Load and validate a course-builder JSON configuration file."""
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Course config not found: {config_path}")

    project_root = find_project_root(config_path)

    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Course config is not valid JSON: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Course config must be a JSON object.")

    sessions_raw = raw.get("sessions")
    if not isinstance(sessions_raw, list) or not sessions_raw:
        raise ValueError("Course config must contain at least one session.")

    sessions: list[SessionConfig] = []
    used_numbers = set()

    for index, item in enumerate(sessions_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Session {index} must be a JSON object.")
        title = str(item.get("title", "")).strip()
        video_value = item.get("video")

        if not title:
            raise ValueError(f"Session {index} is missing a title.")
        if not video_value:
            raise ValueError(f"Session '{title}' is missing a video path.")

        number = item.get("number")
        if number is not None:
            if isinstance(number, bool) or not isinstance(number, int):
                raise ValueError(f"Session '{title}': number must be an integer.")
            if number <= 0:
                raise ValueError(f"Session '{title}': number must be greater than zero.")
            if number in used_numbers:
                raise ValueError(f"Duplicate session number: {number}")
            used_numbers.add(number)

        video_path = _resolve_path(project_root, str(video_value))
        assert video_path is not None

        sessions.append(
            SessionConfig(
                title=title,
                video=video_path,
                number=number,
            )
        )

    toc_raw = raw.get("toc", {})
    render_raw = raw.get("render", {})
    if not isinstance(toc_raw, dict):
        raise ValueError("toc must be a JSON object.")
    if not isinstance(render_raw, dict):
        raise ValueError("render must be a JSON object.")

    toc = TocConfig(
        enabled=_read_bool(toc_raw, "enabled", True, "toc.enabled"),
        items_per_page=int(toc_raw.get("items_per_page", 8)),
        page_duration=float(toc_raw.get("page_duration", 5.0)),
        heading=str(toc_raw.get("heading", "TABLE OF CONTENTS")).strip() or "TABLE OF CONTENTS",
    )
    render = RenderConfig(
        width=int(render_raw.get("width", 1920)),
        height=int(render_raw.get("height", 1080)),
        fps=int(render_raw.get("fps", 30)),
        video_bitrate=str(render_raw.get("video_bitrate", "8M")),
        audio_bitrate=str(render_raw.get("audio_bitrate", "192k")),
        audio_sample_rate=int(render_raw.get("audio_sample_rate", 48000)),
        font_path=_resolve_path(project_root, render_raw.get("font_path")),
    )

    card_duration = float(raw.get("card_duration", 5.0))
    _require_positive(card_duration, "card_duration")
    _require_positive(toc.items_per_page, "toc.items_per_page")
    _require_positive(toc.page_duration, "toc.page_duration")
    _require_positive(render.width, "render.width")
    _require_positive(render.height, "render.height")
    _require_positive(render.fps, "render.fps")
    _require_positive(render.audio_sample_rate, "render.audio_sample_rate")
    if render.width % 2 or render.height % 2:
        raise ValueError("render.width and render.height must be even for yuv420p.")
    if not render.video_bitrate.strip():
        raise ValueError("render.video_bitrate must not be empty.")
    if not render.audio_bitrate.strip():
        raise ValueError("render.audio_bitrate must not be empty.")
    if render.font_path is not None and not render.font_path.is_file():
        raise FileNotFoundError(f"Font not found: {render.font_path}")

    theme_image = _resolve_path(project_root, raw.get("theme_image"))
    output = _resolve_path(project_root, raw.get("output", "data/compilation/course.mp4"))
    work_dir = _resolve_path(project_root, raw.get("work_dir", "data/compilation"))

    assert output is not None
    assert work_dir is not None
    if output.suffix.lower() != ".mp4":
        raise ValueError("Course output must use the .mp4 extension.")
    if output in {session.video for session in sessions}:
        raise ValueError("Course output must not overwrite an input session video.")
    if theme_image is not None and not theme_image.is_file():
        raise FileNotFoundError(f"Theme image not found: {theme_image}")

    return CourseConfig(
        title=str(raw.get("title", "Training Course")).strip() or "Training Course",
        output=output,
        theme_image=theme_image,
        sessions=sessions,
        card_duration=card_duration,
        toc=toc,
        render=render,
        work_dir=work_dir,
        add_chapters=_read_bool(raw, "add_chapters", True, "add_chapters"),
    )

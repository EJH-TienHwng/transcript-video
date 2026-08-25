from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SessionConfig:
    """One session in the final compiled training video."""

    title: str
    video: Path
    number: Optional[int] = None


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
    font_path: Optional[Path] = None


@dataclass
class CourseConfig:
    title: str
    output: Path
    theme_image: Optional[Path]
    sessions: List[SessionConfig]
    card_duration: float = 5.0
    toc: TocConfig = field(default_factory=TocConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    work_dir: Path = Path("data/compilation")
    add_chapters: bool = True


def _resolve_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} phải lớn hơn 0.")


def load_course_config(config_path: Path) -> CourseConfig:
    """Load and validate a course-builder JSON configuration file."""
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy course config: {config_path}")

    project_root = config_path.parent.parent if config_path.parent.name == "courses" else config_path.parent

    with config_path.open("r", encoding="utf-8") as file:
        raw: Dict[str, Any] = json.load(file)

    sessions_raw = raw.get("sessions") or []
    if not sessions_raw:
        raise ValueError("Course config phải có ít nhất 1 session.")

    sessions: List[SessionConfig] = []
    used_numbers = set()

    for index, item in enumerate(sessions_raw, start=1):
        title = str(item.get("title", "")).strip()
        video_value = item.get("video")

        if not title:
            raise ValueError(f"Session thứ {index} thiếu title.")
        if not video_value:
            raise ValueError(f"Session '{title}' thiếu video.")

        number = item.get("number")
        if number is not None:
            number = int(number)
            if number <= 0:
                raise ValueError(f"Session '{title}': number phải > 0.")
            if number in used_numbers:
                raise ValueError(f"Session number bị trùng: {number}")
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

    toc_raw = raw.get("toc") or {}
    render_raw = raw.get("render") or {}

    toc = TocConfig(
        enabled=bool(toc_raw.get("enabled", True)),
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

    theme_image = _resolve_path(project_root, raw.get("theme_image"))
    output = _resolve_path(project_root, raw.get("output", "data/compilation/course.mp4"))
    work_dir = _resolve_path(project_root, raw.get("work_dir", "data/compilation"))

    assert output is not None
    assert work_dir is not None

    return CourseConfig(
        title=str(raw.get("title", "Training Course")).strip() or "Training Course",
        output=output,
        theme_image=theme_image,
        sessions=sessions,
        card_duration=card_duration,
        toc=toc,
        render=render,
        work_dir=work_dir,
        add_chapters=bool(raw.get("add_chapters", True)),
    )

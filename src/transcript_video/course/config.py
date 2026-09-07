from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..artifacts import is_partial_artifact
from ..config import VIDEO_EXTENSIONS, find_project_root


@dataclass
class SessionConfig:
    """One session in the final compiled training video."""

    title: str
    video: Path
    number: int | None = None
    extra: dict = field(default_factory=dict, repr=False, compare=False)


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
    video_encoder: str = "auto"
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
    raw: dict = field(default_factory=dict, repr=False, compare=False)


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
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _read_bool(data: dict[str, Any], key: str, default: bool, name: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def load_course_config(config_path: Path) -> CourseConfig:
    """Load and validate a course-builder JSON configuration file."""
    return parse_course_config(
        read_course_document(config_path), find_project_root(config_path.parent)
    )


def read_course_document(config_path: Path) -> dict:
    """Read a validated editable document, preserving unknown fields at every level."""
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Course config not found: {config_path}")

    project_root = find_project_root(config_path)

    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Course config is not valid JSON: {config_path}") from exc

    parse_course_config(raw, project_root)
    return raw


def course_config_document(config: CourseConfig) -> dict:
    payload = asdict(config)
    original = payload.pop("raw")
    for section in ("toc", "render"):
        payload[section] = {**original.get(section, {}), **payload[section]}
    # Extras belong to the session even when its path/title/order changes.
    for index, item in enumerate(payload["sessions"]):
        extras = item.pop("extra")
        payload["sessions"][index] = {**extras, **item}
    return json.loads(json.dumps({**original, **payload}, default=str))


def save_course_config(
    config: CourseConfig | dict, config_path: Path, project_root: Path | None = None
) -> Path:
    """Validate before creating anything; atomically replace a complete JSON document."""
    config_path = config_path.expanduser().resolve()
    payload = course_config_document(config) if isinstance(config, CourseConfig) else config
    parse_course_config(payload, project_root or find_project_root(config_path.parent))
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=config_path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return config_path


def validate_session_video(value: str, root: Path, selected: list[Path] = ()) -> Path:
    path = _resolve_path(root, value)
    if path is None or not path.is_file():
        raise ValueError(f"Video not found: {path or value}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video format.")
    if is_partial_artifact(path):
        raise ValueError("Incomplete artifact cannot be used as a session video.")
    if path in selected:
        raise ValueError("This video is already selected.")
    return path


def parse_course_config(
    raw: dict, project_root: Path, *, allow_empty_sessions: bool = False
) -> CourseConfig:
    """Validate a draft before writing it, using the same rules as the JSON loader."""
    if not isinstance(raw, dict):
        raise ValueError("Course config must be a JSON object.")
    for section, values, defaults in (
        ("", raw, {"card_duration": 5.0}),
        ("toc.", raw.get("toc", {}), asdict(TocConfig())),
        ("render.", raw.get("render", {}), asdict(RenderConfig())),
    ):
        if not isinstance(values, dict):
            raise ValueError(f"{section.rstrip('.')} must be a JSON object.")
        for key, default in defaults.items():
            value = values.get(key, default)
            if type(default) is int and (type(value) is not int):
                raise ValueError(f"{section}{key} must be an integer.")
            if type(default) is float and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError(f"{section}{key} must be a finite number.")
            if isinstance(default, str) and not isinstance(value, str):
                raise ValueError(f"{section}{key} must be a string.")

    sessions_raw = raw.get("sessions")
    if not isinstance(sessions_raw, list) or (not sessions_raw and not allow_empty_sessions):
        raise ValueError("Course config must contain at least one session.")

    sessions: list[SessionConfig] = []
    used_numbers = set()
    used_videos = set()

    for index, item in enumerate(sessions_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Session {index} must be a JSON object.")
        if not isinstance(item.get("title", ""), str):
            raise ValueError(f"Session {index}: title must be a string.")
        title = item.get("title", "").strip()
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
        effective_number = number if number is not None else index
        if effective_number in used_numbers:
            raise ValueError(f"Duplicate session number: {effective_number}")
        used_numbers.add(effective_number)

        video_path = _resolve_path(project_root, video_value)
        assert video_path is not None
        if video_path in used_videos:
            raise ValueError(f"Duplicate session video: {video_path}")
        used_videos.add(video_path)

        sessions.append(
            SessionConfig(
                title=title,
                video=video_path,
                number=number,
                extra={
                    key: deepcopy(value)
                    for key, value in item.items()
                    if key not in {"number", "title", "video"}
                },
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
        video_encoder=str(render_raw.get("video_encoder", "auto")),
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
    if render.video_encoder not in {"auto", "h264_nvenc", "libx264"}:
        raise ValueError("render.video_encoder must be 'auto', 'h264_nvenc', or 'libx264'.")
    if render.font_path is not None and not render.font_path.is_file():
        raise FileNotFoundError(f"Font not found: {render.font_path}")

    theme_image = _resolve_path(project_root, raw.get("theme_image"))
    output = _resolve_path(project_root, raw.get("output", "data/compilation/course.mp4"))
    work_dir = _resolve_path(project_root, raw.get("work_dir", "data/compilation"))

    if output is None or work_dir is None:
        raise ValueError("output and work_dir must be non-empty paths.")
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
        raw=deepcopy(raw),
    )

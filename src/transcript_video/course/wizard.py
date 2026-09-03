"""Questionary-based course creation wizard."""

from __future__ import annotations

import json
import re
from pathlib import Path

try:
    import questionary
    from questionary import Choice
except ImportError as exc:
    raise RuntimeError(
        "Course Config TUI requires 'questionary'. Install dependencies with: uv sync"
    ) from exc

from ..config import VIDEO_EXTENSIONS, find_project_root


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
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _ask_text(message: str, default: str | None = None) -> str:
    answer = questionary.text(message, default=default or "").ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer.strip()


def _ask_int(message: str, default: int, minimum: int = 1) -> int:
    while True:
        answer = _ask_text(message, str(default))
        try:
            value = int(answer)
        except ValueError:
            print("The value must be an integer.")
            continue

        if value < minimum:
            print(f"The value must be at least {minimum}.")
            continue
        return value


def _ask_float(message: str, default: float, minimum: float = 0.0) -> float:
    while True:
        answer = _ask_text(message, str(default))
        try:
            value = float(answer)
        except ValueError:
            print("The value must be a number.")
            continue

        if value < minimum:
            print(f"The value must be at least {minimum}.")
            continue
        return value


def _pick_video(
    root: Path,
    available_videos: list[Path],
    already_selected: list[Path],
) -> Path | None:
    remaining = [path for path in available_videos if path not in already_selected]

    choices = [
        Choice(
            title=f"{path.name}    [{_relative_or_absolute(path, root)}]",
            value=path,
        )
        for path in remaining
    ]

    choices.extend(
        [
            Choice(title="Enter a video path manually...", value="__manual__"),
            Choice(title="Finish the session list", value="__done__"),
        ]
    )

    answer = questionary.select(
        "Select the next video in the desired final order:",
        choices=choices,
        use_shortcuts=True,
        use_arrow_keys=True,
    ).ask()

    if answer is None:
        raise KeyboardInterrupt

    if answer == "__done__":
        return None

    if answer == "__manual__":
        while True:
            raw = _ask_text("Video path:")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()

            if not path.is_file():
                print(f"File not found: {path}")
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                print(f"Unsupported video format: {path.suffix}")
                continue
            if path in already_selected:
                print("This video is already in the session list.")
                continue
            return path

    return answer


def _collect_sessions(root: Path, output_dir: Path) -> list[dict]:
    scanned = _scan_videos(output_dir)
    selected_paths: list[Path] = []
    sessions: list[dict] = []

    print()
    print(f"Found {len(scanned)} video(s) in: {output_dir}")
    print("Select one video at a time; selection order becomes the final order.")
    print()

    while True:
        video = _pick_video(root, scanned, selected_paths)

        if video is None:
            if sessions:
                break

            retry = questionary.confirm(
                "No sessions have been added. Continue adding videos?",
                default=True,
            ).ask()
            if retry is None:
                raise KeyboardInterrupt
            if not retry:
                raise ValueError("A course must contain at least one session.")
            continue

        selected_paths.append(video)
        position = len(sessions) + 1
        default_title = _default_title_from_video(video)

        while True:
            number = _ask_int(
                f"Session number for '{video.name}':",
                default=position,
                minimum=1,
            )
            if any(item["number"] == number for item in sessions):
                print(f"Session number {number} is already in use.")
                continue
            break
        title = _ask_text(
            f"Session title for '{video.name}':",
            default=default_title,
        )
        if not title:
            title = default_title or f"Session {number}"

        sessions.append(
            {
                "number": number,
                "title": title,
                "video": _relative_or_absolute(video, root),
            }
        )

        print()
        print("Current session list:")
        for idx, item in enumerate(sessions, start=1):
            print(
                f"  {idx:02d}. Session {item['number']:02d} | "
                f"{item['title']} | {Path(item['video']).name}"
            )
        print()

    return sessions


def _choose_theme(root: Path) -> str | None:
    default_theme = root / "assets" / "bosch_theme.png"

    choices = []
    if default_theme.exists():
        choices.append(
            Choice(
                title=f"Use {default_theme.relative_to(root).as_posix()}",
                value=default_theme,
            )
        )

    choices.extend(
        [
            Choice(title="Enter a theme image path...", value="__manual__"),
            Choice(title="Do not use a theme image", value=None),
        ]
    )

    answer = questionary.select(
        "Theme for the table of contents and session cards:",
        choices=choices,
    ).ask()

    if answer is None:
        raise KeyboardInterrupt

    if answer == "__manual__":
        while True:
            raw = _ask_text("Theme image path:")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()

            if not path.is_file():
                print(f"Image not found: {path}")
                continue
            return _relative_or_absolute(path, root)

    if answer is None:
        return None

    return _relative_or_absolute(answer, root)


def _review_sessions(sessions: list[dict]) -> None:
    print()


def _edit_sessions(sessions: list[dict]) -> list[dict]:
    """Review and mutate sessions without restarting the wizard."""
    while True:
        _review_sessions(sessions)
        action = questionary.select(
            "Review session list:",
            choices=["Continue", "Edit title/number", "Move up", "Move down", "Remove"],
        ).ask()
        if action is None:
            raise KeyboardInterrupt
        if action == "Continue":
            return sessions
        choices = [
            Choice(title=f"{index + 1:02d}. {item['title']}", value=index)
            for index, item in enumerate(sessions)
        ]
        selected = questionary.select("Choose a session:", choices=choices).ask()
        if selected is None:
            continue
        if action == "Edit title/number":
            sessions[selected]["title"] = _ask_text("Session title:", sessions[selected]["title"])
            sessions[selected]["number"] = _ask_int(
                "Session number:", sessions[selected]["number"], minimum=1
            )
        elif action == "Remove":
            if len(sessions) == 1:
                print("A course must contain at least one session.")
            else:
                sessions.pop(selected)
        else:
            destination = selected - 1 if action == "Move up" else selected + 1
            if 0 <= destination < len(sessions):
                sessions[selected], sessions[destination] = (
                    sessions[destination],
                    sessions[selected],
                )
    print("=" * 72)
    print("SESSION ORDER")
    print("=" * 72)
    for position, session in enumerate(sessions, start=1):
        print(
            f"{position:02d}. "
            f"Session {session['number']:02d} | "
            f"{session['title']} | "
            f"{session['video']}"
        )
    print("=" * 72)
    print()


def create_course_config_interactive(
    root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    root = (root or _project_root()).expanduser().resolve()
    output_dir = (output_dir or root / "data" / "output").expanduser().resolve()

    print()
    print("=" * 72)
    print("COURSE CONFIG WIZARD")
    print("=" * 72)
    print(f"Project root : {root}")
    print(f"Video folder : {output_dir}")
    print()

    course_title = _ask_text("Course title:", "Training Course")
    sessions = _collect_sessions(root, output_dir)
    sessions = _edit_sessions(sessions)

    theme_image = _choose_theme(root)

    default_slug = _slugify_filename(course_title)
    config_name = _ask_text(
        "JSON config filename:",
        f"{default_slug}.json",
    )
    config_name = Path(config_name).name
    if not config_name.lower().endswith(".json"):
        config_name += ".json"

    output_name = _ask_text(
        "Final compiled video filename:",
        f"{default_slug}.mp4",
    )
    output_name = Path(output_name).name
    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

    card_duration = _ask_float(
        "Session intro-card duration in seconds:",
        default=5.0,
        minimum=0.1,
    )

    toc_enabled = questionary.confirm(
        "Add a table of contents at the beginning?",
        default=True,
    ).ask()
    if toc_enabled is None:
        raise KeyboardInterrupt

    toc_items_per_page = 8
    toc_page_duration = 5.0
    toc_heading = "TABLE OF CONTENTS"

    if toc_enabled:
        toc_heading = _ask_text("Table-of-contents heading:", "TABLE OF CONTENTS")
        toc_items_per_page = _ask_int(
            "Maximum sessions per table-of-contents page:",
            default=8,
            minimum=1,
        )
        toc_page_duration = _ask_float(
            "Table-of-contents page duration in seconds:",
            default=5.0,
            minimum=0.1,
        )

    add_chapters = questionary.confirm(
        "Add MP4 chapter metadata for session navigation?",
        default=True,
    ).ask()
    if add_chapters is None:
        raise KeyboardInterrupt

    advanced = questionary.confirm(
        "Configure advanced rendering settings?",
        default=False,
    ).ask()
    if advanced is None:
        raise KeyboardInterrupt

    render = {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "video_bitrate": "8M",
        "audio_bitrate": "192k",
        "audio_sample_rate": 48000,
        "video_encoder": "auto",
        "font_path": None,
    }

    if advanced:
        render["width"] = _ask_int("Width:", 1920, 1)
        render["height"] = _ask_int("Height:", 1080, 1)
        render["fps"] = _ask_int("FPS:", 30, 1)
        render["video_bitrate"] = _ask_text("Video bitrate:", "8M")
        render["audio_bitrate"] = _ask_text("Audio bitrate:", "192k")
        render["audio_sample_rate"] = _ask_int(
            "Audio sample rate:",
            48000,
            8000,
        )
        render["video_encoder"] = questionary.select(
            "Video encoder:",
            choices=[
                Choice(title="Auto (prefer NVIDIA NVENC)", value="auto"),
                Choice(title="NVIDIA NVENC", value="h264_nvenc"),
                Choice(title="CPU libx264", value="libx264"),
            ],
        ).ask()
        if render["video_encoder"] is None:
            raise KeyboardInterrupt

        custom_font = questionary.confirm(
            "Use a custom .ttf/.otf font?",
            default=False,
        ).ask()
        if custom_font is None:
            raise KeyboardInterrupt

        if custom_font:
            while True:
                raw = _ask_text("Font path:")
                font_path = Path(raw).expanduser()
                if not font_path.is_absolute():
                    font_path = root / font_path
                font_path = font_path.resolve()

                if not font_path.is_file():
                    print(f"Font not found: {font_path}")
                    continue
                render["font_path"] = _relative_or_absolute(font_path, root)
                break

    config = {
        "title": course_title,
        "output": f"data/compilation/{output_name}",
        "work_dir": "data/compilation",
        "theme_image": theme_image,
        "card_duration": card_duration,
        "add_chapters": bool(add_chapters),
        "toc": {
            "enabled": bool(toc_enabled),
            "heading": toc_heading,
            "items_per_page": toc_items_per_page,
            "page_duration": toc_page_duration,
        },
        "render": render,
        "sessions": sessions,
    }

    config_dir = root / "configs" / "courses"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_name

    if config_path.exists():
        overwrite = questionary.confirm(
            f"{config_path.name} already exists. Overwrite it?",
            default=False,
        ).ask()
        if overwrite is None:
            raise KeyboardInterrupt
        if not overwrite:
            print("Config file creation cancelled.")
            raise SystemExit(0)

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("COURSE CONFIG CREATED")
    print("=" * 72)
    print(config_path)
    print()
    print("Build the course video with:")
    print(
        "uv run transcript-video course build --config "
        f'"{_relative_or_absolute(config_path, root)}"'
    )
    print()

    return config_path

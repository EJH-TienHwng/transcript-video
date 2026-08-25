from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import questionary
    from questionary import Choice
except ImportError as exc:
    raise RuntimeError(
        "Course Config TUI cần package 'questionary'. "
        "Cài bằng: python -m pip install questionary"
    ) from exc

from ..project_config import VIDEO_EXTENSIONS


def _project_root() -> Path:
    """Assume this module lives at <root>/src/course/tui.py."""
    return Path(__file__).resolve().parents[2]


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


def _scan_videos(directory: Path) -> List[Path]:
    if not directory.exists():
        return []

    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _ask_text(message: str, default: Optional[str] = None) -> str:
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
            print("Giá trị phải là số nguyên.")
            continue

        if value < minimum:
            print(f"Giá trị phải >= {minimum}.")
            continue
        return value


def _ask_float(message: str, default: float, minimum: float = 0.0) -> float:
    while True:
        answer = _ask_text(message, str(default))
        try:
            value = float(answer)
        except ValueError:
            print("Giá trị phải là số.")
            continue

        if value < minimum:
            print(f"Giá trị phải >= {minimum}.")
            continue
        return value


def _pick_video(
    root: Path,
    available_videos: List[Path],
    already_selected: List[Path],
) -> Optional[Path]:
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
            Choice(title="Nhập đường dẫn video thủ công...", value="__manual__"),
            Choice(title="Hoàn tất danh sách session", value="__done__"),
        ]
    )

    answer = questionary.select(
        "Chọn video tiếp theo theo đúng thứ tự bạn muốn nối:",
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
            raw = _ask_text("Đường dẫn video:")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()

            if not path.exists():
                print(f"Không tìm thấy file: {path}")
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                print(f"Định dạng video không hỗ trợ: {path.suffix}")
                continue
            if path in already_selected:
                print("Video này đã có trong danh sách.")
                continue
            return path

    return answer


def _collect_sessions(root: Path, output_dir: Path) -> List[Dict]:
    scanned = _scan_videos(output_dir)
    selected_paths: List[Path] = []
    sessions: List[Dict] = []

    print()
    print(f"Tìm thấy {len(scanned)} video trong: {output_dir}")
    print("Mỗi lần chọn một video. Thứ tự chọn chính là thứ tự video cuối.")
    print()

    while True:
        video = _pick_video(root, scanned, selected_paths)

        if video is None:
            if sessions:
                break

            retry = questionary.confirm(
                "Chưa có session nào. Bạn có muốn tiếp tục thêm video?",
                default=True,
            ).ask()
            if retry is None:
                raise KeyboardInterrupt
            if not retry:
                raise ValueError("Course phải có ít nhất 1 session.")
            continue

        selected_paths.append(video)
        position = len(sessions) + 1
        default_title = _default_title_from_video(video)

        number = _ask_int(
            f"Session number cho '{video.name}':",
            default=position,
            minimum=1,
        )
        title = _ask_text(
            f"Session title cho '{video.name}':",
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
        print("Danh sách hiện tại:")
        for idx, item in enumerate(sessions, start=1):
            print(
                f"  {idx:02d}. Session {item['number']:02d} | "
                f"{item['title']} | {Path(item['video']).name}"
            )
        print()

    return sessions


def _choose_theme(root: Path) -> Optional[str]:
    default_theme = root / "assets" / "bosch_theme.png"

    choices = []
    if default_theme.exists():
        choices.append(
            Choice(
                title=f"Dùng {default_theme.relative_to(root).as_posix()}",
                value=default_theme,
            )
        )

    choices.extend(
        [
            Choice(title="Nhập đường dẫn theme image...", value="__manual__"),
            Choice(title="Không dùng theme image", value=None),
        ]
    )

    answer = questionary.select(
        "Theme cho TOC và session cards:",
        choices=choices,
    ).ask()

    if answer is None:
        raise KeyboardInterrupt

    if answer == "__manual__":
        while True:
            raw = _ask_text("Đường dẫn theme image:")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()

            if not path.exists():
                print(f"Không tìm thấy image: {path}")
                continue
            return _relative_or_absolute(path, root)

    if answer is None:
        return None

    return _relative_or_absolute(answer, root)


def _review_sessions(sessions: List[Dict]) -> None:
    print()
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
    root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    root = (root or _project_root()).expanduser().resolve()
    output_dir = (output_dir or root / "data" / "output").expanduser().resolve()

    print()
    print("=" * 72)
    print("COURSE CONFIG TUI")
    print("=" * 72)
    print(f"Project root : {root}")
    print(f"Video folder : {output_dir}")
    print()

    course_title = _ask_text("Tên course:", "Training Course")
    sessions = _collect_sessions(root, output_dir)
    _review_sessions(sessions)

    confirmed = questionary.confirm(
        "Giữ thứ tự session như trên?",
        default=True,
    ).ask()
    if confirmed is None:
        raise KeyboardInterrupt
    if not confirmed:
        print(
            "Hãy chạy lại TUI và chọn video theo thứ tự mong muốn. "
            "TUI cố ý dùng thứ tự chọn để tránh nhập index phức tạp."
        )
        raise SystemExit(0)

    theme_image = _choose_theme(root)

    default_slug = _slugify_filename(course_title)
    config_name = _ask_text(
        "Tên file config JSON:",
        f"{default_slug}.json",
    )
    if not config_name.lower().endswith(".json"):
        config_name += ".json"

    output_name = _ask_text(
        "Tên video tổng hợp cuối:",
        f"{default_slug}.mp4",
    )
    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

    card_duration = _ask_float(
        "Thời lượng intro card mỗi session (giây):",
        default=5.0,
        minimum=0.1,
    )

    toc_enabled = questionary.confirm(
        "Tạo Table of Contents ở đầu video?",
        default=True,
    ).ask()
    if toc_enabled is None:
        raise KeyboardInterrupt

    toc_items_per_page = 8
    toc_page_duration = 5.0
    toc_heading = "TABLE OF CONTENTS"

    if toc_enabled:
        toc_heading = _ask_text("Tiêu đề mục lục:", "TABLE OF CONTENTS")
        toc_items_per_page = _ask_int(
            "Số session tối đa trên mỗi trang TOC:",
            default=8,
            minimum=1,
        )
        toc_page_duration = _ask_float(
            "Thời lượng mỗi trang TOC (giây):",
            default=5.0,
            minimum=0.1,
        )

    add_chapters = questionary.confirm(
        "Thêm MP4 chapter metadata để có thể jump tới session?",
        default=True,
    ).ask()
    if add_chapters is None:
        raise KeyboardInterrupt

    advanced = questionary.confirm(
        "Bạn có muốn chỉnh thông số render nâng cao?",
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

        custom_font = questionary.confirm(
            "Dùng font .ttf/.otf riêng?",
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

                if not font_path.exists():
                    print(f"Không tìm thấy font: {font_path}")
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

    config_dir = root / "courses"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_name

    if config_path.exists():
        overwrite = questionary.confirm(
            f"{config_path.name} đã tồn tại. Ghi đè?",
            default=False,
        ).ask()
        if overwrite is None:
            raise KeyboardInterrupt
        if not overwrite:
            print("Đã hủy ghi file.")
            raise SystemExit(0)

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("ĐÃ TẠO COURSE CONFIG")
    print("=" * 72)
    print(config_path)
    print()
    print("Build video bằng:")
    print(
        f'python -m src.course_builder --config '
        f'"{_relative_or_absolute(config_path, root)}"'
    )
    print()

    return config_path

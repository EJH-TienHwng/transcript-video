from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive terminal UI for creating Course Builder JSON configs."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root. Default: auto-detect from pyproject.toml.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Folder to scan for session videos. Default: <root>/data/output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .tui import create_course_config_interactive

    root = Path(args.root).expanduser().resolve() if args.root else None

    if args.video_dir:
        video_dir = Path(args.video_dir).expanduser()
        if root is not None and not video_dir.is_absolute():
            video_dir = root / video_dir
        video_dir = video_dir.resolve()
    else:
        video_dir = None

    try:
        create_course_config_interactive(
            root=root,
            output_dir=video_dir,
        )
    except KeyboardInterrupt:
        print("\nCourse Config TUI cancelled.")


if __name__ == "__main__":
    main()

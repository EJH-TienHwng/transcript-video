from __future__ import annotations

import argparse
from pathlib import Path

from ..config import configure_binary_path, find_project_root, setup_logging
from .config import load_course_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one training/course video from ordered session videos, "
            "including a TOC and per-session title cards."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to course JSON config. Example: configs/courses/training_course.json",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    from .builder import build_course

    config_path = Path(args.config)
    configure_binary_path(find_project_root(config_path))
    config = load_course_config(config_path)
    build_course(config)


if __name__ == "__main__":
    main()

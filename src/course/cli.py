from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..project_config import setup_logging
from .builder import build_course
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
        help="Path to course JSON config. Example: courses/apb_training.json",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config_path = Path(args.config)
    config = load_course_config(config_path)
    build_course(config)


if __name__ == "__main__":
    main()

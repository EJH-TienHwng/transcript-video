from __future__ import annotations


def main() -> None:
    from ..cli import legacy_course_main

    legacy_course_main("transcript-course-config", "create")


if __name__ == "__main__":
    main()

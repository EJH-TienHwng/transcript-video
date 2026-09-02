"""Course/compilation builder for combining processed session videos."""

from pathlib import Path

from .config import CourseConfig, SessionConfig, load_course_config


def build_course(config: CourseConfig) -> Path:
    """Import rendering dependencies only when a course is actually built."""
    from .builder import build_course as _build_course

    return _build_course(config)


__all__ = [
    "CourseConfig",
    "SessionConfig",
    "build_course",
    "load_course_config",
]

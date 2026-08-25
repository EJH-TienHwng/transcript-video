"""Course/compilation builder for combining processed session videos."""

from .builder import build_course
from .config import CourseConfig, SessionConfig, load_course_config

__all__ = [
    "CourseConfig",
    "SessionConfig",
    "build_course",
    "load_course_config",
]

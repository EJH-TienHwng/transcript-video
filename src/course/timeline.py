from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, List

from .config import CourseConfig, SessionConfig


@dataclass
class SessionTimeline:
    session: SessionConfig
    duration: float
    card_start: float
    content_start: float
    content_end: float

    @property
    def display_number(self) -> int:
        raise RuntimeError(
            "display_number is assigned by build_timeline ordering; use numbered_sessions()."
        )


def format_video_timestamp(seconds: float) -> str:
    """Format a video timestamp as MM:SS or HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02}:{minutes:02}:{secs:02}"
    return f"{minutes:02}:{secs:02}"


def calculate_toc_page_count(config: CourseConfig) -> int:
    if not config.toc.enabled:
        return 0
    return ceil(len(config.sessions) / config.toc.items_per_page)


def calculate_toc_duration(config: CourseConfig) -> float:
    return calculate_toc_page_count(config) * config.toc.page_duration


def build_timeline(
    config: CourseConfig,
    durations: Iterable[float],
) -> List[SessionTimeline]:
    """Calculate the absolute final-video position of every session."""
    duration_list = list(durations)
    if len(duration_list) != len(config.sessions):
        raise ValueError("Số duration không khớp với số session.")

    current_time = calculate_toc_duration(config)
    timeline: List[SessionTimeline] = []

    for session, duration in zip(config.sessions, duration_list):
        if duration <= 0:
            raise ValueError(f"Duration không hợp lệ cho session: {session.title}")

        card_start = current_time
        content_start = card_start + config.card_duration
        content_end = content_start + duration

        timeline.append(
            SessionTimeline(
                session=session,
                duration=duration,
                card_start=card_start,
                content_start=content_start,
                content_end=content_end,
            )
        )
        current_time = content_end

    return timeline


def session_number(session: SessionConfig, position: int) -> int:
    """Use a manually supplied session number, otherwise use list order."""
    return session.number if session.number is not None else position

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

from ..config import SubtitleSegment

BAD_PHRASES = [
    "subtitles by the amara.org community",
    "subtitle by the amara.org community",
    "subtitles by amara.org community",
    "amara.org community",
    "subscribe to la la school",
    "please subscribe to la la school",
    "la la school channel",
    "thanks for watching",
    "thank you for watching",
    "see you next time",
    "cảm ơn các bạn đã theo dõi",
    "cảm ơn bạn đã theo dõi",
    "hẹn gặp lại",
    "đừng quên đăng ký kênh",
    "nhớ đăng ký kênh",
]
# Vietnamese phrases above are matching data, not user-facing interface text.


def format_timestamp(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        seconds = 0.0
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    secs = int(seconds)
    millis = round((seconds - secs) * 1000)
    if millis == 1000:
        millis, secs = 0, secs + 1
    if secs == 60:
        secs, minutes = 0, minutes + 1
    if minutes == 60:
        minutes, hours = 0, hours + 1
    return f"{int(hours):02}:{int(minutes):02}:{secs:02},{millis:03}"


def parse_srt_timestamp(timestamp: str) -> float:
    timestamp = timestamp.strip().replace(".", ",")
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2}),(\d{1,3})", timestamp)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {timestamp}")
    hours, minutes, seconds, millis = match.groups()
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ValueError(f"Invalid SRT timestamp: {timestamp}")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")[:3]) / 1000.0
    )


def normalize_text_for_filter(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w\sÀ-ỹ]", "", text)


def is_bad_hallucination_text(text: str) -> bool:
    normalized = normalize_text_for_filter(text)
    if not normalized:
        return True
    return any(
        phrase_norm and phrase_norm in normalized
        for phrase in BAD_PHRASES
        for phrase_norm in [normalize_text_for_filter(phrase)]
    )


def remove_repeated_hallucination_segments(
    segments: Iterable[SubtitleSegment],
    max_same_text_count: int = 3,
    short_segment_seconds: float = 4.0,
) -> list[SubtitleSegment]:
    cleaned: list[SubtitleSegment] = []
    previous_text = None
    repeat_count = 0
    for segment in segments:
        text = (segment.text or "").strip()
        normalized = normalize_text_for_filter(text)
        duration = max(0.0, segment.end - segment.start)
        if not normalized:
            continue
        if normalized == previous_text:
            repeat_count += 1
        else:
            previous_text = normalized
            repeat_count = 1
        if duration <= short_segment_seconds and repeat_count > max_same_text_count:
            logging.warning(
                "Removed repeated hallucination: %.2f --> %.2f | %s",
                segment.start,
                segment.end,
                text,
            )
            continue
        cleaned.append(segment)
    return cleaned


def fix_too_short_or_invalid_timing(segments: Iterable[SubtitleSegment]) -> list[SubtitleSegment]:
    cleaned, last_end = [], 0.0
    for segment in sorted(segments, key=lambda item: item.start):
        if segment.end <= segment.start:
            logging.warning(
                "Removed invalid timing: %.2f --> %.2f | %s",
                segment.start,
                segment.end,
                segment.text,
            )
            continue
        start, end = (
            max(segment.start, last_end),
            max(segment.end, max(segment.start, last_end) + 0.2),
        )
        last_end = end
        cleaned.append(SubtitleSegment(start, end, segment.text))
    return cleaned


def post_process_segments(segments: Iterable[SubtitleSegment]) -> list[SubtitleSegment]:
    filtered = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        if is_bad_hallucination_text(text):
            logging.warning(
                "Removed known hallucination: %.2f --> %.2f | %s", segment.start, segment.end, text
            )
            continue
        filtered.append(SubtitleSegment(segment.start, segment.end, text))
    return fix_too_short_or_invalid_timing(remove_repeated_hallucination_segments(filtered))


def write_srt(segments: Iterable[SubtitleSegment], srt_path: Path) -> None:
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with srt_path.open("w", encoding="utf-8") as file:
        for index, segment in enumerate(post_process_segments(segments), start=1):
            file.write(
                f"{index}\n{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n{segment.text.strip()}\n\n"
            )


def read_srt(srt_path: Path) -> list[SubtitleSegment]:
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    content = srt_path.read_text(encoding="utf-8-sig")
    segments = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_line_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_line_index is None:
            continue
        start_text, end_text = [part.strip() for part in lines[timing_line_index].split("-->", 1)]
        # SRT permits optional positioning metadata after the end timestamp.
        end_parts = end_text.split(maxsplit=1)
        if not end_parts:
            raise ValueError(f"Invalid SRT timing line: {lines[timing_line_index]}")
        end_text = end_parts[0]
        text = " ".join(lines[timing_line_index + 1 :]).strip()
        if text:
            segments.append(
                SubtitleSegment(
                    parse_srt_timestamp(start_text), parse_srt_timestamp(end_text), text
                )
            )
    return segments

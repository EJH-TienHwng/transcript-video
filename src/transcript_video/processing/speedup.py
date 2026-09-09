from __future__ import annotations

import logging
import math
import re
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..events import EventKind, PipelineStage, emit, ffmpeg_events, stage_context
from ..hardware import get_ffmpeg_exe, video_encoder_args
from ..process_runner import ffmpeg_progress_handler
from .media import get_media_duration_seconds, media_has_audio, run_command

logger = logging.getLogger(__name__)
ALLOWED_SPEEDS = {2, 5, 10}


@dataclass(frozen=True, slots=True)
class SpeedupSegment:
    start: float
    end: float
    speed: int
    label: str | None = None


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    start: float
    end: float
    speed: int = 1
    label: str | None = None


@dataclass(frozen=True, slots=True)
class SpeedupResult:
    source: Path
    spec: Path
    output: Path
    original_duration: float
    expected_duration: float
    actual_duration: float
    time_saved: float
    segments: tuple[SpeedupSegment, ...]


def parse_speedup_timestamp(value: str) -> float:
    if not isinstance(value, str):
        raise ValueError("Speed-up timestamps must be strings such as 00:01:25.500.")
    timestamp = value.strip()
    if not re.fullmatch(r"\d+(?::\d{1,2})?:\d{1,2}(?:[.,]\d{1,3})?", timestamp):
        raise ValueError(f"Invalid speed-up timestamp: {value!r}.")
    timestamp = timestamp.replace(",", ".")
    parts = timestamp.split(":")
    if len(parts) not in {2, 3} or any(not part for part in parts):
        raise ValueError(f"Invalid speed-up timestamp: {value!r}.")
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[0]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError(f"Invalid speed-up timestamp: {value!r}.") from exc
    if (
        hours < 0
        or minutes < 0
        or (len(parts) == 3 and minutes >= 60)
        or not math.isfinite(seconds)
        or seconds < 0
        or seconds >= 60
    ):
        raise ValueError(f"Invalid speed-up timestamp: {value!r}.")
    return hours * 3600 + minutes * 60 + seconds


def parse_speedup_spec(path: Path) -> tuple[SpeedupSegment, ...]:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Speed-up spec is not valid TOML: {path}") from exc
    unknown = sorted(set(raw) - {"segment"})
    if unknown:
        raise ValueError(f"Unknown speed-up spec key(s): {', '.join(unknown)}.")
    items = raw.get("segment", [])
    if not isinstance(items, list):
        raise ValueError("Speed-up spec 'segment' must be an array of TOML tables.")
    segments: list[SpeedupSegment] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Speed-up segment {index} must be a TOML table.")
        unknown = sorted(set(item) - {"start", "end", "speed", "label"})
        missing = sorted({"start", "end", "speed"} - set(item))
        if unknown:
            raise ValueError(f"Unknown key(s) in speed-up segment {index}: {', '.join(unknown)}.")
        if missing:
            raise ValueError(f"Speed-up segment {index} is missing: {', '.join(missing)}.")
        speed = item["speed"]
        label = item.get("label")
        if isinstance(speed, bool) or not isinstance(speed, int):
            raise ValueError(f"Speed-up segment {index} speed must be 2, 5, or 10.")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"Speed-up segment {index} label must be a string.")
        segments.append(
            SpeedupSegment(
                parse_speedup_timestamp(item["start"]),
                parse_speedup_timestamp(item["end"]),
                speed,
                label,
            )
        )
    return tuple(segments)


def validate_speedup_segments(
    segments: tuple[SpeedupSegment, ...] | list[SpeedupSegment], duration: float
) -> tuple[SpeedupSegment, ...]:
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("Final video duration must be a positive finite number.")
    ordered = tuple(sorted(segments, key=lambda item: item.start))
    for index, segment in enumerate(ordered, 1):
        if not math.isfinite(segment.start) or segment.start < 0:
            raise ValueError(f"Speed-up segment {index} start must be at least 0.")
        if not math.isfinite(segment.end) or segment.end <= segment.start:
            raise ValueError(f"Speed-up segment {index} end must be after start.")
        if segment.end > duration:
            raise ValueError(
                f"Speed-up segment {index} ends at {format_timestamp(segment.end)}, "
                f"after video duration {format_timestamp(duration)}."
            )
        if (
            isinstance(segment.speed, bool)
            or not isinstance(segment.speed, int)
            or segment.speed not in ALLOWED_SPEEDS
        ):
            raise ValueError(
                f"Speed-up segment {index} speed must be one of: 2, 5, 10; got {segment.speed}."
            )
        if index > 1 and ordered[index - 2].end > segment.start:
            previous = ordered[index - 2]
            raise ValueError(
                "Speed-up intervals overlap:\n"
                f"{format_timestamp(previous.start)}\N{EN DASH}{format_timestamp(previous.end)}\n"
                f"{format_timestamp(segment.start)}\N{EN DASH}{format_timestamp(segment.end)}"
            )
    return ordered


def build_speedup_timeline(
    segments: tuple[SpeedupSegment, ...] | list[SpeedupSegment], duration: float
) -> tuple[TimelineSegment, ...]:
    timeline: list[TimelineSegment] = []
    cursor = 0.0
    for segment in segments:
        if cursor < segment.start:
            timeline.append(TimelineSegment(cursor, segment.start))
        timeline.append(TimelineSegment(segment.start, segment.end, segment.speed, segment.label))
        cursor = segment.end
    if cursor < duration:
        timeline.append(TimelineSegment(cursor, duration))
    return tuple(timeline)


def calculate_speedup_duration(
    duration: float, segments: tuple[SpeedupSegment, ...] | list[SpeedupSegment]
) -> tuple[float, float]:
    saved = sum((item.end - item.start) * (1 - 1 / item.speed) for item in segments)
    return duration - saved, saved


def build_atempo_chain(speed: int) -> str:
    factors = {2: (2,), 5: (2, 2, 1.25), 10: (2, 2, 2, 1.25)}.get(speed)
    if factors is None:
        raise ValueError("Audio tempo speed must be one of: 2, 5, 10.")
    return ",".join(f"atempo={factor:g}" for factor in factors)


def escape_filter_path(path: Path) -> str:
    """Escape a path embedded in a quoted FFmpeg filter option."""
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def build_overlay_text(segment: TimelineSegment) -> str:
    """Keep user text out of filter syntax; drawtext reads this with expansion disabled."""
    label = f"{segment.label}\n" if segment.label else ""
    return f"{label}Speed up \N{MULTIPLICATION SIGN}{segment.speed}"


def build_speedup_filter_complex(
    timeline: tuple[TimelineSegment, ...], overlay_files: dict[int, Path]
) -> str:
    filters: list[str] = []
    inputs: list[str] = []
    for index, segment in enumerate(timeline):
        start, end = _number(segment.start), _number(segment.end)
        video = f"[0:v:0]trim=start={start}:end={end},setpts=(PTS-STARTPTS)/{segment.speed}"
        audio = f"[0:a:0]atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
        if segment.speed != 1:
            textfile = overlay_files[index]
            video += (
                f",drawtext=textfile='{escape_filter_path(textfile)}':expansion=none:"
                "fontcolor=white:fontsize=24:box=1:boxcolor=black@0.65:"
                "boxborderw=10:x=w-tw-24:y=24"
            )
            audio += f",{build_atempo_chain(segment.speed)}"
        filters.extend((f"{video}[v{index}]", f"{audio}[a{index}]"))
        inputs.append(f"[v{index}][a{index}]")
    filters.append(f"{''.join(inputs)}concat=n={len(timeline)}:v=1:a=1[vout][aout]")
    return ";".join(filters)


def create_speedup_template(path: Path, video_stem: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = (
        f"# Speed-up intervals for {video_stem}\n"
        "#\n# Allowed speed values: 2, 5, 10\n#\n# Example:\n#\n"
        '# [[segment]]\n# start = "00:01:25.000"\n'
        '# end = "00:02:10.000"\n# speed = 5\n#\n'
        '# Optional:\n# label = "Running build..."\n'
    )
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(template)
    except FileExistsError:
        return False
    return True


def process_speedup_video(
    source: Path,
    spec: Path,
    output: Path,
    *,
    video_encoder: str,
    video_stem: str,
) -> SpeedupResult | None:
    if not source.is_file():
        raise FileNotFoundError(
            f"Speed-up requires a final rendered video, but it was not found: {source}"
        )
    if not spec.is_file():
        created = create_speedup_template(spec, video_stem)
        emit(
            PipelineStage.SPEEDUP,
            "Speed-up specification not found.\n"
            f"{'Template created' if created else 'Edit the specification'}: {spec}\n"
            "Add intervals and rerun with --speedup, or use the standalone speedup command.",
            kind=EventKind.WARNING,
            artifact=spec,
            details={"status": "spec_missing", "template_created": created},
        )
        return None

    with stage_context(PipelineStage.SPEEDUP, "Creating speed-up video"):
        duration = get_media_duration_seconds(source)
        if duration is None:
            raise ValueError(f"Could not read final video duration: {source}")
        segments = validate_speedup_segments(parse_speedup_spec(spec), duration)
        if not segments:
            emit(
                PipelineStage.SPEEDUP,
                f"No speed-up intervals configured. Normal video is already available: {source}",
                kind=EventKind.WARNING,
                artifact=source,
                details={"status": "empty_spec"},
            )
            return None
        if not media_has_audio(source):
            raise ValueError(f"Speed-up requires a final video with an audio stream: {source}")
        timeline = build_speedup_timeline(segments, duration)
        expected, saved = calculate_speedup_duration(duration, segments)
        _render_speedup_video(source, output, timeline, video_encoder, expected)
        actual = get_media_duration_seconds(output)
        if actual is None:
            raise ValueError(f"Could not read speed-up output duration: {output}")

    result = SpeedupResult(source, spec, output, duration, expected, actual, saved, segments)
    counts = Counter(segment.speed for segment in segments)
    details = {
        "original_duration": duration,
        "segments": len(segments),
        "expected_duration": expected,
        "actual_duration": actual,
        "time_saved": saved,
        "factor_counts": {f"x{speed}": counts[speed] for speed in sorted(counts)},
    }
    emit(
        PipelineStage.SPEEDUP,
        "Speed-up video ready",
        kind=EventKind.ARTIFACT,
        artifact=output,
        details={"category": "Video", **details},
    )
    emit(PipelineStage.SPEEDUP, "Speed-up duration QA", kind=EventKind.REVIEW, details=details)
    tolerance = max(0.25, expected * 0.005)
    if abs(actual - expected) > tolerance:
        emit(
            PipelineStage.SPEEDUP,
            f"Speed-up duration differs from expected by {abs(actual - expected):.3f}s "
            f"(tolerance {tolerance:.3f}s).",
            kind=EventKind.WARNING,
            details=details,
            artifact=output,
        )
    logger.info(
        "Speed-up complete: original %.3fs, %d segments, output %.3fs, saved %.3fs",
        duration,
        len(segments),
        actual,
        saved,
    )
    return result


def format_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _render_speedup_video(
    source: Path,
    output: Path,
    timeline: tuple[TimelineSegment, ...],
    video_encoder: str,
    expected_duration: float,
) -> None:
    ffmpeg = get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="transcript-video-speedup-") as temporary:
        directory = Path(temporary)
        overlay_files: dict[int, Path] = {}
        for index, segment in enumerate(timeline):
            if segment.speed == 1:
                continue
            path = directory / f"overlay-{index}.txt"
            path.write_text(build_overlay_text(segment), encoding="utf-8")
            overlay_files[index] = path
        graph = build_speedup_filter_complex(timeline, overlay_files)
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            *video_encoder_args(ffmpeg, video_encoder),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output),
        ]
        with ffmpeg_progress_handler(
            ffmpeg_events(PipelineStage.SPEEDUP, "Creating speed-up video", expected_duration)
        ):
            run_command(command)


def _number(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"

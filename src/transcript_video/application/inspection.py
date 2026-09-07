from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

from ..config import ProjectPaths, RunSettings
from ..hardware import get_ffprobe_exe
from ..process_runner import probe_media
from ..processing.models import get_model_filename_suffix


def read_media_metadata(video: Path, cache: dict | None = None) -> dict:
    """Cache probes (including failures) by resolved path, modification time and size."""
    video = video.expanduser().resolve()
    stat = video.stat()
    key = (video, stat.st_mtime_ns, stat.st_size)
    if cache is not None and key in cache:
        result = cache[key]
        if isinstance(result, Exception):
            raise result
        return result
    try:
        result = probe_media(get_ffprobe_exe(), video, timeout=10)
    except (OSError, RuntimeError, ValueError) as exc:
        if cache is not None:
            cache[key] = exc
        raise
    if cache is not None:
        # Keep one entry per path; a replaced file cannot accumulate stale probes.
        for old in list(cache):
            if old[0] == video:
                del cache[old]
        cache[key] = result
    return result


def media_fields(metadata: dict) -> dict[str, str]:
    """Shared, lightweight formatting for CLI tables and Textual metadata/review."""
    video = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"), {})
    info = metadata.get("format", {})

    def number(value, scale=1, suffix=""):
        try:
            result = float(Fraction(str(value))) / scale
            return f"{result:g}{suffix}" if math.isfinite(result) else "Unknown"
        except (ValueError, ZeroDivisionError):
            return "Unknown"

    duration = media_duration(metadata)
    if duration is None:
        duration_label = "Unknown"
    else:
        seconds = round(duration)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        duration_label = (f"{hours:02d}:" if hours else "") + f"{minutes:02d}:{seconds:02d}"
    return {
        "Duration": duration_label,
        "Resolution": f"{video['width']} x {video['height']}"
        if video.get("width") and video.get("height")
        else "Unknown",
        "FPS": number(video.get("avg_frame_rate")),
        "Video codec": str(video.get("codec_name", "Unknown")).upper(),
        "Video bitrate": number(video.get("bit_rate", info.get("bit_rate")), 1_000_000, " Mbps"),
        "Audio codec": str(audio.get("codec_name", "Unknown")).upper(),
        "Sample rate": number(audio.get("sample_rate"), 1000, " kHz"),
        "Channels": str(audio.get("channel_layout", audio.get("channels", "Unknown"))),
        "Size": number(info.get("size"), 1024**2, " MiB"),
    }


def media_duration(metadata: dict) -> float | None:
    try:
        value = float(metadata.get("format", {}).get("duration", 0))
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def inspect_video(
    video: Path, settings: RunSettings, cache: dict | None = None
) -> dict[str, object]:
    video = video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    metadata = read_media_metadata(video, cache)
    root = Path(settings.project.root).expanduser().resolve()
    paths = ProjectPaths.from_root(root)
    model = _from_root(root, settings.project.model)
    translation = (
        _from_root(root, settings.project.translation_model)
        if settings.project.translation_model
        else None
    )
    try:
        suffix = get_model_filename_suffix(model, translation)
    except (OSError, ValueError):
        suffix = None  # Artifact prediction is deliberately best-effort; media inspection is independent.
    artifacts = {
        "subtitles": str(paths.subtitle_dir / f"{video.stem}_{suffix}.srt") if suffix else None,
        "subtitled_video": str(paths.output_dir / f"{video.stem}_vi-dub_en-sub.mp4"),
        "tts_audio": str(paths.audio_dir / f"{video.stem}_tts.wav"),
        "tts_video": str(paths.output_dir / f"{video.stem}_en-dub_en-sub.mp4"),
    }
    return {
        "video": str(video),
        "metadata": metadata,
        "artifacts": artifacts,
        "artifact_states": {
            name: "unresolved" if path is None else "exists" if Path(path).is_file() else "missing"
            for name, path in artifacts.items()
        },
    }


def _from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

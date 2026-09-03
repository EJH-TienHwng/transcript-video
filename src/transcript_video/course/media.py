from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from ..hardware import get_ffmpeg_exe, get_ffprobe_exe, video_encoder_args
from ..process_runner import probe_media, run_ffmpeg
from .config import CourseConfig


def run_command(command: Sequence[str], *, hide_output: bool = False) -> None:
    del hide_output
    run_ffmpeg(command)


def get_media_duration_seconds(media_path: Path) -> float:
    """Read duration from ffprobe JSON."""
    if not media_path.is_file():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    metadata = probe_media(get_ffprobe_exe(), media_path)
    duration = metadata.get("format", {}).get("duration")
    if duration is None:
        raise ValueError(f"Could not read media duration: {media_path}")
    return float(duration)


def media_has_audio(media_path: Path) -> bool:
    metadata = probe_media(get_ffprobe_exe(), media_path)
    return any(stream.get("codec_type") == "audio" for stream in metadata.get("streams", []))


def still_image_to_video(
    image_path: Path,
    video_out: Path,
    duration: float,
    config: CourseConfig,
) -> None:
    """Turn a rendered card PNG into a video segment with silent AAC audio."""
    ffmpeg_path = get_ffmpeg_exe()
    video_out.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(config.render.fps),
        "-i",
        str(image_path),
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={config.render.audio_sample_rate}:cl=stereo",
        "-t",
        f"{duration:.3f}",
        "-vf",
        (
            f"scale={config.render.width}:{config.render.height}:force_original_aspect_ratio=decrease,"
            f"pad={config.render.width}:{config.render.height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={config.render.fps},format=yuv420p"
        ),
        *video_encoder_args(
            ffmpeg_path,
            config.render.video_encoder,
            bitrate=config.render.video_bitrate,
        ),
        "-c:a",
        "aac",
        "-b:a",
        config.render.audio_bitrate,
        "-ar",
        str(config.render.audio_sample_rate),
        "-ac",
        "2",
        "-shortest",
        "-movflags",
        "+faststart",
        str(video_out),
    ]
    run_command(command, hide_output=True)


def normalize_session_video(
    video_in: Path,
    video_out: Path,
    config: CourseConfig,
) -> None:
    """Normalize every session so FFmpeg concat can safely stream-copy them later."""
    if not video_in.is_file():
        raise FileNotFoundError(f"Session video not found: {video_in}")
    if video_in.resolve() == video_out.resolve():
        raise ValueError("Normalization input and output paths must be different.")

    ffmpeg_path = get_ffmpeg_exe()
    video_out.parent.mkdir(parents=True, exist_ok=True)
    duration = get_media_duration_seconds(video_in)

    video_filter = (
        f"scale={config.render.width}:{config.render.height}:force_original_aspect_ratio=decrease,"
        f"pad={config.render.width}:{config.render.height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={config.render.fps},format=yuv420p"
    )

    if media_has_audio(video_in):
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_in),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            video_filter,
            *video_encoder_args(
                ffmpeg_path,
                config.render.video_encoder,
                bitrate=config.render.video_bitrate,
            ),
            "-c:a",
            "aac",
            "-b:a",
            config.render.audio_bitrate,
            "-ar",
            str(config.render.audio_sample_rate),
            "-ac",
            "2",
            "-t",
            f"{duration:.3f}",
            "-movflags",
            "+faststart",
            str(video_out),
        ]
    else:
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_in),
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={config.render.audio_sample_rate}:cl=stereo",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            video_filter,
            "-t",
            f"{duration:.3f}",
            *video_encoder_args(
                ffmpeg_path,
                config.render.video_encoder,
                bitrate=config.render.video_bitrate,
            ),
            "-c:a",
            "aac",
            "-b:a",
            config.render.audio_bitrate,
            "-ar",
            str(config.render.audio_sample_rate),
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(video_out),
        ]

    run_command(command, hide_output=True)


def _concat_escape(path: Path) -> str:
    # FFmpeg concat list uses single-quoted paths. Escape embedded single quotes.
    return str(path.resolve()).replace("\\", "/").replace("'", r"'\''")


def concatenate_videos(
    video_paths: Iterable[Path],
    output_path: Path,
    list_path: Path,
) -> None:
    paths = list(video_paths)
    if not paths:
        raise ValueError("No video segments were provided for concatenation.")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing concatenation segment: {path}")
    if output_path.resolve() in {path.resolve() for path in paths}:
        raise ValueError("Concatenation output must not overwrite an input segment.")

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "".join(f"file '{_concat_escape(path)}'\n" for path in paths),
        encoding="utf-8",
    )

    ffmpeg_path = get_ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_command(command)


def add_chapter_metadata(
    video_in: Path,
    video_out: Path,
    chapter_file: Path,
    chapter_starts: Sequence[float],
    chapter_titles: Sequence[str],
) -> None:
    """Copy the compiled MP4 while attaching FFmpeg chapter metadata."""
    if video_in.resolve() == video_out.resolve():
        raise ValueError("Chapter output must not overwrite the input video in place.")
    if len(chapter_starts) != len(chapter_titles):
        raise ValueError("chapter_starts and chapter_titles must have the same length.")
    if not chapter_starts:
        shutil_copy(video_in, video_out)
        return

    total_duration = get_media_duration_seconds(video_in)
    lines: list[str] = [";FFMETADATA1"]

    for index, (start, title) in enumerate(zip(chapter_starts, chapter_titles, strict=True)):
        end = chapter_starts[index + 1] if index + 1 < len(chapter_starts) else total_duration
        start_ms = max(0, round(start * 1000))
        end_ms = max(start_ms + 1, round(end * 1000))

        safe_title = (
            title.replace("\\", "\\\\")
            .replace("=", r"\=")
            .replace(";", r"\;")
            .replace("#", r"\#")
            .replace("\n", " ")
        )

        lines.extend(
            [
                "",
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={safe_title}",
            ]
        )

    chapter_file.parent.mkdir(parents=True, exist_ok=True)
    chapter_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ffmpeg_path = get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_in),
        "-i",
        str(chapter_file),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-map_metadata",
        "1",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(video_out),
    ]
    run_command(command)


def shutil_copy(source: Path, destination: Path) -> None:
    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

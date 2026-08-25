from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import imageio_ffmpeg

from .config import CourseConfig


def run_command(command: Sequence[str], *, hide_output: bool = False) -> None:
    kwargs = {"check": True}
    if hide_output:
        kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})
    subprocess.run(list(command), **kwargs)


def get_media_duration_seconds(media_path: Path) -> float:
    """Read duration by parsing FFmpeg's input probe output."""
    if not media_path.exists():
        raise FileNotFoundError(f"Không tìm thấy media: {media_path}")

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.run(
        [ffmpeg_path, "-i", str(media_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    output = process.stderr or process.stdout or ""
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        raise ValueError(f"Không đọc được duration của: {media_path}")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def media_has_audio(media_path: Path) -> bool:
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.run(
        [ffmpeg_path, "-i", str(media_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    output = process.stderr or process.stdout or ""
    return bool(re.search(r"Stream #.*Audio:", output))


def still_image_to_video(
    image_path: Path,
    video_out: Path,
    duration: float,
    config: CourseConfig,
) -> None:
    """Turn a rendered card PNG into a video segment with silent AAC audio."""
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    video_out.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-loop", "1",
        "-framerate", str(config.render.fps),
        "-i", str(image_path),
        "-f", "lavfi",
        "-i", f"anullsrc=r={config.render.audio_sample_rate}:cl=stereo",
        "-t", f"{duration:.3f}",
        "-vf", (
            f"scale={config.render.width}:{config.render.height}:force_original_aspect_ratio=decrease,"
            f"pad={config.render.width}:{config.render.height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={config.render.fps},format=yuv420p"
        ),
        "-c:v", "libx264",
        "-preset", "medium",
        "-b:v", config.render.video_bitrate,
        "-c:a", "aac",
        "-b:a", config.render.audio_bitrate,
        "-ar", str(config.render.audio_sample_rate),
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(video_out),
    ]
    run_command(command, hide_output=True)


def normalize_session_video(
    video_in: Path,
    video_out: Path,
    config: CourseConfig,
) -> None:
    """Normalize every session so FFmpeg concat can safely stream-copy them later."""
    if not video_in.exists():
        raise FileNotFoundError(f"Không tìm thấy session video: {video_in}")

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    video_out.parent.mkdir(parents=True, exist_ok=True)

    video_filter = (
        f"scale={config.render.width}:{config.render.height}:force_original_aspect_ratio=decrease,"
        f"pad={config.render.width}:{config.render.height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={config.render.fps},format=yuv420p"
    )

    if media_has_audio(video_in):
        command = [
            ffmpeg_path,
            "-y",
            "-i", str(video_in),
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-vf", video_filter,
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", config.render.video_bitrate,
            "-c:a", "aac",
            "-b:a", config.render.audio_bitrate,
            "-ar", str(config.render.audio_sample_rate),
            "-ac", "2",
            "-movflags", "+faststart",
            str(video_out),
        ]
    else:
        duration = get_media_duration_seconds(video_in)
        command = [
            ffmpeg_path,
            "-y",
            "-i", str(video_in),
            "-f", "lavfi",
            "-i", f"anullsrc=r={config.render.audio_sample_rate}:cl=stereo",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-vf", video_filter,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", config.render.video_bitrate,
            "-c:a", "aac",
            "-b:a", config.render.audio_bitrate,
            "-ar", str(config.render.audio_sample_rate),
            "-ac", "2",
            "-shortest",
            "-movflags", "+faststart",
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
        raise ValueError("Không có video segment nào để concat.")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Thiếu concat segment: {path}")

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "".join(f"file '{_concat_escape(path)}'\n" for path in paths),
        encoding="utf-8",
    )

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        "-movflags", "+faststart",
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
    if len(chapter_starts) != len(chapter_titles):
        raise ValueError("chapter_starts và chapter_titles không cùng số lượng.")
    if not chapter_starts:
        shutil_copy(video_in, video_out)
        return

    total_duration = get_media_duration_seconds(video_in)
    lines: List[str] = [";FFMETADATA1"]

    for index, (start, title) in enumerate(zip(chapter_starts, chapter_titles)):
        end = chapter_starts[index + 1] if index + 1 < len(chapter_starts) else total_duration
        start_ms = max(0, int(round(start * 1000)))
        end_ms = max(start_ms + 1, int(round(end * 1000)))

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

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i", str(video_in),
        "-i", str(chapter_file),
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-map_metadata", "1",
        "-c", "copy",
        "-movflags", "+faststart",
        str(video_out),
    ]
    run_command(command)


def shutil_copy(source: Path, destination: Path) -> None:
    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from ..config import VIDEO_EXTENSIONS
from ..hardware import get_ffmpeg_exe, get_ffprobe_exe, video_encoder_args
from ..process_runner import ProcessExecutionError, probe_media, run_ffmpeg

logger = logging.getLogger(__name__)


def escape_subtitle_path_for_ffmpeg(path: Path) -> str:
    """Escape subtitle path so FFmpeg subtitles filter can read Windows paths."""
    escaped = str(path.resolve()).replace("\\", "/")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    return escaped


def run_command(command: Sequence[str], *, hide_output: bool = False) -> None:
    """Run a subprocess command with error checking."""
    del hide_output
    run_ffmpeg(command)


def ensure_ffmpeg_available_for_transformers() -> None:
    """Make the selected FFmpeg executable visible to Transformers/audio loaders."""
    ffmpeg_path = Path(get_ffmpeg_exe()).resolve()
    ffmpeg_dir = str(ffmpeg_path.parent)
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg_path)

    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path

    logger.info("FFmpeg available for Transformers: %s", ffmpeg_path)


def burn_subtitles(
    video_in: Path,
    srt_in: Path,
    video_out: Path,
    video_encoder: str = "auto",
) -> None:
    """Burn hard subtitles into a video using FFmpeg."""
    video_out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = get_ffmpeg_exe()
    srt_escaped = escape_subtitle_path_for_ffmpeg(srt_in)

    # Default subtitle style with white text and black outline. You can customize this as needed.
    subtitle_style = (
        "FontName=Arial,"
        "FontSize=16,"
        "PrimaryColour=&H00FFFFFF,"
        "SecondaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H00000000,"
        "Bold=0,"
        "Italic=0,"
        "Underline=0,"
        "StrikeOut=0,"
        "ScaleX=100,"
        "ScaleY=100,"
        "Spacing=0,"
        "Angle=0,"
        "BorderStyle=1,"
        "Outline=1,"
        "Shadow=0,"
        "Alignment=2,"  # Bottom-center
        "MarginL=10,"
        "MarginR=10,"
        "MarginV=25,"  # Adjust vertical margin to move subtitles up/down
        "Encoding=1"
    )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_in),
        "-vf",
        # f"subtitles='{srt_escaped}':force_style='{subtitle_style}'",
        f"subtitles='{srt_escaped}':force_style='MarginV=25'",
        *video_encoder_args(ffmpeg_path, video_encoder),
        "-c:a",
        "copy",
        str(video_out),
    ]
    run_command(command)


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract mono 16 kHz WAV audio from video for Hugging Face pipeline."""
    ffmpeg_path = get_ffmpeg_exe()
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    run_command(command, hide_output=True)


def read_wav_as_float32_array(audio_path: Path):
    """Read extracted WAV with Python stdlib and return data for HF pipeline."""
    import wave

    import numpy as np

    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        raw_audio = wav_file.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width: {sample_width}")

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return {"array": audio, "sampling_rate": sample_rate}


def find_videos(input_dir: Path, selected_video: str | None = None) -> list[Path]:
    """Find videos in input folder. If selected_video is given, process only that file."""
    if selected_video:
        input_root = input_dir.resolve()
        video_path = (input_root / selected_video).resolve()
        try:
            video_path.relative_to(input_root)
        except ValueError as exc:
            raise ValueError("--video must point to a file inside data/input.") from exc
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {video_path.name}")
        return [video_path]

    videos = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {input_dir}")
    return videos


def get_media_duration_seconds(media_path: Path) -> float | None:
    """Return media duration from ffprobe JSON."""
    try:
        metadata = probe_media(get_ffprobe_exe(), media_path)
        duration = metadata.get("format", {}).get("duration")
        return float(duration) if duration is not None else None
    except (FileNotFoundError, ValueError, TypeError, ProcessExecutionError):
        return None


def split_audio_into_chunks(
    audio_in: Path,
    output_dir: Path,
    chunk_minutes: int = 5,
    overwrite: bool = True,
) -> None:
    """Split a long audio file into smaller review chunks.

    This only splits the generated TTS WAV file for easier checking.
    It does not split the video, subtitle, or original input audio.
    """
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")
    if not audio_in.exists():
        raise FileNotFoundError(f"Audio file to split was not found: {audio_in}")

    ffmpeg_path = get_ffmpeg_exe()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove old chunks first. This avoids leaving stale part_010.wav, part_011.wav, ...
    # from a previous longer generation.
    if overwrite:
        for old_chunk in output_dir.glob(f"{audio_in.stem}_part_*{audio_in.suffix}"):
            try:
                old_chunk.unlink()
            except OSError:
                logger.warning("Could not delete stale audio chunk: %s", old_chunk)

    chunk_seconds = int(chunk_minutes * 60)
    output_pattern = output_dir / f"{audio_in.stem}_part_%03d{audio_in.suffix}"

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(audio_in),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(output_pattern),
    ]
    run_command(command)
    logger.info("Split TTS audio into %d-minute chunks: %s", chunk_minutes, output_dir)


def mux_audio_into_video_replace(video_in: Path, audio_in: Path, video_out: Path) -> None:
    """Replace the video's original audio with generated TTS audio."""
    ffmpeg_path = get_ffmpeg_exe()
    video_duration = get_media_duration_seconds(video_in)
    if video_duration is None:
        raise ValueError(f"Could not read video duration: {video_in}")
    video_out.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_in),
        "-i",
        str(audio_in),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{video_duration:.3f}",
        str(video_out),
    ]
    run_command(command)


def mux_audio_into_video_mix(video_in: Path, audio_in: Path, video_out: Path) -> None:
    """Mix original video audio with generated TTS audio."""
    ffmpeg_path = get_ffmpeg_exe()
    video_duration = get_media_duration_seconds(video_in)
    if video_duration is None:
        raise ValueError(f"Could not read video duration: {video_in}")
    video_out.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_in),
        "-i",
        str(audio_in),
        "-filter_complex",
        "[0:a:0][1:a:0]amix=inputs=2:duration=longest:dropout_transition=2[aout]",
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{video_duration:.3f}",
        str(video_out),
    ]
    run_command(command)

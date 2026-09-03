from __future__ import annotations

import logging
import os
import shutil
from functools import cache
from pathlib import Path

from .process_runner import ProcessExecutionError, run_process

logger = logging.getLogger(__name__)


def get_ffmpeg_exe() -> str:
    """Prefer a user-supplied FFmpeg and fall back to imageio-ffmpeg."""
    configured = os.environ.get("TRANSCRIPT_VIDEO_FFMPEG")
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        if not configured_path.is_file():
            raise FileNotFoundError(
                f"TRANSCRIPT_VIDEO_FFMPEG does not point to a file: {configured_path}"
            )
        return str(configured_path)

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return str(Path(system_ffmpeg).resolve())

    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def get_ffprobe_exe() -> str:
    """Locate ffprobe next to configured FFmpeg or on PATH."""
    configured = os.environ.get("TRANSCRIPT_VIDEO_FFPROBE")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"TRANSCRIPT_VIDEO_FFPROBE does not point to a file: {path}")
    system = shutil.which("ffprobe")
    if system:
        return str(Path(system).resolve())
    sibling = Path(get_ffmpeg_exe()).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError("ffprobe was not found on PATH or next to FFmpeg.")


def resolve_torch_device(requested: str, workload: str) -> str:
    """Prefer CUDA for model inference and configure safe GPU optimizations."""
    import torch

    if requested == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_name = torch.cuda.get_device_name(0)
        logger.info("%s device: CUDA (%s)", workload, device_name)
        return "cuda"

    if requested == "cuda":
        logger.warning("CUDA is unavailable for %s; falling back to CPU.", workload)
    else:
        logger.info("%s device: CPU (explicitly configured)", workload)
    return "cpu"


@cache
def ffmpeg_encoder_available(ffmpeg_path: str, encoder: str) -> bool:
    """Probe an FFmpeg encoder, including the driver and physical GPU path."""
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=256x256:rate=1",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    ]
    try:
        run_process(command, timeout=15)
    except (OSError, TimeoutError, ProcessExecutionError) as exc:
        logger.debug(
            "FFmpeg encoder probe failed for %s: %s",
            encoder,
            exc,
        )
        return False
    return True


def resolve_video_encoder(ffmpeg_path: str, requested: str) -> str:
    """Choose NVENC whenever available, with a reliable software fallback."""
    if requested == "libx264":
        logger.info("FFmpeg video encoder: libx264 (explicitly configured)")
        return "libx264"

    if ffmpeg_encoder_available(ffmpeg_path, "h264_nvenc"):
        logger.info("FFmpeg video encoder: h264_nvenc (NVIDIA GPU)")
        return "h264_nvenc"

    if requested == "h264_nvenc":
        logger.warning(
            "h264_nvenc was requested but the FFmpeg/NVIDIA runtime probe failed; "
            "falling back to libx264."
        )
    else:
        logger.info("NVENC is unavailable; FFmpeg video encoder: libx264")
    return "libx264"


def video_encoder_args(
    ffmpeg_path: str,
    requested: str,
    *,
    bitrate: str | None = None,
) -> list[str]:
    """Return FFmpeg H.264 options for the selected hardware/software encoder."""
    encoder = resolve_video_encoder(ffmpeg_path, requested)
    if encoder == "h264_nvenc":
        quality_args = ["-b:v", bitrate] if bitrate else ["-cq", "23", "-b:v", "0"]
        return ["-c:v", encoder, "-preset", "p4", *quality_args, "-pix_fmt", "yuv420p"]

    quality_args = ["-b:v", bitrate] if bitrate else ["-crf", "23"]
    return ["-c:v", encoder, "-preset", "medium", *quality_args, "-pix_fmt", "yuv420p"]

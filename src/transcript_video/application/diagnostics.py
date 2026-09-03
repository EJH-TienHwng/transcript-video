from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import ProjectPaths, RunSettings
from ..hardware import ffmpeg_encoder_available, get_ffmpeg_exe, get_ffprobe_exe
from ..process_runner import run_process


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_doctor(settings: RunSettings) -> list[Check]:
    root = Path(settings.project.root).expanduser().resolve()
    paths = ProjectPaths.from_root(root)
    checks = [
        Check("Python", sys.version_info[:2] == (3, 12), sys.version.split()[0]),
        Check("Project root", root.is_dir(), str(root)),
        Check("Free storage", _free_space(root) >= 2 * 1024**3, _format_bytes(_free_space(root))),
    ]
    try:
        ffmpeg = get_ffmpeg_exe()
        checks.append(Check("FFmpeg", True, ffmpeg))
        try:
            filters = run_process([ffmpeg, "-hide_banner", "-filters"]).stdout
            checks.append(Check("Subtitle filter", "subtitles" in filters, "libass subtitles"))
        except Exception as exc:
            checks.append(Check("Subtitle filter", False, str(exc)))
        checks.append(
            Check(
                "Video encoder",
                ffmpeg_encoder_available(ffmpeg, "h264_nvenc")
                or ffmpeg_encoder_available(ffmpeg, "libx264"),
                "h264_nvenc" if ffmpeg_encoder_available(ffmpeg, "h264_nvenc") else "libx264",
            )
        )
    except Exception as exc:
        checks.append(Check("FFmpeg", False, str(exc)))
    try:
        checks.append(Check("ffprobe", True, get_ffprobe_exe()))
    except Exception as exc:
        checks.append(Check("ffprobe", False, str(exc)))
    model = _from_root(root, settings.project.model)
    checks.append(Check("Transcription model", model.is_dir(), str(model)))
    for name in ("input_dir", "subtitle_dir", "audio_dir", "output_dir", "temp_dir"):
        folder = getattr(paths, name)
        parent = next((item for item in (folder, *folder.parents) if item.exists()), root)
        checks.append(Check(f"Writable {name}", _writable(parent), str(folder)))
    try:
        import torch

        available = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if available else "CUDA unavailable"
        checks.append(Check("CUDA", available, detail, required=settings.hardware.device == "cuda"))
    except Exception as exc:
        checks.append(
            Check("PyTorch", False, str(exc), required=settings.hardware.device == "cuda")
        )
    return checks


def _from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _writable(path: Path) -> bool:
    return path.is_dir() and bool(path.stat().st_mode & 0o200)


def _free_space(path: Path) -> int:
    existing = next(
        (candidate for candidate in (path, *path.parents) if candidate.exists()), Path.cwd()
    )
    return shutil.disk_usage(existing).free


def _format_bytes(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB free"

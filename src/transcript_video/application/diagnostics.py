from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import ProjectPaths, RunSettings
from ..hardware import (
    ffmpeg_encoder_available,
    flash_attention_status,
    get_ffmpeg_exe,
    get_ffprobe_exe,
)
from ..process_runner import run_process
from .settings import validate_settings


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
        _python_check(root),
        Check("Project root", root.is_dir(), str(root)),
        Check("Free storage", _free_space(root) >= 2 * 1024**3, _format_bytes(_free_space(root))),
    ]
    try:
        validate_settings(settings)
        checks.append(Check("Configuration", True, "Settings valid"))
    except ValueError as exc:
        checks.append(Check("Configuration", False, str(exc)))
    try:
        ffmpeg = get_ffmpeg_exe()
        checks.append(Check("FFmpeg", True, ffmpeg))
        try:
            filters = run_process([ffmpeg, "-hide_banner", "-filters"]).stdout
            checks.append(Check("Subtitle filter", "subtitles" in filters, "libass subtitles"))
        except Exception as exc:
            checks.append(Check("Subtitle filter", False, str(exc)))
        encoders = run_process([ffmpeg, "-hide_banner", "-encoders"], timeout=15).stdout
        nvenc = ffmpeg_encoder_available(ffmpeg, "h264_nvenc")
        fallback = ffmpeg_encoder_available(ffmpeg, "libx264")
        requested = settings.hardware.video_encoder
        configured_ok = (
            fallback
            if requested == "libx264"
            else nvenc
            if requested == "h264_nvenc"
            else nvenc or fallback
        )
        checks.extend(
            [
                Check("Configured encoder", configured_ok, requested),
                Check(
                    "NVENC encoder availability",
                    "h264_nvenc" in encoders,
                    "h264_nvenc",
                    required=False,
                ),
                Check("NVENC runtime usability", nvenc, "One-frame runtime probe", required=False),
                Check(
                    "Fallback encoder",
                    fallback,
                    "libx264",
                    required=requested == "libx264" or not nvenc,
                ),
            ]
        )
    except Exception as exc:
        checks.append(Check("FFmpeg", False, str(exc)))
    try:
        checks.append(Check("ffprobe", True, get_ffprobe_exe()))
    except Exception as exc:
        checks.append(Check("ffprobe", False, str(exc)))
    model = _from_root(root, settings.project.model)
    checks.append(Check("Transcription model", model.is_dir(), str(model)))
    if settings.tts.enabled:
        supported, detail = flash_attention_status(settings.hardware.device)
        checks.append(Check("FlashAttention 2 (optional)", supported, detail, required=False))
        tts_model = _from_root(root, settings.tts.model)
        checks.append(Check("TTS model", tts_model.is_dir(), str(tts_model)))
        checks.append(
            Check(
                "TTS output directory",
                _writable(
                    next(p for p in (paths.audio_dir, *paths.audio_dir.parents) if p.exists())
                ),
                str(paths.audio_dir),
            )
        )
        checks.append(
            Check(
                "TTS configuration",
                True,
                f"{settings.tts.generation_mode} / {settings.tts.mode}; {settings.tts.language}; {settings.tts.speaker}",
            )
        )
    for name in (
        "input_dir",
        "source_subtitle_dir",
        "translated_subtitle_dir",
        "audio_dir",
        "output_dir",
        "temp_dir",
        "report_dir",
    ):
        folder = getattr(paths, name)
        parent = next((item for item in (folder, *folder.parents) if item.exists()), root)
        checks.append(Check(f"Writable {name}", _writable(parent), str(folder)))
    try:
        import torch

        checks.append(Check("PyTorch version", True, str(torch.__version__)))
        checks.append(
            Check(
                "PyTorch CUDA build",
                bool(torch.version.cuda),
                str(torch.version.cuda or "CPU build"),
                required=settings.hardware.device == "cuda",
            )
        )
        available = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if available else "CUDA unavailable"
        checks.append(Check("CUDA", available, detail, required=settings.hardware.device == "cuda"))
    except Exception as exc:
        checks.append(
            Check("PyTorch", False, str(exc), required=settings.hardware.device == "cuda")
        )
    try:
        import ctranslate2

        device = settings.hardware.device
        support = ctranslate2.get_supported_compute_types(device)
        checks.append(
            Check(
                "ASR compute support",
                settings.hardware.compute_type in support
                or settings.hardware.compute_type in {"auto", "default"},
                ", ".join(sorted(support)),
                required=False,
            )
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        checks.append(Check("ASR compute support", False, str(exc), required=False))
    return checks


def _python_check(root: Path) -> Check:
    version = sys.version.split()[0]
    source = root / ".python-version"
    try:
        expected = source.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        return Check("Python", False, f"{version}; cannot read {source}: {exc}", required=False)
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", expected):
        return Check(
            "Python", False, f"{version}; invalid version in {source}: {expected!r}", required=False
        )
    parts = tuple(int(part) for part in expected.split("."))
    return Check(
        "Python",
        sys.version_info[: len(parts)] == parts,
        f"{version} (project requires {expected}, from {source})",
    )


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

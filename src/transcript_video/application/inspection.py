from __future__ import annotations

from pathlib import Path

from ..config import ProjectPaths, RunSettings
from ..hardware import get_ffprobe_exe
from ..process_runner import probe_media
from ..processing.models import get_model_filename_suffix


def inspect_video(video: Path, settings: RunSettings) -> dict[str, object]:
    video = video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    metadata = probe_media(get_ffprobe_exe(), video)
    root = Path(settings.project.root).expanduser().resolve()
    paths = ProjectPaths.from_root(root)
    model = _from_root(root, settings.project.model)
    translation = (
        _from_root(root, settings.project.translation_model)
        if settings.project.translation_model
        else None
    )
    suffix = get_model_filename_suffix(model, translation)
    artifacts = {
        "subtitles": str(paths.subtitle_dir / f"{video.stem}_{suffix}.srt"),
        "subtitled_video": str(paths.output_dir / f"{video.stem}_vi-dub_en-sub.mp4"),
        "tts_audio": str(paths.audio_dir / f"{video.stem}_tts.wav"),
        "tts_video": str(paths.output_dir / f"{video.stem}_en-dub_en-sub.mp4"),
    }
    return {"video": str(video), "metadata": metadata, "artifacts": artifacts}


def _from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

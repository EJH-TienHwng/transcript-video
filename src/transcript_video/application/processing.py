from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import ProjectPaths, RunSettings, configure_binary_path
from ..events import NullObserver, PipelineObserver
from ..processing.media import find_videos
from ..processing.models import get_model_filename_suffix
from ..processing.pipeline import process_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessPlan:
    settings: RunSettings
    root: Path
    paths: ProjectPaths
    model: Path
    translation_model: Path | None
    videos: tuple[Path, ...]
    artifacts: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RunSummary:
    succeeded: int
    total: int
    elapsed_seconds: float
    failures: tuple[str, ...]
    artifacts: tuple[Path, ...]


def build_process_plan(settings: RunSettings) -> ProcessPlan:
    root = Path(settings.project.root).expanduser().resolve()
    paths = ProjectPaths.from_root(root)
    model = _from_root(root, settings.project.model)
    translation = (
        _from_root(root, settings.project.translation_model)
        if settings.project.translation_model
        else None
    )
    requested = settings.project.video
    if requested and Path(requested).is_absolute():
        videos = (Path(requested).resolve(),)
        if not videos[0].is_file():
            raise FileNotFoundError(f"Video not found: {videos[0]}")
    else:
        videos = tuple(find_videos(paths.input_dir, requested))
    try:
        suffix = get_model_filename_suffix(model, translation)
    except (FileNotFoundError, ValueError):
        suffix = "<model-suffix>"
    artifacts = tuple(
        artifact for video in videos for artifact in _artifacts_for(video, suffix, paths, settings)
    )
    return ProcessPlan(settings, root, paths, model, translation, videos, artifacts)


def execute_process_plan(plan: ProcessPlan, observer: PipelineObserver | None = None) -> RunSummary:
    plan.paths.create_dirs()
    configure_binary_path(plan.root)
    if not plan.model.is_dir():
        raise FileNotFoundError(f"Model folder not found: {plan.model}")
    if plan.translation_model is not None and not plan.translation_model.is_dir():
        raise FileNotFoundError(f"Translation model folder not found: {plan.translation_model}")
    events = observer or NullObserver()
    failures: list[str] = []
    started = time.perf_counter()
    for video in plan.videos:
        try:
            process_video(
                video,
                plan.model,
                plan.translation_model,
                plan.paths,
                plan.settings,
                events,
            )
        except Exception as exc:
            logger.debug("Video processing failed", exc_info=True)
            failures.append(f"{video.name}: {exc}")
    elapsed = time.perf_counter() - started
    existing = tuple(path for path in plan.artifacts if path.exists())
    return RunSummary(
        len(plan.videos) - len(failures), len(plan.videos), elapsed, tuple(failures), existing
    )


def _artifacts_for(
    video: Path, suffix: str, paths: ProjectPaths, settings: RunSettings
) -> list[Path]:
    artifacts = [paths.subtitle_dir / f"{video.stem}_{suffix}.srt"]
    if not settings.transcription.skip_burn:
        artifacts.append(paths.output_dir / f"{video.stem}_vi-dub_en-sub.mp4")
    if settings.tts.enabled:
        artifacts.extend(
            [
                paths.audio_dir / f"{video.stem}_tts.wav",
                paths.output_dir / f"{video.stem}_en-dub_en-sub.mp4",
            ]
        )
    return artifacts


def _from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

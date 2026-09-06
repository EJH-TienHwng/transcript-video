from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import VIDEO_EXTENSIONS, ProjectPaths, RunSettings, configure_binary_path
from ..context import log_context
from ..events import EventKind, PipelineObserver, PipelineStage, emit, event_scope, stage_context
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
    errors: tuple[Exception, ...] = ()


def build_process_plan(settings: RunSettings, videos: Sequence[Path] | None = None) -> ProcessPlan:
    root = Path(settings.project.root).expanduser().resolve()
    paths = ProjectPaths.from_root(root)
    model = _from_root(root, settings.project.model)
    translation = (
        _from_root(root, settings.project.translation_model)
        if settings.project.translation_model
        else None
    )
    requested = videos or ([Path(settings.project.video)] if settings.project.video else [])
    selected: list[Path] = []
    for video in requested:
        video = video.expanduser()
        if video.is_absolute():
            video = video.resolve()
            if not video.is_file():
                raise FileNotFoundError(f"Video not found: {video}")
            if video.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"Unsupported video format: {video.name}")
        else:
            video = find_videos(paths.input_dir, str(video))[0]
        selected.append(video)
    resolved_videos = (
        tuple(dict.fromkeys(selected)) if requested else tuple(find_videos(paths.input_dir))
    )
    # Output and cache names use the stem; two distinct inputs must not overwrite each other.
    stems: dict[str, Path] = {}
    for video in resolved_videos:
        if video.stem.casefold() in stems:
            raise ValueError(
                f"Videos share an output name: {stems[video.stem.casefold()]} and {video}; rename one input."
            )
        stems[video.stem.casefold()] = video
    try:
        suffix = get_model_filename_suffix(model, translation)
    except (FileNotFoundError, ValueError):
        suffix = "<model-suffix>"
    artifacts = tuple(
        artifact
        for video in resolved_videos
        for artifact in _artifacts_for(video, suffix, paths, settings)
    )
    return ProcessPlan(settings, root, paths, model, translation, resolved_videos, artifacts)


def execute_process_plan(plan: ProcessPlan, observer: PipelineObserver | None = None) -> RunSummary:
    failures: list[str] = []
    errors: list[Exception] = []
    started = time.perf_counter()
    with (
        event_scope(observer),
        stage_context(PipelineStage.RUN, "Processing videos", total=len(plan.videos)),
    ):
        plan.paths.create_dirs()
        configure_binary_path(plan.root)
        if not plan.model.is_dir():
            raise FileNotFoundError(f"Model folder not found: {plan.model}")
        if plan.translation_model is not None and not plan.translation_model.is_dir():
            raise FileNotFoundError(f"Translation model folder not found: {plan.translation_model}")
        for position, video in enumerate(plan.videos, 1):
            with log_context(
                video=video.name, stage="video", operation=None, chunk=None, subtitle=None
            ):
                stages = ["subtitles"]
                if not plan.settings.transcription.skip_burn:
                    stages += ["render"] + (["tts", "mux"] if plan.settings.tts.enabled else [])
                video_started = time.perf_counter()
                emit(
                    PipelineStage.VIDEO,
                    f"[{position}/{len(plan.videos)}] {video.name}",
                    kind=EventKind.START,
                    current=position - 1,
                    total=len(plan.videos),
                    details={"stages": stages},
                )
                try:
                    process_video(
                        video,
                        plan.model,
                        plan.translation_model,
                        plan.paths,
                        plan.settings,
                        observer,
                    )
                except Exception as exc:
                    logger.debug(
                        "Video processing failed", exc_info=True, extra={"diagnostic_only": True}
                    )
                    failures.append(f"{video.name}: {exc}")
                    errors.append(exc)
                    emit(
                        PipelineStage.VIDEO,
                        str(exc),
                        kind=EventKind.FAILURE,
                        details={"elapsed_seconds": time.perf_counter() - video_started},
                    )
                else:
                    emit(
                        PipelineStage.VIDEO,
                        f"{video.name} completed",
                        kind=EventKind.COMPLETE,
                        details={"elapsed_seconds": time.perf_counter() - video_started},
                    )
    elapsed = time.perf_counter() - started
    existing = tuple(path for path in plan.artifacts if path.exists())
    return RunSummary(
        len(plan.videos) - len(failures),
        len(plan.videos),
        elapsed,
        tuple(failures),
        existing,
        tuple(errors),
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
        if settings.tts.generation_mode == "chunked" or settings.tts.mode == "timed":
            review = paths.tts_review_path(paths.audio_dir / f"{video.stem}_tts.wav")
            artifacts.extend([review, review.with_suffix(".pretty.json")])
    return artifacts


def _from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

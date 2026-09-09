from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..artifacts import is_partial_artifact
from ..config import VIDEO_EXTENSIONS, ProjectPaths, RunSettings, configure_binary_path
from ..context import log_context
from ..events import EventKind, PipelineObserver, PipelineStage, emit, event_scope, stage_context
from ..processing.media import find_videos
from ..processing.models import get_model_filename_suffix
from ..processing.pipeline import process_video
from ..processing.runtime import model_runtime
from .errors import describe_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessPlan:
    settings: RunSettings
    root: Path
    paths: ProjectPaths
    model: Path
    videos: tuple[Path, ...]
    source_srt_paths: tuple[Path, ...]
    translated_srt_paths: tuple[Path, ...]
    speedup_spec_paths: tuple[Path, ...]
    normal_output_paths: tuple[Path, ...]
    speedup_output_paths: tuple[Path, ...]
    artifacts: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RunSummary:
    succeeded: int
    total: int
    elapsed_seconds: float
    failures: tuple[str, ...]
    artifacts: tuple[Path, ...]
    errors: tuple[Exception, ...] = ()


def build_process_plan(
    settings: RunSettings,
    videos: Sequence[Path] | None = None,
    translated_srt: Path | None = None,
) -> ProcessPlan:
    from .settings import validate_settings

    validate_settings(settings)
    root = Path(settings.project.root).expanduser().resolve()
    paths = ProjectPaths.from_root(root)
    model = _from_root(root, settings.project.model)
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
            if is_partial_artifact(video):
                raise ValueError(f"Incomplete artifact cannot be used as input: {video}")
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
    if not model.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model}")
    suffix = get_model_filename_suffix(model)
    if translated_srt is not None and len(resolved_videos) != 1:
        raise ValueError("--translated-srt can only be used when processing one video.")
    if settings.speedup.enabled and settings.speedup.spec is not None and len(resolved_videos) != 1:
        raise ValueError("--speedup-spec can only be used when processing one video.")
    source_srt_paths = tuple(
        paths.source_subtitle_dir / f"{video.stem}_vi_{suffix}.srt" for video in resolved_videos
    )
    translated_srt_paths = tuple(
        translated_srt.expanduser().resolve()
        if translated_srt is not None
        else paths.translated_subtitle_dir / f"{video.stem}_en.srt"
        for video in resolved_videos
    )
    for source_srt_path, translated_srt_path in zip(
        source_srt_paths, translated_srt_paths, strict=True
    ):
        if source_srt_path.resolve() == translated_srt_path.resolve():
            raise ValueError("Source and translated subtitle paths must be different.")
    speedup_spec_paths = tuple(
        paths.speedup_spec_path(video, settings.speedup.spec) for video in resolved_videos
    )
    normal_output_paths = tuple(
        normal
        for video in resolved_videos
        for normal in paths.normal_video_paths(video, tts_enabled=settings.tts.enabled)
    )
    speedup_output_paths = tuple(
        output
        for video in resolved_videos
        for output in _planned_speedup_outputs(video, paths, settings)
    )
    for variable in ("TRANSCRIPT_VIDEO_FFMPEG", "TRANSCRIPT_VIDEO_FFPROBE"):
        configured = os.environ.get(variable)
        if configured and not Path(configured).expanduser().is_file():
            raise FileNotFoundError(f"{variable} does not point to a file: {configured}")
    artifacts = tuple(
        artifact
        for video, source_srt_path, translated_srt_path in zip(
            resolved_videos, source_srt_paths, translated_srt_paths, strict=True
        )
        for artifact in _artifacts_for(video, source_srt_path, translated_srt_path, paths, settings)
    )
    for artifact in artifacts:
        parent = next(p for p in artifact.parents if p.exists())
        if artifact.is_dir() or not parent.is_dir():
            raise ValueError(f"Invalid output path: {artifact}")
    return ProcessPlan(
        settings=settings,
        root=root,
        paths=paths,
        model=model,
        videos=resolved_videos,
        source_srt_paths=source_srt_paths,
        translated_srt_paths=translated_srt_paths,
        speedup_spec_paths=speedup_spec_paths,
        normal_output_paths=normal_output_paths,
        speedup_output_paths=speedup_output_paths,
        artifacts=artifacts,
    )


def execute_process_plan(plan: ProcessPlan, observer: PipelineObserver | None = None) -> RunSummary:
    failures: list[str] = []
    errors: list[Exception] = []
    started = time.perf_counter()
    with (
        event_scope(observer),
        model_runtime(),
        stage_context(PipelineStage.RUN, "Processing videos", total=len(plan.videos)),
    ):
        plan.paths.create_dirs()
        configure_binary_path(plan.root)
        if not plan.model.is_dir():
            raise FileNotFoundError(f"Model folder not found: {plan.model}")
        for position, (video, source_srt_path, translated_srt_path) in enumerate(
            zip(
                plan.videos,
                plan.source_srt_paths,
                plan.translated_srt_paths,
                strict=True,
            ),
            1,
        ):
            with log_context(
                video=video.name, stage="video", operation=None, chunk=None, subtitle=None
            ):
                stages = ["source subtitles", "translated subtitles"]
                if translated_srt_path.is_file() and not plan.settings.transcription.skip_burn:
                    stages += ["tts", "render", "mux"] if plan.settings.tts.enabled else ["render"]
                if plan.settings.speedup.enabled:
                    stages.append("speedup")
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
                        source_srt_path,
                        translated_srt_path,
                        plan.paths,
                        plan.settings,
                        observer,
                    )
                except Exception as exc:
                    logger.debug(
                        "Video processing failed", exc_info=True, extra={"diagnostic_only": True}
                    )
                    failures.append(f"{video.name}: {describe_error(exc)}")
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
    video: Path,
    source_srt_path: Path,
    translated_srt_path: Path,
    paths: ProjectPaths,
    settings: RunSettings,
) -> list[Path]:
    artifacts = [source_srt_path, translated_srt_path]
    if not settings.transcription.skip_burn:
        artifacts.append(paths.normal_video_path(video, tts_enabled=False))
    if settings.tts.enabled and not settings.transcription.skip_burn:
        artifacts.extend(
            [
                paths.audio_dir / f"{video.stem}_tts.wav",
                paths.normal_video_path(video, tts_enabled=True),
            ]
        )
        if settings.tts.generation_mode == "chunked" or settings.tts.mode == "timed":
            artifacts.append(paths.retimed_subtitle_dir / f"{video.stem}_en_retimed.srt")
        if settings.tts.generation_mode == "chunked" or settings.tts.mode == "timed":
            review = paths.tts_review_path(paths.audio_dir / f"{video.stem}_tts.wav")
            artifacts.extend([review, review.with_suffix(".pretty.json")])
        review = paths.tts_review_path(paths.audio_dir / f"{video.stem}_tts.wav")
        artifacts.append(review.with_suffix(".duration.json"))
    if settings.speedup.enabled:
        artifacts.extend(_planned_speedup_outputs(video, paths, settings))
    return artifacts


def _planned_speedup_outputs(
    video: Path, paths: ProjectPaths, settings: RunSettings
) -> tuple[Path, ...]:
    if not settings.speedup.enabled:
        return ()
    spec = paths.speedup_spec_path(video, settings.speedup.spec)
    if not spec.is_file():
        return ()
    from ..processing.speedup import parse_speedup_spec

    if not parse_speedup_spec(spec):
        return ()
    candidates = paths.normal_video_paths(video, tts_enabled=settings.tts.enabled)
    if settings.transcription.skip_burn:
        candidates = tuple(path for path in candidates if path.is_file())
    return tuple(paths.speedup_output_path(path) for path in candidates)


def _from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

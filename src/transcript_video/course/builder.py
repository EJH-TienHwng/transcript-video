from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from ..context import log_context
from ..events import (
    EventKind,
    PipelineObserver,
    PipelineStage,
    emit,
    event_scope,
    ffmpeg_events,
    stage_context,
)
from ..process_runner import ffmpeg_progress_handler
from .cards import render_session_card, render_toc_pages
from .config import CourseConfig, course_config_document, parse_course_config
from .media import (
    add_chapter_metadata,
    concatenate_videos,
    get_media_duration_seconds,
    normalize_session_video,
    shutil_copy,
    still_image_to_video,
)
from .timeline import (
    SessionTimeline,
    build_timeline,
    format_video_timestamp,
    session_number,
)

logger = logging.getLogger(__name__)


def _prepare_directories(config: CourseConfig) -> dict:
    root = config.work_dir
    directories = {
        "root": root,
        "images": root / "images",
        "cards": root / "cards",
        "toc": root / "toc",
        "normalized": root / "normalized",
        "temp": root / "temp",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def _read_durations(config: CourseConfig) -> list[float]:
    durations = []

    for position, session in enumerate(config.sessions, start=1):
        if not session.video.exists():
            raise FileNotFoundError(f"Session {position} video not found: {session.video}")

        duration = get_media_duration_seconds(session.video)
        durations.append(duration)
        logger.info(
            "Session %02d duration: %s | %s",
            session_number(session, position),
            format_video_timestamp(duration),
            session.title,
        )

    return durations


def _normalize_sessions(
    config: CourseConfig,
    timeline: Sequence[SessionTimeline],
    output_dir: Path,
) -> list[Path]:
    normalized_paths: list[Path] = []

    for position, item in enumerate(timeline, start=1):
        number = session_number(item.session, position)
        output_path = output_dir / f"session_{position:03d}_n{number:03d}.mp4"

        logger.info(
            "Normalizing session %02d/%02d: %s",
            position,
            len(timeline),
            item.session.title,
        )
        emit(PipelineStage.NORMALIZE, item.session.title, current=position - 1, total=len(timeline))
        with (
            log_context(video=item.session.video.name),
            ffmpeg_progress_handler(
                ffmpeg_events(PipelineStage.NORMALIZE, item.session.title, item.duration)
            ),
        ):
            normalize_session_video(
                video_in=item.session.video,
                video_out=output_path,
                config=config,
            )
        normalized_paths.append(output_path)
        emit(PipelineStage.NORMALIZE, item.session.title, current=position, total=len(timeline))

    return normalized_paths


def _build_toc_videos(
    config: CourseConfig,
    timeline: Sequence[SessionTimeline],
    dirs: dict,
) -> list[Path]:
    toc_images = render_toc_pages(config, timeline, dirs["images"])
    toc_videos: list[Path] = []

    for index, image_path in enumerate(toc_images, start=1):
        video_path = dirs["toc"] / f"toc_{index:03d}.mp4"
        logger.info("Rendering TOC page %03d", index)
        with ffmpeg_progress_handler(
            ffmpeg_events(PipelineStage.TOC, f"TOC page {index}", config.toc.page_duration)
        ):
            still_image_to_video(
                image_path=image_path,
                video_out=video_path,
                duration=config.toc.page_duration,
                config=config,
            )
        toc_videos.append(video_path)
        emit(PipelineStage.TOC, "TOC pages rendered", current=index, total=len(toc_images))

    return toc_videos


def _build_session_cards(
    config: CourseConfig,
    timeline: Sequence[SessionTimeline],
    dirs: dict,
) -> list[Path]:
    card_videos: list[Path] = []

    for position, item in enumerate(timeline, start=1):
        number = session_number(item.session, position)
        image_path = dirs["images"] / f"session_{position:03d}_n{number:03d}.png"
        video_path = dirs["cards"] / f"session_{position:03d}_n{number:03d}.mp4"

        logger.info(
            "Rendering session card %02d: %s",
            number,
            item.session.title,
        )
        with ffmpeg_progress_handler(
            ffmpeg_events(PipelineStage.CARDS, item.session.title, config.card_duration)
        ):
            render_session_card(
                config=config,
                timeline_item=item,
                position=position,
                output_path=image_path,
            )
            still_image_to_video(
                image_path=image_path,
                video_out=video_path,
                duration=config.card_duration,
                config=config,
            )
        card_videos.append(video_path)
        emit(PipelineStage.CARDS, item.session.title, current=position, total=len(timeline))

    return card_videos


def _print_timeline(timeline: Sequence[SessionTimeline]) -> None:
    logger.info("Final course timeline:")
    for position, item in enumerate(timeline, start=1):
        number = session_number(item.session, position)
        logger.info(
            "  %02d | %s | %s",
            number,
            format_video_timestamp(item.content_start),
            item.session.title,
        )


def build_course(config: CourseConfig, observer: PipelineObserver | None = None) -> Path:
    """Build the existing media timeline while publishing one shared event stream."""
    config = parse_course_config(course_config_document(config), Path.cwd())
    with (
        event_scope(observer),
        stage_context(
            PipelineStage.RUN,
            f"Building course: {config.title}",
            total=1,
            details={
                "stages": ["validate", "normalize", "toc", "cards", "concatenate", "chapters"]
            },
        ),
    ):
        with stage_context(PipelineStage.VALIDATE, "Validating course sessions"):
            durations = _read_durations(config)
            preliminary_timeline = build_timeline(config, durations)
            dirs = _prepare_directories(config)
        with stage_context(
            PipelineStage.NORMALIZE, "Normalizing sessions", total=len(config.sessions)
        ):
            normalized_sessions = _normalize_sessions(
                config, preliminary_timeline, dirs["normalized"]
            )
            normalized_durations = [
                get_media_duration_seconds(path) for path in normalized_sessions
            ]
            timeline = build_timeline(config, normalized_durations)
            _print_timeline(timeline)
        with stage_context(PipelineStage.TOC, "Rendering table of contents"):
            toc_videos = _build_toc_videos(config, timeline, dirs)
        with stage_context(
            PipelineStage.CARDS, "Rendering session cards", total=len(config.sessions)
        ):
            card_videos = _build_session_cards(config, timeline, dirs)
        concat_segments = list(toc_videos)
        for card, session in zip(card_videos, normalized_sessions, strict=True):
            concat_segments.extend([card, session])
        compiled_without_chapters = dirs["temp"] / "compiled_without_chapters.mp4"
        duration = timeline[-1].content_end
        with (
            stage_context(PipelineStage.CONCAT, "Concatenating course video"),
            ffmpeg_progress_handler(
                ffmpeg_events(PipelineStage.CONCAT, "Concatenating course video", duration)
            ),
        ):
            concatenate_videos(
                video_paths=concat_segments,
                output_path=compiled_without_chapters,
                list_path=dirs["temp"] / "concat.txt",
            )
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with (
            stage_context(
                PipelineStage.CHAPTERS,
                "Writing chapters" if config.add_chapters else "Publishing course",
            ),
            ffmpeg_progress_handler(
                ffmpeg_events(PipelineStage.CHAPTERS, "Writing chapters", duration)
            ),
        ):
            if config.add_chapters:
                add_chapter_metadata(
                    video_in=compiled_without_chapters,
                    video_out=config.output,
                    chapter_file=dirs["temp"] / "chapters.ffmeta",
                    chapter_starts=[item.content_start for item in timeline],
                    chapter_titles=[
                        f"{session_number(item.session, position):02d} - {item.session.title}"
                        for position, item in enumerate(timeline, 1)
                    ],
                )
            else:
                shutil_copy(compiled_without_chapters, config.output)
        emit(
            PipelineStage.COMPLETE,
            "Course built",
            kind=EventKind.ARTIFACT,
            artifact=config.output,
            details={
                "category": "Video",
                "sessions": len(config.sessions),
                "duration": duration,
                "chapters": len(config.sessions) if config.add_chapters else 0,
            },
        )
        return config.output

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Sequence

from .cards import render_session_card, render_toc_pages
from .config import CourseConfig
from .media import (
    add_chapter_metadata,
    concatenate_videos,
    get_media_duration_seconds,
    normalize_session_video,
    still_image_to_video,
)
from .timeline import (
    SessionTimeline,
    build_timeline,
    format_video_timestamp,
    session_number,
)


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


def _read_durations(config: CourseConfig) -> List[float]:
    durations = []

    for position, session in enumerate(config.sessions, start=1):
        if not session.video.exists():
            raise FileNotFoundError(
                f"Session {position} không tìm thấy video: {session.video}"
            )

        duration = get_media_duration_seconds(session.video)
        durations.append(duration)
        logging.info(
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
) -> List[Path]:
    normalized_paths: List[Path] = []

    for position, item in enumerate(timeline, start=1):
        number = session_number(item.session, position)
        output_path = output_dir / f"session_{position:03d}_n{number:03d}.mp4"

        logging.info(
            "Normalizing session %02d/%02d: %s",
            position,
            len(timeline),
            item.session.title,
        )
        normalize_session_video(
            video_in=item.session.video,
            video_out=output_path,
            config=config,
        )
        normalized_paths.append(output_path)

    return normalized_paths


def _build_toc_videos(
    config: CourseConfig,
    timeline: Sequence[SessionTimeline],
    dirs: dict,
) -> List[Path]:
    toc_images = render_toc_pages(config, timeline, dirs["images"])
    toc_videos: List[Path] = []

    for index, image_path in enumerate(toc_images, start=1):
        video_path = dirs["toc"] / f"toc_{index:03d}.mp4"
        logging.info("Rendering TOC page %03d", index)
        still_image_to_video(
            image_path=image_path,
            video_out=video_path,
            duration=config.toc.page_duration,
            config=config,
        )
        toc_videos.append(video_path)

    return toc_videos


def _build_session_cards(
    config: CourseConfig,
    timeline: Sequence[SessionTimeline],
    dirs: dict,
) -> List[Path]:
    card_videos: List[Path] = []

    for position, item in enumerate(timeline, start=1):
        number = session_number(item.session, position)
        image_path = dirs["images"] / f"session_{position:03d}_n{number:03d}.png"
        video_path = dirs["cards"] / f"session_{position:03d}_n{number:03d}.mp4"

        logging.info(
            "Rendering session card %02d: %s",
            number,
            item.session.title,
        )
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

    return card_videos


def _print_timeline(timeline: Sequence[SessionTimeline]) -> None:
    logging.info("Final course timeline:")
    for position, item in enumerate(timeline, start=1):
        number = session_number(item.session, position)
        logging.info(
            "  %02d | %s | %s",
            number,
            format_video_timestamp(item.content_start),
            item.session.title,
        )


def build_course(config: CourseConfig) -> Path:
    """Build TOC + title cards + normalized sessions into one final MP4."""
    dirs = _prepare_directories(config)

    logging.info("Course: %s", config.title)
    logging.info("Sessions: %d", len(config.sessions))

    # 1. Read original session durations.
    durations = _read_durations(config)

    # 2. Calculate the absolute final-video timeline before rendering the TOC.
    timeline = build_timeline(config, durations)
    _print_timeline(timeline)

    # 3. Render visual intro material using the calculated timestamps.
    toc_videos = _build_toc_videos(config, timeline, dirs)
    card_videos = _build_session_cards(config, timeline, dirs)

    # 4. Normalize user videos to one common codec/resolution/fps/audio format.
    normalized_sessions = _normalize_sessions(config, timeline, dirs["normalized"])

    # 5. Interleave TOC + [card, session] pairs in the chosen JSON order.
    concat_segments: List[Path] = list(toc_videos)
    for card, session in zip(card_videos, normalized_sessions):
        concat_segments.extend([card, session])

    compiled_without_chapters = dirs["temp"] / "compiled_without_chapters.mp4"
    concat_list_path = dirs["temp"] / "concat.txt"

    logging.info("Concatenating %d segment(s)...", len(concat_segments))
    concatenate_videos(
        video_paths=concat_segments,
        output_path=compiled_without_chapters,
        list_path=concat_list_path,
    )

    # 6. Add MP4 chapter metadata at each real session-content start.
    config.output.parent.mkdir(parents=True, exist_ok=True)

    if config.add_chapters:
        chapter_starts = [item.content_start for item in timeline]
        chapter_titles = [
            f"{session_number(item.session, position):02d} - {item.session.title}"
            for position, item in enumerate(timeline, start=1)
        ]
        chapter_file = dirs["temp"] / "chapters.ffmeta"

        logging.info("Adding %d chapter(s)...", len(chapter_titles))
        add_chapter_metadata(
            video_in=compiled_without_chapters,
            video_out=config.output,
            chapter_file=chapter_file,
            chapter_starts=chapter_starts,
            chapter_titles=chapter_titles,
        )
    else:
        shutil.copy2(compiled_without_chapters, config.output)

    logging.info("DONE: %s", config.output)
    return config.output

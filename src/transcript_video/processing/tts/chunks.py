from __future__ import annotations

import json
import logging
from math import ceil
from pathlib import Path

from ...config import ProjectPaths, SubtitleSegment, find_project_root
from ...context import log_context
from ...events import PipelineStage, emit, stage_context
from ..media import get_media_duration_seconds
from .core import (
    TTSContextGroup,
    build_tts_context_groups,
    generate_context_group_items,
    invalidate_tts_review_log,
    load_faster_whisper_aligner,
    load_qwen_tts_model,
    log_tts_summary,
    overlay_tts_items,
    tts_review_is_current,
    write_tts_review_log,
)

logger = logging.getLogger(__name__)


def build_fixed_time_tts_chunks(
    segments: list[SubtitleSegment],
    chunk_minutes: int,
    video_path: Path,
) -> list[tuple[int, float, float, list[SubtitleSegment]]]:
    """Build fixed-length cache/review units over the video timeline."""
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")
    valid_segments = [segment for segment in segments if segment.text.strip()]
    if not valid_segments:
        raise ValueError("No subtitle segments are available for TTS generation.")

    chunk_seconds = chunk_minutes * 60
    video_duration = get_media_duration_seconds(video_path) or 0.0
    total_duration = max(video_duration, max(segment.end for segment in valid_segments) + 1.0)
    chunks = []
    for chunk_index in range(max(1, ceil(total_duration / chunk_seconds))):
        chunk_start = chunk_index * chunk_seconds
        chunk_end = min((chunk_index + 1) * chunk_seconds, total_duration)
        chunks.append(
            (
                chunk_index,
                chunk_start,
                chunk_end,
                [segment for segment in valid_segments if chunk_start <= segment.start < chunk_end],
            )
        )
    return chunks


def synthesize_one_fixed_time_chunk(
    model,
    chunk_index: int,
    chunk_start: float,
    chunk_end: float,
    chunk_segments: list[SubtitleSegment],
    chunk_audio_out: Path,
    tts_language: str,
    tts_speaker: str,
    tts_instruct: str,
    sample_rate: int | None = None,
    max_speedup: float = 1.15,
    all_segments: list[SubtitleSegment] | None = None,
    video_duration: float | None = None,
    chunk_tail_seconds: float = 10.0,
    context_groups: list[TTSContextGroup] | None = None,
    aligner=None,
    review_path: Path | None = None,
) -> int:
    """Generate context groups owned by one fixed cache chunk and overlay their sentences."""
    import numpy as np
    import soundfile as sf

    chunk_audio_out.parent.mkdir(parents=True, exist_ok=True)
    source_segments = all_segments if all_segments is not None else chunk_segments
    available_groups = context_groups or build_tts_context_groups(source_segments)
    groups = [
        group for group in available_groups if chunk_start <= group.segments[0][1].start < chunk_end
    ]
    items, generated_rate, reviews = generate_context_group_items(
        model=model,
        aligner=aligner,
        groups=groups,
        all_segments=source_segments,
        language=tts_language,
        speaker=tts_speaker,
        instruct=tts_instruct,
        max_speedup=max_speedup,
        video_duration=video_duration,
    )
    if sample_rate is not None and generated_rate is not None and sample_rate != generated_rate:
        raise ValueError(f"Inconsistent sample rate: {generated_rate} != {sample_rate}")
    final_rate = generated_rate or sample_rate or 24000
    review_path = review_path or ProjectPaths.from_root(find_project_root()).tts_review_path(
        chunk_audio_out, chunk=True
    )
    if any(entry["action"] == "generation_failed" for entry in reviews):
        write_tts_review_log(review_path, reviews)
        log_tts_summary(len(reviews), reviews, review_path)
        raise ValueError(
            f"TTS chunk {chunk_index}: individual generation failed; see {review_path}"
        )
    minimum_duration = max(0.1, chunk_end - chunk_start + (chunk_tail_seconds if items else 0.0))
    audio = overlay_tts_items(
        items, final_rate, minimum_duration, offset=chunk_start, reviews=reviews
    )
    # Invalidate before replacing the waveform so interrupted writes cannot reuse old ranges.
    invalidate_tts_review_log(review_path)
    sf.write(str(chunk_audio_out), np.asarray(audio, dtype=np.float32), final_rate)
    write_tts_review_log(review_path, reviews)
    logger.info(
        "Wrote TTS chunk %03d with %d context group(s): %s",
        chunk_index,
        len(groups),
        chunk_audio_out,
    )
    return final_rate


def rebuild_full_tts_audio_from_chunks(
    chunk_infos: list[tuple[Path, float, float]],
    audio_out: Path,
    expected_total_duration: float | None = None,
    review_paths: list[Path] | None = None,
) -> list[dict[str, object]]:
    """Recover complete cached sentences and place them on one collision-free timeline."""
    import soundfile as sf

    if not chunk_infos:
        raise ValueError("No audio chunks are available to rebuild the full track.")
    if review_paths is None:
        paths = ProjectPaths.from_root(find_project_root())
        review_paths = [paths.tts_review_path(info[0], chunk=True) for info in chunk_infos]
    items = []
    reviews = []
    seen: set[int] = set()
    final_rate = None
    for (chunk_path, _chunk_start, _), review_path in zip(chunk_infos, review_paths, strict=True):
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing audio chunk: {chunk_path}")
        wav, sr = sf.read(str(chunk_path), dtype="float32")
        if final_rate is not None and sr != final_rate:
            raise ValueError(f"Inconsistent chunk sample rate: {sr} != {final_rate}")
        final_rate = sr
        chunk_reviews = _read_reviews([review_path])
        expected = [
            (entry["subtitle_index"], SubtitleSegment(entry["start"], entry["end"], entry["text"]))
            for entry in chunk_reviews
        ]
        if not tts_review_is_current(review_path, expected):
            raise ValueError(
                f"Unsafe or outdated TTS cache: {chunk_path}; regenerate with --force tts"
            )
        if not chunk_reviews and wav.any():
            raise ValueError(f"Missing sentence ranges in TTS cache: {chunk_path}")
        for entry in chunk_reviews:
            index = entry["subtitle_index"]
            first, last = entry["cache_start_sample"], entry["cache_end_sample"]
            if index in seen or not 0 <= first < last <= len(wav):
                raise ValueError(
                    f"Duplicate subtitle or invalid cached audio range: {chunk_path}, #{index}"
                )
            seen.add(index)
            segment = SubtitleSegment(entry["start"], entry["end"], entry["text"])
            items.append((index, segment, wav[first:last].copy()))
        reviews.extend(chunk_reviews)
    if final_rate is None:
        raise ValueError("Could not read a sample rate from the audio chunks.")

    audio = overlay_tts_items(items, final_rate, expected_total_duration or 0.0, reviews=reviews)
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_out), audio, final_rate)
    logger.info("Rebuilt full TTS audio from cached sentences: %s", audio_out)
    return sorted(reviews, key=lambda entry: (entry["start"], entry["subtitle_index"]))


def _dependent_owner_chunks(
    groups: list[TTSContextGroup], chunk_seconds: float, target_chunk: int
) -> set[int]:
    owners = {target_chunk}
    for group in groups:
        member_chunks = {int(segment.start // chunk_seconds) for _, segment in group.segments}
        if target_chunk in member_chunks:
            owners.add(int(group.segments[0][1].start // chunk_seconds))
    return owners


def _read_reviews(paths: list[Path]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def synthesize_tts_audio_by_time_chunks(
    segments: list[SubtitleSegment],
    audio_out: Path,
    chunks_dir: Path,
    video_path: Path,
    tts_model_name: str,
    tts_language: str,
    tts_speaker: str,
    tts_instruct: str,
    device: str,
    attn_implementation: str,
    chunk_minutes: int = 5,
    rerun_chunk: int | None = None,
    overwrite_all_chunks: bool = False,
    max_speedup: float = 1.15,
    chunk_tail_seconds: float = 10.0,
    alignment_model_name: str | Path | None = None,
    context_max_sentences: int = 4,
    context_max_chars: int = 450,
    context_break_seconds: float = 3.0,
    review_log_path: Path | None = None,
) -> None:
    """Generate contextual TTS while retaining fixed multi-minute cache units."""
    if rerun_chunk is not None and rerun_chunk < 0:
        raise ValueError("rerun_chunk must be zero or greater.")
    if chunk_tail_seconds < 0 or max_speedup < 1.0:
        raise ValueError("TTS timing settings are outside their supported ranges.")
    valid_segments = sorted(
        [segment for segment in segments if segment.text.strip()], key=lambda item: item.start
    )
    if not valid_segments:
        raise ValueError("No subtitle segments are available for TTS generation.")

    groups = build_tts_context_groups(
        segments, context_max_sentences, context_max_chars, context_break_seconds
    )
    chunks = build_fixed_time_tts_chunks(valid_segments, chunk_minutes, video_path)
    if rerun_chunk is not None and rerun_chunk >= len(chunks):
        raise ValueError(
            f"Chunk {rerun_chunk} does not exist; valid indexes are 0 through {len(chunks) - 1}."
        )
    chunks_dir.mkdir(parents=True, exist_ok=True)
    video_duration = get_media_duration_seconds(video_path) or 0.0
    expected_duration = max(video_duration, max(segment.end for segment in valid_segments) + 1.0)
    rerun_owners = (
        _dependent_owner_chunks(groups, chunk_minutes * 60, rerun_chunk)
        if rerun_chunk is not None
        else set()
    )

    infos = []
    sentences_by_chunk = {}
    final_review = review_log_path or ProjectPaths.from_root(find_project_root()).tts_review_path(
        audio_out
    )
    chunk_report_dir = final_review.parent / chunks_dir.name
    for chunk_index, chunk_start, chunk_end, chunk_segments in chunks:
        chunk_path = chunks_dir / f"{audio_out.stem}_chunk_{chunk_index:03d}.wav"
        review_path = chunk_report_dir / f"{chunk_path.stem}.review.jsonl"
        owned_segments = [
            item
            for group in groups
            if chunk_start <= group.segments[0][1].start < chunk_end
            for item in group.segments
        ]
        sentences_by_chunk[chunk_index] = len(owned_segments)
        should_generate = (
            overwrite_all_chunks
            or chunk_index in rerun_owners
            or not chunk_path.exists()
            or not tts_review_is_current(review_path, owned_segments)
        )
        if (
            should_generate
            and chunk_path.exists()
            and not tts_review_is_current(review_path, owned_segments)
        ):
            logger.info(
                "TTS chunk %03d: missing or outdated report metadata at %s; regenerating safely",
                chunk_index,
                review_path,
            )
        infos.append(
            (
                chunk_index,
                chunk_start,
                chunk_end,
                chunk_segments,
                chunk_path,
                review_path,
                should_generate,
            )
        )

    to_generate = [info[0] for info in infos if info[-1]]
    completed_chunks = len(infos) - len(to_generate)
    completed_sentences = sum(sentences_by_chunk[info[0]] for info in infos if not info[-1])
    if to_generate:
        logger.info("Chunks to generate/regenerate: %s", to_generate)
        model = load_qwen_tts_model(tts_model_name, device, attn_implementation)
        aligner = (
            load_faster_whisper_aligner(alignment_model_name) if alignment_model_name else None
        )
        sample_rate = None
        for (
            chunk_index,
            chunk_start,
            chunk_end,
            chunk_segments,
            chunk_path,
            chunk_review,
            should_generate,
        ) in infos:
            if not should_generate:
                continue
            with (
                log_context(chunk=chunk_index),
                stage_context(
                    PipelineStage.TTS,
                    f"Generating TTS chunk {chunk_index + 1}/{len(infos)}",
                    operation="chunk",
                ),
            ):
                sample_rate = synthesize_one_fixed_time_chunk(
                    model=model,
                    aligner=aligner,
                    chunk_index=chunk_index,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    chunk_segments=chunk_segments,
                    chunk_audio_out=chunk_path,
                    tts_language=tts_language,
                    tts_speaker=tts_speaker,
                    tts_instruct=tts_instruct,
                    sample_rate=sample_rate,
                    max_speedup=max_speedup,
                    all_segments=segments,
                    video_duration=video_duration,
                    chunk_tail_seconds=chunk_tail_seconds,
                    context_groups=groups,
                    review_path=chunk_review,
                )
            with log_context(operation="chunks", chunk=chunk_index):
                completed_chunks += 1
                completed_sentences += sentences_by_chunk[chunk_index]
                emit(
                    PipelineStage.TTS,
                    "TTS chunks ready",
                    current=completed_chunks,
                    total=len(infos),
                    details={
                        "unit": "chunks",
                        "sentences": completed_sentences,
                        "total_sentences": len(valid_segments),
                    },
                )

    else:
        logger.info("All TTS chunks and review metadata exist; rebuilding without model loading.")

    with log_context(operation="chunks", chunk=None):
        emit(
            PipelineStage.TTS,
            "TTS chunks ready",
            current=len(infos),
            total=len(infos),
            details={
                "unit": "chunks",
                "sentences": len(valid_segments),
                "total_sentences": len(valid_segments),
            },
        )
    invalidate_tts_review_log(final_review)
    reviews = rebuild_full_tts_audio_from_chunks(
        [(info[4], info[1], info[2]) for info in infos],
        audio_out,
        expected_duration,
        review_paths=[info[5] for info in infos],
    )
    write_tts_review_log(final_review, reviews)
    log_tts_summary(len(valid_segments), reviews, final_review)

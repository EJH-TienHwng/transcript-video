from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ..artifacts import write_text
from ..config import ProjectPaths, RunSettings
from ..context import log_context
from ..events import (
    EventKind,
    PipelineObserver,
    PipelineStage,
    emit,
    event_scope,
    ffmpeg_events,
    stage_context,
    warn,
)
from ..process_runner import ffmpeg_progress_handler
from .media import (
    burn_subtitles,
    get_media_duration_seconds,
    mux_audio_into_video_mix,
    mux_audio_into_video_replace,
    split_audio_into_chunks,
)
from .models import get_model_filename_suffix
from .provenance import (
    cache_matches,
    manifest_path,
    model_identity,
    subtitle_provenance,
    tts_provenance,
    write_provenance,
)
from .subtitles import post_process_segments, read_srt, write_srt
from .transcription import transcribe_video, translate_segments_with_vinai
from .tts import (
    synthesize_simple_tts_audio,
    synthesize_timed_tts_audio,
    synthesize_tts_audio_by_time_chunks,
)
from .tts.chunks import _read_reviews
from .tts.core import log_tts_summary, tts_review_is_current


def process_video(
    video_path: Path,
    model_path: Path,
    translation_model_path: Path | None,
    paths: ProjectPaths,
    settings: RunSettings,
    observer: PipelineObserver | None = None,
) -> None:
    """Generate subtitles, optionally generate TTS, and render the output video."""
    with event_scope(observer, video=video_path.name):
        return _process_video(video_path, model_path, translation_model_path, paths, settings)


def _process_video(
    video_path: Path,
    model_path: Path,
    translation_model_path: Path | None,
    paths: ProjectPaths,
    settings: RunSettings,
) -> None:
    transcription = settings.transcription
    hardware = settings.hardware
    tts = settings.tts
    model_suffix = get_model_filename_suffix(model_path, translation_model_path)
    srt_path = paths.subtitle_dir / f"{video_path.stem}_{model_suffix}.srt"

    subtitled_output_path = paths.output_dir / f"{video_path.stem}_vi-dub_en-sub.mp4"
    tts_audio_path = paths.audio_dir / f"{video_path.stem}_tts.wav"
    tts_chunks_dir = paths.audio_dir / f"{video_path.stem}_tts_chunks"
    tts_review_path = paths.tts_review_path(tts_audio_path)
    final_tts_output_path = paths.output_dir / f"{video_path.stem}_en-dub_en-sub.mp4"

    logger = logging.getLogger(__name__)

    subtitle_fingerprint = subtitle_provenance(
        video_path, model_path, translation_model_path, settings
    )
    subtitles_reused = not transcription.overwrite_srt and cache_matches(
        srt_path, subtitle_fingerprint, PipelineStage.SUBTITLES
    )
    if subtitles_reused:
        with stage_context(PipelineStage.SUBTITLES, "Reusing cached subtitles", reused=True):
            segments = read_srt(srt_path)
            if not segments:
                raise ValueError(f"SRT contains no valid subtitles: {srt_path}")
    else:
        # A separate text translation model needs source-language transcription first.
        transcription_task = (
            "transcribe" if translation_model_path is not None else transcription.task
        )
        with stage_context(PipelineStage.TRANSCRIBE, "Transcribing audio"):
            segments = transcribe_video(
                video_path=video_path,
                model_path=model_path,
                paths=paths,
                task=transcription_task,
                language=transcription.language.strip() or None,
                device=hardware.device,
                compute_type=hardware.compute_type,
            )

        if translation_model_path is not None:
            with stage_context(PipelineStage.TRANSLATE, "Translating subtitles"):
                segments = translate_segments_with_vinai(
                    segments=segments,
                    model_path=translation_model_path,
                    device=hardware.device,
                    batch_size=transcription.translation_batch_size,
                )

        segments = post_process_segments(segments)
        if not segments:
            raise ValueError(f"No valid subtitles were generated for: {video_path.name}")

        with stage_context(PipelineStage.SUBTITLES, "Writing subtitles"):
            manifest_path(srt_path).unlink(missing_ok=True)
            write_srt(segments, srt_path)
            write_provenance(srt_path, subtitle_fingerprint)
            # Use the exact published millisecond timestamps on every run.
            segments = read_srt(srt_path)
    emit(
        PipelineStage.SUBTITLES,
        "Subtitles ready",
        kind=EventKind.ARTIFACT,
        artifact=srt_path,
        details={"category": "Subtitles"},
    )

    if transcription.skip_burn:
        return

    duration = get_media_duration_seconds(video_path)
    with (
        stage_context(PipelineStage.RENDER, "Rendering subtitles"),
        ffmpeg_progress_handler(
            ffmpeg_events(PipelineStage.RENDER, "Rendering subtitles", duration)
        ),
    ):
        burn_subtitles(
            video_path,
            srt_path,
            subtitled_output_path,
            video_encoder=hardware.video_encoder,
            subtitle_style=settings.subtitle_style,
        )

    emit(
        PipelineStage.RENDER,
        "Subtitled video ready",
        kind=EventKind.ARTIFACT,
        artifact=subtitled_output_path,
        details={"category": "Video"},
    )
    if not tts.enabled:
        return
    tts_model_path = Path(tts.model).expanduser()
    tts_model_name = (
        str((paths.root / tts_model_path).resolve())
        if not tts_model_path.is_absolute()
        else str(tts_model_path)
    )

    tts_fingerprint = tts_provenance(
        segments,
        {
            **asdict(tts),
            "model": model_identity(tts_model_name),
            "alignment_model": model_identity(model_path),
            "input": subtitle_fingerprint["input"],
            "device": hardware.device,
        },
    )

    with stage_context(PipelineStage.TTS, "Generating voice-over"):
        if tts.generation_mode == "chunked":
            logger.info("Generating/rebuilding chunked Qwen TTS audio: %s", tts_audio_path)
            synthesize_tts_audio_by_time_chunks(
                segments=segments,
                audio_out=tts_audio_path,
                chunks_dir=tts_chunks_dir,
                video_path=video_path,
                tts_model_name=tts_model_name,
                tts_language=tts.language,
                tts_speaker=tts.speaker,
                tts_instruct=tts.instruct,
                device=hardware.device,
                attn_implementation=tts.attn_implementation,
                chunk_minutes=tts.chunk_minutes,
                rerun_chunk=tts.rerun_chunk,
                overwrite_all_chunks=tts.overwrite or not subtitles_reused,
                max_speedup=tts.max_speedup,
                chunk_tail_seconds=tts.chunk_tail_seconds,
                alignment_model_name=model_path,
                context_max_sentences=tts.context_max_sentences,
                context_max_chars=tts.context_max_chars,
                context_break_seconds=tts.context_break_seconds,
                review_log_path=tts_review_path,
                verify_final_audio=tts.verify_final_audio,
            )
        else:
            can_reuse_tts = (
                tts_audio_path.exists()
                and cache_matches(tts_audio_path, tts_fingerprint, PipelineStage.TTS)
                and subtitles_reused
                and not tts.overwrite
                and tts.rerun_chunk is None
                and (
                    tts.mode == "simple"
                    or tts_review_is_current(
                        tts_review_path,
                        [
                            (index, segment)
                            for index, segment in enumerate(segments, 1)
                            if segment.text.strip()
                        ],
                    )
                )
            )
            if can_reuse_tts:
                logger.info("Reusing existing TTS audio: %s", tts_audio_path)
                emit(
                    PipelineStage.TTS,
                    "Reusing existing TTS audio",
                    kind=EventKind.REUSED,
                    artifact=tts_audio_path,
                )
                if tts.mode == "timed":
                    log_tts_summary(
                        len(segments), _read_reviews([tts_review_path]), tts_review_path
                    )
            else:
                if tts.rerun_chunk is not None:
                    warn(
                        logger,
                        "tts.rerun_chunk only applies to chunked generation and will be ignored.",
                    )
                if tts.mode == "simple":
                    synthesize_simple_tts_audio(
                        segments=segments,
                        audio_out=tts_audio_path,
                        tts_model_name=tts_model_name,
                        tts_language=tts.language,
                        tts_speaker=tts.speaker,
                        tts_instruct=tts.instruct,
                        device=hardware.device,
                        attn_implementation=tts.attn_implementation,
                    )
                else:
                    synthesize_timed_tts_audio(
                        segments=segments,
                        audio_out=tts_audio_path,
                        video_path=video_path,
                        tts_model_name=tts_model_name,
                        tts_language=tts.language,
                        tts_speaker=tts.speaker,
                        tts_instruct=tts.instruct,
                        device=hardware.device,
                        attn_implementation=tts.attn_implementation,
                        alignment_model_name=model_path,
                        max_speedup=tts.max_speedup,
                        context_max_sentences=tts.context_max_sentences,
                        context_max_chars=tts.context_max_chars,
                        context_break_seconds=tts.context_break_seconds,
                        review_log_path=tts_review_path,
                        verify_final_audio=tts.verify_final_audio,
                    )

                write_provenance(tts_audio_path, tts_fingerprint)

            if tts.split_audio:
                logger.info(
                    "Splitting TTS audio into %d-minute review chunks: %s",
                    tts.chunk_minutes,
                    tts_chunks_dir,
                )
                with ffmpeg_progress_handler(
                    ffmpeg_events(
                        PipelineStage.TTS,
                        "Splitting review audio",
                        get_media_duration_seconds(tts_audio_path),
                    )
                ):
                    split_audio_into_chunks(
                        audio_in=tts_audio_path,
                        output_dir=tts_chunks_dir,
                        chunk_minutes=tts.chunk_minutes,
                        overwrite=True,
                    )

    emit(
        PipelineStage.TTS,
        "Voice-over audio ready",
        kind=EventKind.ARTIFACT,
        artifact=tts_audio_path,
        details={"category": "Audio"},
    )
    if tts_review_path.exists() and (tts.generation_mode == "chunked" or tts.mode == "timed"):
        emit(
            PipelineStage.TTS,
            "TTS review",
            kind=EventKind.ARTIFACT,
            artifact=tts_review_path.with_suffix(".pretty.json"),
            details={"category": "Review"},
        )
    from .tts.qa import duration_qa

    audio_duration = get_media_duration_seconds(tts_audio_path)
    report_path = tts_review_path.with_suffix(".duration.json")
    write_text(
        report_path, json.dumps(duration_qa(duration, audio_duration, None), indent=2) + "\n"
    )
    mux_duration = max(duration or 0, audio_duration or 0) or None
    with (
        stage_context(PipelineStage.MUX, "Muxing voice-over"),
        ffmpeg_progress_handler(
            ffmpeg_events(PipelineStage.MUX, "Muxing voice-over", mux_duration)
        ),
    ):
        if tts.audio_mode == "mix":
            mux_audio_into_video_mix(subtitled_output_path, tts_audio_path, final_tts_output_path)
        else:
            mux_audio_into_video_replace(
                subtitled_output_path, tts_audio_path, final_tts_output_path
            )

    emit(
        PipelineStage.MUX,
        "Voice-over video ready",
        kind=EventKind.ARTIFACT,
        artifact=final_tts_output_path,
        details={"category": "Video"},
    )
    report = duration_qa(
        duration,
        audio_duration,
        get_media_duration_seconds(final_tts_output_path),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(report_path, json.dumps(report, indent=2) + "\n")
    emit(
        PipelineStage.MUX,
        "Duration QA report",
        kind=EventKind.ARTIFACT,
        details={"category": "Review"},
        artifact=report_path,
    )
    with log_context(operation="duration_qa"):
        emit(
            PipelineStage.MUX,
            "Duration QA",
            kind=EventKind.REVIEW,
            details=report,
            artifact=report_path,
        )

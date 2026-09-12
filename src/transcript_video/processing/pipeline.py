from __future__ import annotations

import json
import logging
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
from .speedup import process_speedup_outputs
from .subtitles import post_process_segments, read_srt, write_srt
from .transcription import transcribe_video
from .tts import (
    synthesize_simple_tts_audio,
    synthesize_timed_tts_audio,
    synthesize_tts_audio_by_time_chunks,
)
from .tts.core import tts_review_is_current


def process_video(
    video_path: Path,
    model_path: Path,
    source_srt_path: Path,
    translated_srt_path: Path,
    paths: ProjectPaths,
    settings: RunSettings,
    observer: PipelineObserver | None = None,
) -> None:
    """Generate subtitles, optionally generate TTS, and render the output video."""
    with event_scope(observer, video=video_path.name):
        return _process_video(
            video_path,
            model_path,
            source_srt_path,
            translated_srt_path,
            paths,
            settings,
        )


def _process_video(
    video_path: Path,
    model_path: Path,
    source_srt_path: Path,
    translated_srt_path: Path,
    paths: ProjectPaths,
    settings: RunSettings,
) -> None:
    transcription = settings.transcription
    hardware = settings.hardware
    tts = settings.tts
    if source_srt_path.resolve().parent != paths.source_subtitle_dir.resolve():
        raise ValueError("Vietnamese source subtitles must be written under data/subtitles/source.")
    if source_srt_path.resolve() == translated_srt_path.resolve():
        raise ValueError("Source and translated subtitle paths must be different.")
    subtitled_output_path = paths.normal_video_path(video_path, tts_enabled=False)
    tts_audio_path = paths.audio_dir / f"{video_path.stem}_tts.wav"
    tts_chunks_dir = paths.audio_dir / f"{video_path.stem}_tts_chunks"
    tts_review_path = paths.tts_review_path(tts_audio_path)
    retimed_subtitle_path = paths.retimed_subtitle_dir / f"{video_path.stem}_en_retimed.srt"
    final_tts_output_path = paths.normal_video_path(video_path, tts_enabled=True)
    speedup_spec_path = paths.speedup_spec_path(video_path, settings.speedup.spec)

    logger = logging.getLogger(__name__)

    if source_srt_path.is_file() and not transcription.overwrite_srt:
        with stage_context(
            PipelineStage.SUBTITLES,
            "Using existing Vietnamese source subtitles",
            reused=True,
        ):
            source_segments = read_srt(source_srt_path)
            if not source_segments:
                raise ValueError(f"SRT contains no valid subtitles: {source_srt_path}")
    else:
        with stage_context(PipelineStage.TRANSCRIBE, "Generating Vietnamese source subtitles"):
            source_segments = transcribe_video(
                video_path=video_path,
                model_path=model_path,
                paths=paths,
                language=transcription.language.strip() or None,
                device=hardware.device,
                compute_type=hardware.compute_type,
            )

        source_segments = post_process_segments(source_segments)
        if not source_segments:
            raise ValueError(f"No valid subtitles were generated for: {video_path.name}")

        with stage_context(PipelineStage.SUBTITLES, "Writing Vietnamese source subtitles"):
            write_srt(source_segments, source_srt_path)
            # Use the exact published millisecond timestamps on every run.
            source_segments = read_srt(source_srt_path)
    emit(
        PipelineStage.SUBTITLES,
        "Vietnamese source subtitles ready",
        kind=EventKind.ARTIFACT,
        artifact=source_srt_path,
        details={"category": "Source subtitles"},
    )

    if not translated_srt_path.is_file():
        emit(
            PipelineStage.TRANSLATE,
            "Waiting for translated English subtitles\n\n"
            f"Source SRT:\n{source_srt_path}\n\n"
            "Create the English translated subtitle using:\n"
            f"{paths.root / 'docs/prompts/optimal_prompt.md'}\n\n"
            f"Expected English SRT:\n{translated_srt_path}\n\n"
            "After creating the English SRT, rerun the process command.",
            kind=EventKind.WARNING,
            artifact=translated_srt_path,
            details={
                "category": "Translated subtitles",
                "status": "translation_handoff",
                "source_srt": str(source_srt_path),
                "prompt": str(paths.root / "docs/prompts/optimal_prompt.md"),
            },
        )
        return

    translated_segments = read_srt(translated_srt_path)
    if not translated_segments:
        raise ValueError(f"SRT contains no valid subtitles: {translated_srt_path}")
    emit(
        PipelineStage.TRANSLATE,
        "Using translated English subtitles",
        kind=EventKind.REUSED,
        artifact=translated_srt_path,
        details={"category": "Translated subtitles"},
    )

    if transcription.skip_burn:
        _process_speedup_outputs_if_enabled(
            paths.normal_video_paths(video_path, tts_enabled=tts.enabled),
            speedup_spec_path,
            video_path.stem,
            settings,
        )
        return

    duration = get_media_duration_seconds(video_path)
    if not tts.enabled:
        with (
            stage_context(PipelineStage.RENDER, "Burning English subtitles"),
            ffmpeg_progress_handler(
                ffmpeg_events(PipelineStage.RENDER, "Burning English subtitles", duration)
            ),
        ):
            burn_subtitles(
                video_path,
                translated_srt_path,
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
        _process_speedup_outputs_if_enabled(
            (subtitled_output_path,), speedup_spec_path, video_path.stem, settings
        )
        return
    tts_model_path = Path(tts.model).expanduser()
    tts_model_name = (
        str((paths.root / tts_model_path).resolve())
        if not tts_model_path.is_absolute()
        else str(tts_model_path)
    )

    retimed_segments = None
    cached_audio = (
        tts.generation_mode == "full"
        and not tts.regenerate
        and tts_audio_path.is_file()
        and bool(get_media_duration_seconds(tts_audio_path))
        and tts_audio_path.stat().st_mtime_ns >= translated_srt_path.stat().st_mtime_ns
    )
    reuse_full_tts = bool(cached_audio)
    cached_retimed_segments = None
    if reuse_full_tts and tts.mode == "timed":
        reuse_full_tts = retimed_subtitle_path.is_file() and tts_review_is_current(
            tts_review_path, list(enumerate(translated_segments, 1))
        )
        if reuse_full_tts:
            cached_retimed_segments = read_srt(retimed_subtitle_path)
            reuse_full_tts = bool(cached_retimed_segments)
    with stage_context(
        PipelineStage.TTS,
        "Using existing English voice-over" if reuse_full_tts else "Generating English voice-over",
        reused=reuse_full_tts,
    ):
        if tts.generation_mode == "chunked":
            logger.info("Generating/rebuilding chunked Qwen TTS audio: %s", tts_audio_path)
            retimed_segments = synthesize_tts_audio_by_time_chunks(
                segments=translated_segments,
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
                regenerate_all_chunks=tts.regenerate,
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
            if tts.rerun_chunk is not None:
                warn(
                    logger,
                    "tts.rerun_chunk only applies to chunked generation and will be ignored.",
                )
            if reuse_full_tts:
                if tts.mode == "timed":
                    retimed_segments = cached_retimed_segments
            elif tts.mode == "simple":
                synthesize_simple_tts_audio(
                    segments=translated_segments,
                    audio_out=tts_audio_path,
                    tts_model_name=tts_model_name,
                    tts_language=tts.language,
                    tts_speaker=tts.speaker,
                    tts_instruct=tts.instruct,
                    device=hardware.device,
                    attn_implementation=tts.attn_implementation,
                )
            else:
                retimed_segments = synthesize_timed_tts_audio(
                    segments=translated_segments,
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

            if tts.split_audio and not reuse_full_tts:
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
    subtitle_path = translated_srt_path
    if isinstance(retimed_segments, list):
        subtitle_path = retimed_subtitle_path
        if not reuse_full_tts:
            write_srt(retimed_segments, subtitle_path, post_process=False)
        emit(
            PipelineStage.SUBTITLES,
            "Using existing retimed English subtitles"
            if reuse_full_tts
            else "Retimed English subtitles ready",
            kind=EventKind.REUSED if reuse_full_tts else EventKind.ARTIFACT,
            artifact=subtitle_path,
            details={"category": "Generated subtitles"},
        )
    with (
        stage_context(PipelineStage.RENDER, "Burning English subtitles"),
        ffmpeg_progress_handler(
            ffmpeg_events(PipelineStage.RENDER, "Burning English subtitles", duration)
        ),
    ):
        burn_subtitles(
            video_path,
            subtitle_path,
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
    _process_speedup_outputs_if_enabled(
        (subtitled_output_path, final_tts_output_path),
        speedup_spec_path,
        video_path.stem,
        settings,
    )


def _process_speedup_outputs_if_enabled(
    normal_output_paths: tuple[Path, ...],
    spec_path: Path,
    video_stem: str,
    settings: RunSettings,
) -> None:
    if not settings.speedup.enabled:
        return
    process_speedup_outputs(
        normal_output_paths,
        spec_path,
        video_encoder=settings.hardware.video_encoder,
        video_stem=video_stem,
    )

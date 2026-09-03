from __future__ import annotations

import logging
from pathlib import Path

from ..config import ProjectPaths, RunSettings
from ..events import NullObserver, PipelineEvent, PipelineObserver, PipelineStage
from ..process_runner import FFmpegProgress, ffmpeg_progress_handler
from .media import (
    burn_subtitles,
    mux_audio_into_video_mix,
    mux_audio_into_video_replace,
    split_audio_into_chunks,
)
from .models import get_model_filename_suffix
from .subtitles import post_process_segments, read_srt, write_srt
from .transcription import transcribe_video, translate_segments_with_vinai
from .tts import (
    synthesize_simple_tts_audio,
    synthesize_timed_tts_audio,
    synthesize_tts_audio_by_time_chunks,
)


def process_video(
    video_path: Path,
    model_path: Path,
    translation_model_path: Path | None,
    paths: ProjectPaths,
    settings: RunSettings,
    observer: PipelineObserver | None = None,
) -> None:
    """Generate subtitles, optionally generate TTS, and render the output video."""
    return _process_video(
        video_path, model_path, translation_model_path, paths, settings, observer or NullObserver()
    )


def _process_video(
    video_path: Path,
    model_path: Path,
    translation_model_path: Path | None,
    paths: ProjectPaths,
    settings: RunSettings,
    observer: PipelineObserver,
) -> None:
    transcription = settings.transcription
    hardware = settings.hardware
    tts = settings.tts
    model_suffix = get_model_filename_suffix(model_path, translation_model_path)
    srt_path = paths.subtitle_dir / f"{video_path.stem}_{model_suffix}.srt"

    subtitled_output_path = paths.output_dir / f"{video_path.stem}_vi-dub_en-sub.mp4"
    tts_audio_path = paths.audio_dir / f"{video_path.stem}_tts.wav"
    tts_chunks_dir = paths.audio_dir / f"{video_path.stem}_tts_chunks"
    final_tts_output_path = paths.output_dir / f"{video_path.stem}_en-dub_en-sub.mp4"

    logger = logging.getLogger(__name__)
    logger.info("Processing: %s", video_path.name)

    if srt_path.exists() and not transcription.overwrite_srt:
        logger.info("Reusing existing SRT: %s", srt_path)
        observer.notify(
            PipelineEvent(PipelineStage.SUBTITLES, "Reusing cached subtitles", artifact=srt_path)
        )
        segments = read_srt(srt_path)
        if not segments:
            raise ValueError(f"SRT contains no valid subtitles: {srt_path}")
    else:
        logger.info("Transcribing audio")
        observer.notify(PipelineEvent(PipelineStage.TRANSCRIBE, "Transcribing audio"))

        # A separate text translation model needs source-language transcription first.
        transcription_task = (
            "transcribe" if translation_model_path is not None else transcription.task
        )
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
            logger.info("Translating segments with VinAI")
            observer.notify(
                PipelineEvent(PipelineStage.TRANSLATE, "Translating subtitles", total=len(segments))
            )
            segments = translate_segments_with_vinai(
                segments=segments,
                model_path=translation_model_path,
                device=hardware.device,
                batch_size=transcription.translation_batch_size,
            )

        segments = post_process_segments(segments)
        if not segments:
            raise ValueError(f"No valid subtitles were generated for: {video_path.name}")

        logger.info("Writing subtitles: %s", srt_path)
        write_srt(segments, srt_path)
        observer.notify(
            PipelineEvent(PipelineStage.SUBTITLES, "Subtitles written", artifact=srt_path)
        )

    if transcription.skip_burn:
        logger.info("Subtitle-only mode enabled")
        observer.notify(
            PipelineEvent(
                PipelineStage.COMPLETE, "Subtitle-only processing complete", artifact=srt_path
            )
        )
        return

    logger.info("Burning subtitles: %s", subtitled_output_path)
    observer.notify(
        PipelineEvent(
            PipelineStage.RENDER, "Rendering subtitled video", artifact=subtitled_output_path
        )
    )
    with ffmpeg_progress_handler(
        _ffmpeg_events(observer, PipelineStage.RENDER, "Rendering subtitles")
    ):
        burn_subtitles(
            video_path,
            srt_path,
            subtitled_output_path,
            video_encoder=hardware.video_encoder,
        )

    if not tts.enabled:
        logger.info("Completed: %s", subtitled_output_path)
        observer.notify(
            PipelineEvent(
                PipelineStage.COMPLETE, "Video processing complete", artifact=subtitled_output_path
            )
        )
        return

    if tts.generation_mode == "chunked":
        logger.info("Generating/rebuilding chunked Qwen TTS audio: %s", tts_audio_path)
        observer.notify(
            PipelineEvent(
                PipelineStage.TTS, "Generating chunked voice-over", artifact=tts_audio_path
            )
        )
        synthesize_tts_audio_by_time_chunks(
            segments=segments,
            audio_out=tts_audio_path,
            chunks_dir=tts_chunks_dir,
            video_path=video_path,
            tts_model_name=tts.model,
            tts_language=tts.language,
            tts_speaker=tts.speaker,
            tts_instruct=tts.instruct,
            device=hardware.device,
            attn_implementation=tts.attn_implementation,
            chunk_minutes=tts.chunk_minutes,
            rerun_chunk=tts.rerun_chunk,
            overwrite_all_chunks=tts.overwrite,
            max_speedup=tts.max_speedup,
            chunk_tail_seconds=tts.chunk_tail_seconds,
        )
    else:
        if tts_audio_path.exists() and not tts.overwrite and tts.rerun_chunk is None:
            logger.info("Reusing existing TTS audio: %s", tts_audio_path)
        else:
            if tts.rerun_chunk is not None:
                logger.warning(
                    "tts.rerun_chunk only applies to chunked generation and will be ignored."
                )
            logger.info("Generating Qwen TTS audio: %s", tts_audio_path)
            observer.notify(
                PipelineEvent(PipelineStage.TTS, "Generating voice-over", artifact=tts_audio_path)
            )
            if tts.mode == "simple":
                synthesize_simple_tts_audio(
                    segments=segments,
                    audio_out=tts_audio_path,
                    tts_model_name=tts.model,
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
                    tts_model_name=tts.model,
                    tts_language=tts.language,
                    tts_speaker=tts.speaker,
                    tts_instruct=tts.instruct,
                    device=hardware.device,
                    attn_implementation=tts.attn_implementation,
                )

        if tts.split_audio:
            logger.info(
                "Splitting TTS audio into %d-minute review chunks: %s",
                tts.chunk_minutes,
                tts_chunks_dir,
            )
            with ffmpeg_progress_handler(
                _ffmpeg_events(observer, PipelineStage.TTS, "Splitting review audio")
            ):
                split_audio_into_chunks(
                    audio_in=tts_audio_path,
                    output_dir=tts_chunks_dir,
                    chunk_minutes=tts.chunk_minutes,
                    overwrite=True,
                )

    logger.info("Muxing TTS audio: %s", final_tts_output_path)
    observer.notify(
        PipelineEvent(PipelineStage.MUX, "Muxing voice-over", artifact=final_tts_output_path)
    )
    with ffmpeg_progress_handler(_ffmpeg_events(observer, PipelineStage.MUX, "Muxing voice-over")):
        if tts.audio_mode == "mix":
            mux_audio_into_video_mix(subtitled_output_path, tts_audio_path, final_tts_output_path)
        else:
            mux_audio_into_video_replace(
                subtitled_output_path, tts_audio_path, final_tts_output_path
            )

    logger.info("Completed: %s", final_tts_output_path)
    observer.notify(
        PipelineEvent(
            PipelineStage.COMPLETE, "Video processing complete", artifact=final_tts_output_path
        )
    )


def _ffmpeg_events(observer: PipelineObserver, stage: PipelineStage, message: str):
    def notify(progress: FFmpegProgress) -> None:
        speed = f" at {progress.speed}" if progress.speed else ""
        elapsed = (
            f" ({progress.elapsed_seconds:.1f}s)" if progress.elapsed_seconds is not None else ""
        )
        observer.notify(
            PipelineEvent(
                stage,
                f"{message}{elapsed}{speed}",
                current=progress.elapsed_seconds,
                details={"speed": progress.speed, "state": progress.state},
            )
        )

    return notify

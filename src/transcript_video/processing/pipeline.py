from __future__ import annotations

import logging
from pathlib import Path

from ..config import ProjectPaths, RunSettings
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
) -> None:
    """Generate subtitles, optionally generate TTS, and render the output video."""
    transcription = settings.transcription
    hardware = settings.hardware
    tts = settings.tts
    model_suffix = get_model_filename_suffix(model_path, translation_model_path)
    srt_path = paths.subtitle_dir / f"{video_path.stem}_{model_suffix}.srt"

    subtitled_output_path = paths.output_dir / f"{video_path.stem}_vi-dub_en-sub.mp4"
    tts_audio_path = paths.audio_dir / f"{video_path.stem}_tts.wav"
    tts_chunks_dir = paths.audio_dir / f"{video_path.stem}_tts_chunks"
    final_tts_output_path = paths.output_dir / f"{video_path.stem}_en-dub_en-sub.mp4"

    logging.info("Processing: %s", video_path.name)

    if srt_path.exists() and not transcription.overwrite_srt:
        logging.info("Reusing existing SRT: %s", srt_path)
        segments = read_srt(srt_path)
        if not segments:
            raise ValueError(f"SRT contains no valid subtitles: {srt_path}")
    else:
        logging.info("Step 1: Transcribing/translating...")

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
            logging.info("Step 2: Translating segments with VinAI...")
            segments = translate_segments_with_vinai(
                segments=segments,
                model_path=translation_model_path,
                device=hardware.device,
                batch_size=transcription.translation_batch_size,
            )

        segments = post_process_segments(segments)
        if not segments:
            raise ValueError(f"No valid subtitles were generated for: {video_path.name}")

        write_step = 3 if translation_model_path is not None else 2
        logging.info("Step %d: Writing SRT: %s", write_step, srt_path)
        write_srt(segments, srt_path)

    if transcription.skip_burn:
        logging.info("Subtitle-only mode enabled. Finished after writing the SRT file.")
        return

    burn_step = 4 if translation_model_path is not None else 3
    logging.info("Step %d: Burning subtitles: %s", burn_step, subtitled_output_path)
    burn_subtitles(
        video_path,
        srt_path,
        subtitled_output_path,
        video_encoder=hardware.video_encoder,
    )

    if not tts.enabled:
        logging.info("TTS disabled. DONE: %s", subtitled_output_path)
        return

    if tts.generation_mode == "chunked":
        logging.info(
            "Step %d: Generating/rebuilding chunked Qwen TTS audio: %s",
            burn_step + 1,
            tts_audio_path,
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
            logging.info("Reusing existing TTS audio: %s", tts_audio_path)
        else:
            if tts.rerun_chunk is not None:
                logging.warning(
                    "tts.rerun_chunk only applies to chunked generation and will be ignored."
                )
            logging.info("Step %d: Generating Qwen TTS audio: %s", burn_step + 1, tts_audio_path)
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
            logging.info(
                "Step %d: Splitting TTS audio into %d-minute review chunks: %s",
                burn_step + 2,
                tts.chunk_minutes,
                tts_chunks_dir,
            )
            split_audio_into_chunks(
                audio_in=tts_audio_path,
                output_dir=tts_chunks_dir,
                chunk_minutes=tts.chunk_minutes,
                overwrite=True,
            )

    mux_step = burn_step + 2
    logging.info("Step %d: Muxing TTS audio: %s", mux_step, final_tts_output_path)
    if tts.audio_mode == "mix":
        mux_audio_into_video_mix(subtitled_output_path, tts_audio_path, final_tts_output_path)
    else:
        mux_audio_into_video_replace(subtitled_output_path, tts_audio_path, final_tts_output_path)

    logging.info("DONE: %s", final_tts_output_path)

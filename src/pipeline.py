from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .media import (
    burn_subtitles,
    mux_audio_into_video_mix,
    mux_audio_into_video_replace,
    split_audio_into_chunks,
)
from .model_utils import get_model_filename_suffix
from .project_config import ProjectPaths
from .subtitles import read_srt, write_srt
from .transcription import transcribe_video, translate_segments_with_vinai
from .tts import (
    synthesize_simple_tts_audio,
    synthesize_timed_tts_audio,
    synthesize_tts_audio_by_time_chunks,
)


def process_video(
	video_path: Path,
	model_path: Path,
	translation_model_path: Optional[Path],
	paths: ProjectPaths,
	task: str,
	language: Optional[str],
	device: str,
	compute_type: str,
	translation_batch_size: int,
	overwrite_srt: bool,
	skip_burn: bool,
	enable_tts: bool,
	overwrite_tts: bool,
	tts_mode: str,
	tts_generation_mode: str,
	rerun_tts_chunk: Optional[int],
	tts_model: str,
	tts_language: str,
	tts_speaker: str,
	tts_instruct: str,
	tts_attn_implementation: str,
	audio_mode: str,
	split_tts_audio: bool,
	tts_chunk_minutes: int,
	tts_max_speedup: float,
	tts_chunk_tail_seconds: float,
) -> None:
	"""Generate SRT, burn subtitles, optionally generate chunked TTS and mux audio."""
	model_suffix = get_model_filename_suffix(model_path, translation_model_path)
	srt_path = paths.subtitle_dir / f"{video_path.stem}_{model_suffix}.srt"

	subtitled_output_path = paths.output_dir / f"{video_path.stem}_vi-dub_en-sub.mp4"
	tts_audio_path = paths.audio_dir / f"{video_path.stem}_tts.wav"
	tts_chunks_dir = paths.audio_dir / f"{video_path.stem}_tts_chunks"
	final_tts_output_path = paths.output_dir / f"{video_path.stem}_en-dub_en-sub.mp4"

	logging.info("Processing: %s", video_path.name)

	if srt_path.exists() and not overwrite_srt:
		logging.info("SRT đã tồn tại, đọc lại SRT cũ: %s", srt_path)
		segments = read_srt(srt_path)
	else:
		logging.info("Step 1: Transcribing/translating...")

		# If a separate text translation model is used, Whisper must first transcribe.
		transcription_task = "transcribe" if translation_model_path is not None else task
		segments = transcribe_video(
			video_path=video_path,
			model_path=model_path,
			paths=paths,
			task=transcription_task,
			language=language,
			device=device,
			compute_type=compute_type,
		)

		if translation_model_path is not None:
			logging.info("Step 2: Translating segments with VinAI...")
			segments = translate_segments_with_vinai(
				segments=segments,
				model_path=translation_model_path,
				device=device,
				batch_size=translation_batch_size,
			)

		write_step = 3 if translation_model_path is not None else 2
		logging.info("Step %d: Writing SRT: %s", write_step, srt_path)
		write_srt(segments, srt_path)

	if skip_burn:
		logging.info("Skip burn enabled. Done after SRT generation.")
		return

	burn_step = 4 if translation_model_path is not None else 3
	logging.info("Step %d: Burning subtitles to video: %s", burn_step, subtitled_output_path)
	burn_subtitles(video_path, srt_path, subtitled_output_path)

	if not enable_tts:
		logging.info("TTS disabled. DONE: %s", subtitled_output_path)
		return

	# ===== TTS generation =====
	if tts_generation_mode == "chunked":
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
			tts_model_name=tts_model,
			tts_language=tts_language,
			tts_speaker=tts_speaker,
			tts_instruct=tts_instruct,
			device=device,
			attn_implementation=tts_attn_implementation,
			chunk_minutes=tts_chunk_minutes,
			rerun_chunk=rerun_tts_chunk,
			overwrite_all_chunks=overwrite_tts,
			max_speedup=tts_max_speedup,
			chunk_tail_seconds=tts_chunk_tail_seconds,
		)
	else:
		if tts_audio_path.exists() and not overwrite_tts and rerun_tts_chunk is None:
			logging.info("TTS audio đã tồn tại, bỏ qua generate: %s", tts_audio_path)
		else:
			if rerun_tts_chunk is not None:
				logging.warning(
					"--rerun-tts-chunk chỉ có tác dụng với --tts-generation-mode chunked. "
					"Đang bỏ qua rerun chunk trong full mode."
				)
			logging.info("Step %d: Generating Qwen TTS audio: %s", burn_step + 1, tts_audio_path)
			if tts_mode == "simple":
				synthesize_simple_tts_audio(
					segments=segments,
					audio_out=tts_audio_path,
					tts_model_name=tts_model,
					tts_language=tts_language,
					tts_speaker=tts_speaker,
					tts_instruct=tts_instruct,
					device=device,
					attn_implementation=tts_attn_implementation,
				)
			else:
				synthesize_timed_tts_audio(
					segments=segments,
					audio_out=tts_audio_path,
					video_path=video_path,
					tts_model_name=tts_model,
					tts_language=tts_language,
					tts_speaker=tts_speaker,
					tts_instruct=tts_instruct,
					device=device,
					attn_implementation=tts_attn_implementation,
				)

		if split_tts_audio:
			logging.info(
				"Step %d: Splitting full TTS audio into %d-minute review chunks: %s",
				burn_step + 2,
				tts_chunk_minutes,
				tts_chunks_dir,
			)
			split_audio_into_chunks(
				audio_in=tts_audio_path,
				output_dir=tts_chunks_dir,
				chunk_minutes=tts_chunk_minutes,
				overwrite=True,
			)

	mux_step = burn_step + 2
	logging.info("Step %d: Muxing TTS audio into video: %s", mux_step, final_tts_output_path)
	if audio_mode == "mix":
		mux_audio_into_video_mix(subtitled_output_path, tts_audio_path, final_tts_output_path)
	else:
		mux_audio_into_video_replace(subtitled_output_path, tts_audio_path, final_tts_output_path)

	logging.info("DONE: %s", final_tts_output_path)

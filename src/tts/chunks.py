from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..media import get_media_duration_seconds
from ..project_config import SubtitleSegment
from .core import (
    fit_wav_to_available_duration,
    generate_qwen_custom_voice,
    load_qwen_tts_model,
)


def build_fixed_time_tts_chunks(
	segments: List[SubtitleSegment],
	chunk_minutes: int,
	video_path: Path,
) -> List[Tuple[int, float, float, List[SubtitleSegment]]]:
	"""Build fixed-length chunk groups that cover the whole video timeline."""
	if chunk_minutes <= 0:
		raise ValueError("chunk_minutes phải lớn hơn 0.")

	valid_segments = [segment for segment in segments if segment.text.strip()]
	if not valid_segments:
		raise ValueError("Không có subtitle segment nào để tạo TTS audio.")

	chunk_seconds = chunk_minutes * 60
	video_duration = get_media_duration_seconds(video_path) or 0.0
	subtitle_duration = max(segment.end for segment in valid_segments) + 1.0
	total_duration = max(video_duration, subtitle_duration, chunk_seconds)

	num_chunks = int((total_duration + chunk_seconds - 1) // chunk_seconds)
	chunks: List[Tuple[int, float, float, List[SubtitleSegment]]] = []

	for chunk_index in range(num_chunks):
		chunk_start = chunk_index * chunk_seconds
		chunk_end = min((chunk_index + 1) * chunk_seconds, total_duration)
		chunk_segments = [
			segment
			for segment in valid_segments
			if chunk_start <= segment.start < chunk_end
		]
		chunks.append((chunk_index, chunk_start, chunk_end, chunk_segments))

	return chunks


def synthesize_one_fixed_time_chunk(
	model,
	chunk_index: int,
	chunk_start: float,
	chunk_end: float,
	chunk_segments: List[SubtitleSegment],
	chunk_audio_out: Path,
	tts_language: str,
	tts_speaker: str,
	tts_instruct: str,
	sample_rate: Optional[int] = None,
	max_speedup: float = 1.15,
	all_segments: Optional[List[SubtitleSegment]] = None,
	video_duration: Optional[float] = None,
	chunk_tail_seconds: float = 10.0,
) -> int:
	"""Generate one fixed-time TTS chunk with a safe tail after the chunk boundary.

	Why the tail is needed:
	- A subtitle can start at 04:58 while the fixed chunk ends at 05:00.
	- If the generated voice needs 5 seconds, cutting the chunk at 05:00 loses the last 3 seconds.
	- This function lets the chunk WAV extend a few seconds past chunk_end.
	- Later, chunks are rebuilt by overlaying each chunk at its original timeline start,
	  not by concatenating them, so the extra tail does not shift the following chunks.
	"""
	import numpy as np
	import soundfile as sf

	chunk_audio_out.parent.mkdir(parents=True, exist_ok=True)

	GAP_SECONDS = 0.08
	MIN_SLOT_SECONDS = 0.35
	chunk_tail_seconds = max(0.0, float(chunk_tail_seconds))

	# Sort all segments once so we can find the next subtitle globally.
	# This prevents the last subtitle in a chunk from being blindly cut at chunk_end.
	global_segments = sorted(
		all_segments if all_segments is not None else chunk_segments,
		key=lambda item: item.start,
	)

	def find_next_global_start(current_segment: SubtitleSegment) -> Optional[float]:
		for candidate in global_segments:
			if candidate.start > current_segment.start + 1e-6:
				return candidate.start
		return None

	# IMPORTANT:
	# Every non-empty chunk must allocate extra tail duration from the beginning.
	# In the old version, only chunk 0 got the tail because sample_rate was None.
	# Later chunks reused sample_rate from the previous chunk, so their buffers stayed exactly 5 minutes
	# and audio that crossed the chunk boundary was still cut off.
	sr_final = sample_rate or 24000
	base_chunk_duration = max(0.1, chunk_end - chunk_start)
	chunk_duration_with_tail = max(0.1, chunk_end - chunk_start + chunk_tail_seconds)
	full_chunk = np.zeros(int(chunk_duration_with_tail * sr_final), dtype=np.float32)

	if not chunk_segments:
		# Silence chunks do not need tail because there is no voice line to preserve.
		full_chunk = np.zeros(int(base_chunk_duration * sr_final), dtype=np.float32)
		sf.write(str(chunk_audio_out), full_chunk, sr_final)
		logging.info("Chunk %03d has no subtitle. Wrote silence: %s", chunk_index, chunk_audio_out)
		return sr_final

	generated_items = []

	for local_index, segment in enumerate(chunk_segments):
		text = segment.text.strip()
		logging.info(
			"TTS chunk %03d segment %d/%d: %s",
			chunk_index,
			local_index + 1,
			len(chunk_segments),
			text[:90],
		)

		wav, sr = generate_qwen_custom_voice(
			model=model,
			text=text,
			language=tts_language,
			speaker=tts_speaker,
			instruct=tts_instruct,
		)
		wav = np.asarray(wav, dtype=np.float32)

		if local_index == 0:
			if sample_rate is None:
				sr_final = sr
				# Recreate the buffer with the real model sample rate, still including the tail.
				full_chunk = np.zeros(int(chunk_duration_with_tail * sr_final), dtype=np.float32)
			elif sr != sr_final:
				raise ValueError(f"Sample rate không đồng nhất: {sr} != {sr_final}")
		elif sr != sr_final:
			raise ValueError(f"Sample rate không đồng nhất: {sr} != {sr_final}")

		# Limit by the next subtitle in the whole video, not just inside this chunk.
		# If the next subtitle starts after the 5-minute boundary, this segment can continue
		# past chunk_end as long as it remains before the next global subtitle.
		next_global_start = find_next_global_start(segment)
		if next_global_start is not None:
			slot_end = next_global_start - GAP_SECONDS
		else:
			# Last subtitle in the video: allow it until the video end if known,
			# otherwise use its SRT end time plus a small tail.
			if video_duration is not None and video_duration > 0:
				slot_end = video_duration
			else:
				slot_end = segment.end + chunk_tail_seconds

		# Do not allow this chunk audio to write infinitely past its review chunk.
		# The tail is only a safety margin for boundary-crossing lines.
		max_tail_end = chunk_end + chunk_tail_seconds
		slot_end = min(slot_end, max_tail_end)

		available_duration = max(MIN_SLOT_SECONDS, slot_end - segment.start)
		original_duration = len(wav) / sr_final

		if original_duration > available_duration:
			logging.warning(
				"Chunk %03d segment %d too long: %.2fs > %.2fs. Fit with max speedup %.2fx.",
				chunk_index,
				local_index + 1,
				original_duration,
				available_duration,
				max_speedup,
			)
			wav = fit_wav_to_available_duration(
				wav=wav,
				sample_rate=sr_final,
				available_duration=available_duration,
				max_speedup=max_speedup,
			)

		local_start = max(0.0, segment.start - chunk_start)
		generated_items.append((local_start, wav))

	for local_start, wav in generated_items:
		start_sample = int(local_start * sr_final)
		end_sample = start_sample + len(wav)

		if start_sample >= len(full_chunk):
			continue

		if end_sample > len(full_chunk):
			wav = wav[: len(full_chunk) - start_sample]
			end_sample = len(full_chunk)

		full_chunk[start_sample:end_sample] += wav

	full_chunk = np.clip(full_chunk, -1.0, 1.0)
	sf.write(str(chunk_audio_out), full_chunk, sr_final)
	logging.info("Wrote TTS chunk %03d: %s", chunk_index, chunk_audio_out)
	return sr_final


def rebuild_full_tts_audio_from_chunks(
	chunk_infos: List[Tuple[Path, float, float]],
	audio_out: Path,
	expected_total_duration: Optional[float] = None,
) -> None:
	"""Rebuild full TTS WAV by placing every chunk at its original timeline position.

	Do NOT concatenate chunks when chunks have a tail.
	Example:
	- chunk_000 starts at 00:00 and may be 5:10 long because of a tail.
	- chunk_001 must still start at exactly 05:00.
	Therefore, we overlay each chunk at chunk_start instead of doing chunk_000 + chunk_001.
	"""
	import numpy as np
	import soundfile as sf

	if not chunk_infos:
		raise ValueError("Không có chunk audio nào để ghép lại.")

	loaded_chunks = []
	final_sr = None

	for chunk_path, chunk_start, _chunk_end in chunk_infos:
		if not chunk_path.exists():
			raise FileNotFoundError(f"Thiếu chunk audio: {chunk_path}")

		wav, sr = sf.read(str(chunk_path), dtype="float32")
		if final_sr is None:
			final_sr = sr
		elif sr != final_sr:
			raise ValueError(f"Sample rate chunk không đồng nhất: {sr} != {final_sr}")

		loaded_chunks.append((chunk_start, wav))

	if final_sr is None:
		raise ValueError("Không đọc được sample rate từ chunk audio.")

	if expected_total_duration is not None and expected_total_duration > 0:
		total_samples = int(expected_total_duration * final_sr)
	else:
		total_samples = 0
		for chunk_start, wav in loaded_chunks:
			total_samples = max(total_samples, int(chunk_start * final_sr) + len(wav))

	# Add a tiny safety buffer to avoid rounding off the last sample.
	total_samples = max(1, total_samples + int(0.05 * final_sr))
	full_audio = np.zeros(total_samples, dtype=np.float32)

	for chunk_start, wav in loaded_chunks:
		start_sample = max(0, int(chunk_start * final_sr))
		end_sample = start_sample + len(wav)

		if start_sample >= len(full_audio):
			continue

		if end_sample > len(full_audio):
			wav = wav[: len(full_audio) - start_sample]
			end_sample = len(full_audio)

		full_audio[start_sample:end_sample] += wav

	full_audio = np.clip(full_audio, -1.0, 1.0)
	audio_out.parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(audio_out), full_audio, final_sr)
	logging.info("Rebuilt full TTS audio from timeline-overlaid chunks: %s", audio_out)


def synthesize_tts_audio_by_time_chunks(
	segments: List[SubtitleSegment],
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
	rerun_chunk: Optional[int] = None,
	overwrite_all_chunks: bool = False,
	max_speedup: float = 1.15,
	chunk_tail_seconds: float = 10.0,
) -> None:
	"""Generate TTS by fixed time chunks and rebuild full audio.

	Boundary-safe behavior:
	- Chunks are still organized by 5-minute review windows.
	- Each generated chunk can contain a small tail after its official end.
	- Full audio is rebuilt by overlaying chunks at their original chunk_start timestamp.
	- This prevents losing the end of lines that start just before a chunk boundary.
	"""
	if rerun_chunk is not None and rerun_chunk < 0:
		raise ValueError("--rerun-tts-chunk phải >= 0.")
	if chunk_tail_seconds < 0:
		raise ValueError("--tts-chunk-tail-seconds phải >= 0.")

	valid_segments = sorted(
		[segment for segment in segments if segment.text.strip()],
		key=lambda item: item.start,
	)
	if not valid_segments:
		raise ValueError("Không có subtitle segment nào để tạo TTS audio.")

	chunks = build_fixed_time_tts_chunks(
		segments=valid_segments,
		chunk_minutes=chunk_minutes,
		video_path=video_path,
	)
	chunks_dir.mkdir(parents=True, exist_ok=True)

	video_duration = get_media_duration_seconds(video_path) or 0.0
	subtitle_duration = max(segment.end for segment in valid_segments) + 1.0
	expected_total_duration = max(video_duration, subtitle_duration)

	if rerun_chunk is not None and rerun_chunk >= len(chunks):
		raise ValueError(
			f"Chunk {rerun_chunk} không tồn tại. Video này chỉ có chunk 0 đến {len(chunks) - 1}."
		)

	chunk_infos = []
	chunks_to_generate = []

	for chunk_index, chunk_start, chunk_end, chunk_segments in chunks:
		chunk_path = chunks_dir / f"{audio_out.stem}_chunk_{chunk_index:03d}.wav"
		should_generate = (
			overwrite_all_chunks
			or rerun_chunk == chunk_index
			or not chunk_path.exists()
		)
		chunk_infos.append((chunk_index, chunk_start, chunk_end, chunk_segments, chunk_path, should_generate))
		if should_generate:
			chunks_to_generate.append(chunk_index)

	if chunks_to_generate:
		logging.info("Chunks to generate/regenerate: %s", chunks_to_generate)
		logging.info("Chunk tail safety margin: %.2fs", chunk_tail_seconds)
		model = load_qwen_tts_model(tts_model_name, device, attn_implementation)
		sample_rate = None

		for chunk_index, chunk_start, chunk_end, chunk_segments, chunk_path, should_generate in chunk_infos:
			if not should_generate:
				logging.info("Chunk %03d đã tồn tại, bỏ qua generate: %s", chunk_index, chunk_path)
				continue

			logging.info(
				"Generating TTS chunk %03d / %03d | %.2fs → %.2fs (+%.2fs tail) | %d segment(s)",
				chunk_index,
				len(chunks) - 1,
				chunk_start,
				chunk_end,
				chunk_tail_seconds,
				len(chunk_segments),
			)
			sample_rate = synthesize_one_fixed_time_chunk(
				model=model,
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
				all_segments=valid_segments,
				video_duration=video_duration,
				chunk_tail_seconds=chunk_tail_seconds,
			)
	else:
		logging.info("All TTS chunks already exist. Rebuilding full WAV without loading TTS model.")

	rebuild_full_tts_audio_from_chunks(
		chunk_infos=[(chunk_path, chunk_start, chunk_end) for (_chunk_index, chunk_start, chunk_end, _segments, chunk_path, _should_generate) in chunk_infos],
		audio_out=audio_out,
		expected_total_duration=expected_total_duration,
	)

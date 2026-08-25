from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .media import (
    ensure_ffmpeg_available_for_transformers,
    extract_audio,
    read_wav_as_float32_array,
)
from .model_utils import detect_model_type, detect_translation_model_type
from .project_config import ProjectPaths, SubtitleSegment


def transcribe_with_faster_whisper(
	video_path: Path,
	model_path: Path,
	task: str,
	language: Optional[str],
	device: str,
	compute_type: str,
) -> List[SubtitleSegment]:
	"""Transcribe/translate using faster-whisper."""
	from faster_whisper import WhisperModel

	logging.info("Engine: faster-whisper")
	model = WhisperModel(str(model_path), device=device, compute_type=compute_type)

	segments, _info = model.transcribe(
		str(video_path),
		task=task,
		language=language,
		beam_size=5,

		# Anti-hallucination settings:
		# 1) Do not let a hallucinated previous segment affect the next segment.
		condition_on_previous_text=False,

		# 2) VAD removes silent/non-speech parts before Whisper decoding.
		vad_filter=True,
		vad_parameters=dict(
			min_silence_duration_ms=500,
			speech_pad_ms=300,
		),

		# 3) Conservative thresholds to reduce false speech in silent/noisy regions.
		no_speech_threshold=0.6,
		compression_ratio_threshold=2.4,
		log_prob_threshold=-1.0,
	)

	return [SubtitleSegment(s.start, s.end, s.text or "") for s in segments]


def transcribe_with_huggingface(
	video_path: Path,
	model_path: Path,
	temp_dir: Path,
	task: str,
	device: str,
) -> List[SubtitleSegment]:
	"""Transcribe/translate using the standard Hugging Face Whisper model."""
	# Must run before importing Transformers because it caches ffmpeg availability.
	ensure_ffmpeg_available_for_transformers()

	import torch
	from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

	logging.info("Engine: Hugging Face Transformers")
	if device == "cuda" and not torch.cuda.is_available():
		logging.warning("CUDA không khả dụng, tự chuyển sang CPU.")
		device = "cpu"

	torch_dtype = torch.float16 if device == "cuda" else torch.float32
	hf_device = 0 if device == "cuda" else -1

	audio_path = temp_dir / f"{video_path.stem}_audio.wav"
	extract_audio(video_path, audio_path)

	try:
		use_safetensors = (model_path / "model.safetensors").exists()
		model = AutoModelForSpeechSeq2Seq.from_pretrained(
			str(model_path),
			torch_dtype=torch_dtype,
			low_cpu_mem_usage=True,
			use_safetensors=use_safetensors,
		)
		processor = AutoProcessor.from_pretrained(str(model_path))

		asr_pipeline = pipeline(
			"automatic-speech-recognition",
			model=model,
			tokenizer=processor.tokenizer,
			feature_extractor=processor.feature_extractor,
			chunk_length_s=30,
			device=hf_device,
			torch_dtype=torch_dtype,
		)

		# Do NOT pass the audio filename to Transformers here.
		# Passing a filename makes Transformers call its own ffmpeg loader again,
		# which is exactly what caused: "ffmpeg was not found".
		audio_input = read_wav_as_float32_array(audio_path)
		result = asr_pipeline(
			audio_input,
			generate_kwargs={"task": task},
			return_timestamps=True,
		)

		segments: List[SubtitleSegment] = []
		for chunk in result.get("chunks", []):
			timestamp = chunk.get("timestamp")
			text = chunk.get("text", "")
			if not timestamp or timestamp[0] is None:
				continue

			start = float(timestamp[0])
			end = float(timestamp[1]) if timestamp[1] is not None else start + 3.0
			segments.append(SubtitleSegment(start, end, text))

		return segments

	finally:
		if audio_path.exists():
			audio_path.unlink()


def translate_segments_with_vinai(
	segments: List[SubtitleSegment],
	model_path: Path,
	device: str,
	batch_size: int,
) -> List[SubtitleSegment]:
	"""Translate Vietnamese subtitle segments to English with VinAI Translate."""
	if batch_size < 1:
		raise ValueError("--translation-batch-size phải lớn hơn 0.")

	detect_translation_model_type(model_path)

	import torch
	from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

	if device == "cuda" and not torch.cuda.is_available():
		logging.warning("CUDA không khả dụng cho VinAI Translate, tự chuyển sang CPU.")
		device = "cpu"

	torch_device = torch.device(device)
	logging.info("Translation engine: VinAI Translate (%s)", torch_device)

	tokenizer = AutoTokenizer.from_pretrained(str(model_path), src_lang="vi_VN")
	model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
	model.to(torch_device)
	model.eval()

	source_segments = [segment for segment in segments if (segment.text or "").strip()]
	translated_segments: List[SubtitleSegment] = []

	for start in range(0, len(source_segments), batch_size):
		batch = source_segments[start : start + batch_size]
		texts = [segment.text.strip() for segment in batch]
		inputs = tokenizer(
			texts,
			padding=True,
			truncation=True,
			max_length=1024,
			return_tensors="pt",
		).to(torch_device)

		with torch.inference_mode():
			output_ids = model.generate(
				**inputs,
				decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
				num_return_sequences=1,
				num_beams=5,
				early_stopping=True,
			)

		translations = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
		translated_segments.extend(
			SubtitleSegment(segment.start, segment.end, translation)
			for segment, translation in zip(batch, translations)
		)
		logging.info(
			"Translated %d/%d subtitle segment(s).",
			min(start + batch_size, len(source_segments)),
			len(source_segments),
		)

	return translated_segments


def transcribe_video(
	video_path: Path,
	model_path: Path,
	paths: ProjectPaths,
	task: str,
	language: Optional[str],
	device: str,
	compute_type: str,
) -> List[SubtitleSegment]:
	"""Choose the correct engine and transcribe/translate video."""
	model_type = detect_model_type(model_path)

	if model_type == "faster-whisper":
		return transcribe_with_faster_whisper(
			video_path=video_path,
			model_path=model_path,
			task=task,
			language=language,
			device=device,
			compute_type=compute_type,
		)

	return transcribe_with_huggingface(
		video_path=video_path,
		model_path=model_path,
		temp_dir=paths.temp_dir,
		task=task,
		device=device,
	)

"""
Professional video transcription + subtitle + TTS voice-over pipeline.

Folder structure:
	project/
	├── input/       # put original videos here
	├── subtitles/   # generated .srt files
	├── audio/       # generated .wav TTS audio files and 5-minute review chunks
	├── output/      # final videos
	└── temp/        # temporary extracted audio

Basic usage:
	python transcript_video_tts_complete.py
	python transcript_video_tts_complete.py --video my_video.mp4

Generate subtitles only, then burn subtitles:
	python transcript_video_tts_complete.py --video my_video.mp4 --task translate --language vi

Generate subtitles + Qwen TTS audio + final video with TTS audio:
	python transcript_video_tts_complete.py --video my_video.mp4 --task translate --language vi --enable-tts

By default, when TTS is enabled, the generated TTS WAV is also split into 5-minute chunks:
	audio/<video_name>_tts_chunks/<video_name>_tts_part_000.wav

Use a local Whisper model folder:
	python transcript_video_tts_complete.py --model "C:/Users/<USERNAME>/.faster-whisper-large-v3"

Install needed packages:
	pip install imageio-ffmpeg faster-whisper
	pip install -U qwen-tts soundfile numpy

Optional for Hugging Face Whisper:
	pip install transformers torch accelerate safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import imageio_ffmpeg


# =========================
# Configuration
# =========================

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
DEFAULT_MODEL_PATH = r"C:\Users\AHG5HC\.faster-whisper-large-v3"

MODEL_FILENAME_SUFFIXES = {
	"faster-whisper": "faster",
	"huggingface": "huggingface",
}

TRANSLATION_MODEL_FILENAME_SUFFIXES = {
	"vinai-translate": "vinai",
}

DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


@dataclass
class SubtitleSegment:
	"""A normalized subtitle segment used by all transcription engines."""

	start: float
	end: float
	text: str


@dataclass
class ProjectPaths:
	"""All important folders in the project."""

	root: Path
	input_dir: Path
	subtitle_dir: Path
	audio_dir: Path
	output_dir: Path
	temp_dir: Path

	@classmethod
	def from_root(cls, root: Path) -> "ProjectPaths":
		return cls(
			root=root,
			input_dir=root / "input",
			subtitle_dir=root / "subtitles",
			audio_dir=root / "audio",
			output_dir=root / "output",
			temp_dir=root / "temp",
		)

	def create_dirs(self) -> None:
		for folder in [
			self.input_dir,
			self.subtitle_dir,
			self.audio_dir,
			self.output_dir,
			self.temp_dir,
		]:
			folder.mkdir(parents=True, exist_ok=True)


# =========================
# Utility functions
# =========================


def setup_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="[%(levelname)s] %(message)s",
	)


def format_timestamp(seconds: Optional[float]) -> str:
	"""Convert seconds to SRT timestamp format: HH:MM:SS,mmm."""
	if seconds is None or seconds < 0:
		seconds = 0.0

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	secs = int(seconds % 60)
	millis = int(round((seconds - int(seconds)) * 1000))

	# Handle rare rounding case: 59.9996 -> 60.000
	if millis == 1000:
		millis = 0
		secs += 1
	if secs == 60:
		secs = 0
		minutes += 1
	if minutes == 60:
		minutes = 0
		hours += 1

	return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def parse_srt_timestamp(timestamp: str) -> float:
	"""Convert SRT timestamp HH:MM:SS,mmm to seconds."""
	timestamp = timestamp.strip().replace(".", ",")
	match = re.match(r"^(\d+):(\d{2}):(\d{2}),(\d{1,3})$", timestamp)
	if not match:
		raise ValueError(f"Invalid SRT timestamp: {timestamp}")

	hours, minutes, seconds, millis = match.groups()
	return (
		int(hours) * 3600
		+ int(minutes) * 60
		+ int(seconds)
		+ int(millis.ljust(3, "0")[:3]) / 1000.0
	)


def write_srt(segments: Iterable[SubtitleSegment], srt_path: Path) -> None:
	"""Write subtitle segments to an .srt file."""
	srt_path.parent.mkdir(parents=True, exist_ok=True)

	with srt_path.open("w", encoding="utf-8") as file:
		index = 1
		for segment in segments:
			text = (segment.text or "").strip()
			if not text:
				continue

			start = format_timestamp(segment.start)
			end = format_timestamp(segment.end)
			file.write(f"{index}\n{start} --> {end}\n{text}\n\n")
			index += 1


def read_srt(srt_path: Path) -> List[SubtitleSegment]:
	"""Read an existing SRT file back into SubtitleSegment objects."""
	if not srt_path.exists():
		raise FileNotFoundError(f"Không tìm thấy SRT: {srt_path}")

	content = srt_path.read_text(encoding="utf-8-sig")
	blocks = re.split(r"\n\s*\n", content.strip())
	segments: List[SubtitleSegment] = []

	for block in blocks:
		lines = [line.strip() for line in block.splitlines() if line.strip()]
		if not lines:
			continue

		timing_line_index = None
		for i, line in enumerate(lines):
			if "-->" in line:
				timing_line_index = i
				break

		if timing_line_index is None:
			continue

		timing_line = lines[timing_line_index]
		start_text, end_text = [part.strip() for part in timing_line.split("-->", 1)]
		text = " ".join(lines[timing_line_index + 1 :]).strip()

		if not text:
			continue

		segments.append(
			SubtitleSegment(
				start=parse_srt_timestamp(start_text),
				end=parse_srt_timestamp(end_text),
				text=text,
			)
		)

	return segments


def escape_subtitle_path_for_ffmpeg(path: Path) -> str:
	"""Escape subtitle path so FFmpeg subtitles filter can read Windows paths."""
	escaped = str(path.resolve()).replace("\\", "/")
	escaped = escaped.replace(":", r"\:")
	escaped = escaped.replace("'", r"\'")
	return escaped


def run_command(command: Sequence[str], *, hide_output: bool = False) -> None:
	"""Run a subprocess command with error checking."""
	kwargs = {"check": True}
	if hide_output:
		kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})
	subprocess.run(list(command), **kwargs)


def ensure_ffmpeg_available_for_transformers() -> None:
	"""Make imageio-ffmpeg's bundled ffmpeg visible to Transformers/audio loaders."""
	ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
	ffmpeg_dir = str(ffmpeg_path.parent)
	os.environ["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg_path)

	current_path = os.environ.get("PATH", "")
	if ffmpeg_dir not in current_path.split(os.pathsep):
		os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path

	logging.info("FFmpeg available for Transformers: %s", ffmpeg_path)


def burn_subtitles(video_in: Path, srt_in: Path, video_out: Path) -> None:
	"""Burn hard subtitles into a video using FFmpeg."""
	video_out.parent.mkdir(parents=True, exist_ok=True)

	ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
	srt_escaped = escape_subtitle_path_for_ffmpeg(srt_in)

	# Default subtitle style with white text and black outline. You can customize this as needed.
	subtitle_style = (
		"FontName=Arial,"
		"FontSize=16,"
		"PrimaryColour=&H00FFFFFF,"
		"SecondaryColour=&H00FFFFFF,"
		"OutlineColour=&H00000000,"
		"BackColour=&H00000000,"
		"Bold=0,"
		"Italic=0,"
		"Underline=0,"
		"StrikeOut=0,"
		"ScaleX=100,"
		"ScaleY=100,"
		"Spacing=0,"
		"Angle=0,"
		"BorderStyle=1,"
		"Outline=1,"
		"Shadow=0,"
		"Alignment=2," # Bottom-center
		"MarginL=10,"
		"MarginR=10,"
		"MarginV=25," # Adjust vertical margin to move subtitles up/down
		"Encoding=1"
	)

	command = [
		ffmpeg_path,
		"-y",
		"-i",
		str(video_in),
		"-vf",
		# f"subtitles='{srt_escaped}':force_style='{subtitle_style}'",
		f"subtitles='{srt_escaped}':force_style='MarginV=25'",
		"-c:a",
		"copy",
		str(video_out),
	]
	run_command(command)


def extract_audio(video_path: Path, audio_path: Path) -> None:
	"""Extract mono 16 kHz WAV audio from video for Hugging Face pipeline."""
	ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
	audio_path.parent.mkdir(parents=True, exist_ok=True)

	command = [
		ffmpeg_path,
		"-y",
		"-i",
		str(video_path),
		"-vn",
		"-acodec",
		"pcm_s16le",
		"-ar",
		"16000",
		"-ac",
		"1",
		str(audio_path),
	]
	run_command(command, hide_output=True)


def read_wav_as_float32_array(audio_path: Path):
	"""Read extracted WAV with Python stdlib and return data for HF pipeline."""
	import wave

	import numpy as np

	with wave.open(str(audio_path), "rb") as wav_file:
		sample_rate = wav_file.getframerate()
		n_channels = wav_file.getnchannels()
		sample_width = wav_file.getsampwidth()
		n_frames = wav_file.getnframes()
		raw_audio = wav_file.readframes(n_frames)

	if sample_width != 2:
		raise ValueError(f"Expected 16-bit PCM WAV, got sample width: {sample_width}")

	audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

	if n_channels > 1:
		audio = audio.reshape(-1, n_channels).mean(axis=1)

	return {"array": audio, "sampling_rate": sample_rate}


def find_videos(input_dir: Path, selected_video: Optional[str] = None) -> List[Path]:
	"""Find videos in input folder. If selected_video is given, process only that file."""
	if selected_video:
		video_path = input_dir / selected_video
		if not video_path.exists():
			raise FileNotFoundError(f"Không tìm thấy video: {video_path}")
		if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
			raise ValueError(f"File không phải định dạng video được hỗ trợ: {video_path.name}")
		return [video_path]

	videos = sorted(
		path
		for path in input_dir.iterdir()
		if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
	)
	if not videos:
		raise FileNotFoundError(f"Không có video nào trong folder: {input_dir}")
	return videos


def get_media_duration_seconds(media_path: Path) -> Optional[float]:
	"""Return media duration by parsing FFmpeg output."""
	ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
	process = subprocess.run(
		[ffmpeg_path, "-i", str(media_path)],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		encoding="utf-8",
		errors="ignore",
	)
	output = process.stderr or process.stdout or ""
	match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
	if not match:
		return None

	hours, minutes, seconds = match.groups()
	return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def split_audio_into_chunks(
	audio_in: Path,
	output_dir: Path,
	chunk_minutes: int = 5,
	overwrite: bool = True,
) -> None:
	"""Split a long audio file into smaller review chunks.

	This only splits the generated TTS WAV file for easier checking.
	It does not split the video, subtitle, or original input audio.
	"""
	if chunk_minutes <= 0:
		raise ValueError("chunk_minutes phải lớn hơn 0.")
	if not audio_in.exists():
		raise FileNotFoundError(f"Không tìm thấy audio để split: {audio_in}")

	ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
	output_dir.mkdir(parents=True, exist_ok=True)

	# Remove old chunks first. This avoids leaving stale part_010.wav, part_011.wav, ...
	# from a previous longer generation.
	if overwrite:
		for old_chunk in output_dir.glob(f"{audio_in.stem}_part_*{audio_in.suffix}"):
			try:
				old_chunk.unlink()
			except OSError:
				logging.warning("Không thể xóa chunk cũ: %s", old_chunk)

	chunk_seconds = int(chunk_minutes * 60)
	output_pattern = output_dir / f"{audio_in.stem}_part_%03d{audio_in.suffix}"

	command = [
		ffmpeg_path,
		"-y",
		"-i",
		str(audio_in),
		"-f",
		"segment",
		"-segment_time",
		str(chunk_seconds),
		"-reset_timestamps",
		"1",
		"-c",
		"copy",
		str(output_pattern),
	]
	run_command(command)
	logging.info("Split TTS audio into %d-minute chunks: %s", chunk_minutes, output_dir)


# =========================
# Transcription engines
# =========================


def read_transformers_model_config(model_path: Path) -> dict:
	"""Read a local Transformers config file when one is available."""
	config_path = model_path / "config.json"
	if not config_path.exists():
		return {}

	with config_path.open("r", encoding="utf-8") as file:
		return json.load(file)


def has_transformers_model_weights(model_path: Path) -> bool:
	"""Return whether a local Transformers model folder contains model weights."""
	return (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists()


def detect_model_type(model_path: Path) -> str:
	"""Detect whether the ASR model folder is faster-whisper or Hugging Face Whisper."""
	if (model_path / "model.bin").exists():
		return "faster-whisper"

	config = read_transformers_model_config(model_path)
	if has_transformers_model_weights(model_path) and config.get("model_type") == "whisper":
		return "huggingface"

	if has_transformers_model_weights(model_path) and config.get("model_type") == "mbart":
		raise ValueError(
			f"Model tại {model_path} là model dịch văn bản. "
			"Hãy truyền nó qua --translation-model và dùng model Whisper cho --model."
		)

	raise ValueError(f"Không nhận diện được định dạng model tại: {model_path}")


def detect_translation_model_type(model_path: Path) -> str:
	"""Detect a supported local text translation model."""
	config = read_transformers_model_config(model_path)
	if has_transformers_model_weights(model_path) and config.get("model_type") == "mbart":
		return "vinai-translate"

	raise ValueError(f"Không nhận diện được model dịch văn bản tại: {model_path}")


def get_model_filename_suffix(model_path: Path, translation_model_path: Optional[Path] = None) -> str:
	"""Return the short model name used as a subtitle filename suffix."""
	suffix = MODEL_FILENAME_SUFFIXES[detect_model_type(model_path)]
	if translation_model_path is not None:
		translation_type = detect_translation_model_type(translation_model_path)
		suffix += f"_{TRANSLATION_MODEL_FILENAME_SUFFIXES[translation_type]}"
	return suffix


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


# =========================
# Qwen TTS functions
# =========================


def split_text_for_tts(text: str, max_chars: int = 450) -> List[str]:
	"""Split long text into smaller chunks for TTS generation."""
	text = re.sub(r"\s+", " ", text).strip()
	if not text:
		return []

	sentence_parts = re.split(r"(?<=[.!?。！？])\s+", text)
	chunks: List[str] = []
	current = ""

	for sentence in sentence_parts:
		sentence = sentence.strip()
		if not sentence:
			continue

		if len(sentence) > max_chars:
			if current:
				chunks.append(current.strip())
				current = ""
			for i in range(0, len(sentence), max_chars):
				chunks.append(sentence[i : i + max_chars].strip())
			continue

		if len(current) + len(sentence) + 1 <= max_chars:
			current = f"{current} {sentence}".strip()
		else:
			if current:
				chunks.append(current.strip())
			current = sentence

	if current:
		chunks.append(current.strip())

	return chunks


def load_qwen_tts_model(
	tts_model_name: str,
	device: str,
	attn_implementation: str,
):
	"""Load Qwen TTS model lazily so the script can still run without TTS."""
	import torch
	from qwen_tts import Qwen3TTSModel

	if device == "cuda" and not torch.cuda.is_available():
		logging.warning("CUDA không khả dụng cho Qwen TTS, tự chuyển sang CPU.")
		device = "cpu"

	kwargs = {
		"device_map": "cuda:0" if device == "cuda" else "cpu",
		"dtype": torch.float16 if device == "cuda" else torch.float32,
	}

	if attn_implementation != "auto":
		kwargs["attn_implementation"] = attn_implementation

	logging.info("Loading Qwen TTS model: %s", tts_model_name)
	return Qwen3TTSModel.from_pretrained(tts_model_name, **kwargs)


def generate_qwen_custom_voice(
	model,
	text: str,
	language: str,
	speaker: str,
	instruct: str,
) -> Tuple[object, int]:
	"""Generate one waveform with Qwen CustomVoice."""
	wavs, sr = model.generate_custom_voice(
		text=text,
		language=language,
		speaker=speaker,
		instruct=instruct,
	)
	return wavs[0], sr


def synthesize_simple_tts_audio(
	segments: List[SubtitleSegment],
	audio_out: Path,
	tts_model_name: str,
	tts_language: str,
	tts_speaker: str,
	tts_instruct: str,
	device: str,
	attn_implementation: str,
) -> None:
	"""
	Generate a simple continuous audio file from all subtitle text.

	This is easy to generate, but it is not timestamp-aligned.
	For most video dubbing use-cases, synthesize_timed_tts_audio is better.
	"""
	import numpy as np
	import soundfile as sf

	text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
	chunks = split_text_for_tts(text)
	if not chunks:
		raise ValueError("Không có text để tạo TTS audio.")

	model = load_qwen_tts_model(tts_model_name, device, attn_implementation)

	wav_list = []
	sample_rate = None
	for index, chunk in enumerate(chunks, start=1):
		logging.info("TTS chunk %d/%d", index, len(chunks))
		wav, sr = generate_qwen_custom_voice(
			model=model,
			text=chunk,
			language=tts_language,
			speaker=tts_speaker,
			instruct=tts_instruct,
		)
		wav = np.asarray(wav, dtype=np.float32)
		wav_list.append(wav)
		sample_rate = sr if sample_rate is None else sample_rate
		if sr != sample_rate:
			raise ValueError(f"Sample rate không đồng nhất: {sr} != {sample_rate}")

	full_audio = np.concatenate(wav_list)
	full_audio = np.clip(full_audio, -1.0, 1.0)
	audio_out.parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(audio_out), full_audio, sample_rate)


def synthesize_timed_tts_audio(
	segments: List[SubtitleSegment],
	audio_out: Path,
	video_path: Path,
	tts_model_name: str,
	tts_language: str,
	tts_speaker: str,
	tts_instruct: str,
	device: str,
	attn_implementation: str,
) -> None:
	"""
	Generate one WAV file by placing every TTS segment at its subtitle timestamp.

	Important fix:
	- Each TTS segment is only allowed to play until before the next subtitle starts.
	- If the generated TTS is too long, it is sped up.
	- If it is still too long, it is trimmed with a small fade-out.
	"""
	import numpy as np
	import soundfile as sf

	# ===== Timing control =====
	MIN_GAP_SECONDS = 0.08       # small silence before next voice starts
	MAX_SPEEDUP = 1.35           # do not speed up too much, or voice becomes unnatural
	MIN_SLOT_SECONDS = 0.35      # minimum allowed slot
	FADE_OUT_SECONDS = 0.04      # fade out when trimming

	def speedup_wav_by_resample(wav: np.ndarray, speed_factor: float) -> np.ndarray:
		"""
		Simple in-memory speed-up.
		Note: this changes pitch slightly, but avoids overlap without extra dependencies.
		"""
		if speed_factor <= 1.0 or len(wav) < 2:
			return wav

		new_length = max(1, int(len(wav) / speed_factor))
		old_positions = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
		new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)

		return np.interp(new_positions, old_positions, wav).astype(np.float32)

	def trim_with_fade_out(wav: np.ndarray, max_samples: int, sample_rate: int) -> np.ndarray:
		"""Trim wav to max_samples and apply a small fade-out to avoid clicking."""
		if max_samples <= 0:
			return np.zeros(0, dtype=np.float32)

		if len(wav) <= max_samples:
			return wav

		wav = wav[:max_samples].copy()
		fade_samples = min(len(wav), int(FADE_OUT_SECONDS * sample_rate))

		if fade_samples > 0:
			wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

		return wav

	valid_segments = [segment for segment in segments if segment.text.strip()]
	if not valid_segments:
		raise ValueError("Không có subtitle segment nào để tạo TTS audio.")

	model = load_qwen_tts_model(tts_model_name, device, attn_implementation)

	generated_chunks = []
	sample_rate = None
	video_duration = get_media_duration_seconds(video_path)

	for index, segment in enumerate(valid_segments):
		text = segment.text.strip()
		logging.info("TTS segment %d/%d: %s", index + 1, len(valid_segments), text[:80])

		wav, sr = generate_qwen_custom_voice(
			model=model,
			text=text,
			language=tts_language,
			speaker=tts_speaker,
			instruct=tts_instruct,
		)

		wav = np.asarray(wav, dtype=np.float32)

		if sample_rate is None:
			sample_rate = sr
		elif sr != sample_rate:
			raise ValueError(f"Sample rate không đồng nhất: {sr} != {sample_rate}")

		# Slot end = before next subtitle starts.
		if index + 1 < len(valid_segments):
			next_start = valid_segments[index + 1].start
			slot_end = next_start - MIN_GAP_SECONDS
		else:
			# Last segment can run until video end, or at least its own subtitle end.
			slot_end = video_duration if video_duration else segment.end + 1.0

		available_duration = max(MIN_SLOT_SECONDS, slot_end - segment.start)
		available_samples = int(available_duration * sample_rate)

		original_duration = len(wav) / sample_rate

		if original_duration > available_duration:
			needed_speedup = original_duration / available_duration
			speedup = min(needed_speedup, MAX_SPEEDUP)

			logging.warning(
				"TTS segment %d too long: %.2fs > %.2fs. Speedup %.2fx.",
				index + 1,
				original_duration,
				available_duration,
				speedup,
			)

			wav = speedup_wav_by_resample(wav, speedup)

			if len(wav) > available_samples:
				logging.warning(
					"TTS segment %d still too long after speedup. Trim to %.2fs.",
					index + 1,
					available_duration,
				)
				wav = trim_with_fade_out(wav, available_samples, sample_rate)

		generated_chunks.append((segment, wav))

	subtitle_duration = max(segment.end for segment in valid_segments) + 1.0
	total_duration = max(video_duration or 0.0, subtitle_duration)
	total_samples = int(total_duration * sample_rate) + sample_rate
	full_audio = np.zeros(total_samples, dtype=np.float32)

	for segment, wav in generated_chunks:
		start_sample = max(0, int(segment.start * sample_rate))
		end_sample = start_sample + len(wav)

		if start_sample >= len(full_audio):
			continue

		if end_sample > len(full_audio):
			wav = wav[: len(full_audio) - start_sample]
			end_sample = len(full_audio)

		# No overlap now because each wav was already fitted to its slot.
		full_audio[start_sample:end_sample] += wav

	full_audio = np.clip(full_audio, -1.0, 1.0)
	audio_out.parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(audio_out), full_audio, sample_rate)



def fit_wav_to_available_duration(
	wav,
	sample_rate: int,
	available_duration: float,
	max_speedup: float = 1.15,
	fade_out_seconds: float = 0.04,
):
	"""Fit a generated TTS waveform into an available time slot.

	If the waveform is longer than the slot, it is sped up slightly by
	resampling. If it is still too long, it is trimmed with a small fade-out.
	"""
	import numpy as np

	wav = np.asarray(wav, dtype=np.float32)
	available_samples = max(1, int(available_duration * sample_rate))

	if len(wav) <= available_samples:
		return wav

	original_duration = len(wav) / sample_rate
	needed_speedup = original_duration / max(available_duration, 1e-6)
	speedup = min(max(1.0, needed_speedup), max_speedup)

	if speedup > 1.0 and len(wav) > 1:
		new_length = max(1, int(len(wav) / speedup))
		old_positions = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
		new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
		wav = np.interp(new_positions, old_positions, wav).astype(np.float32)

	if len(wav) > available_samples:
		wav = wav[:available_samples].copy()
		fade_samples = min(len(wav), int(fade_out_seconds * sample_rate))
		if fade_samples > 0:
			wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

	return wav


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
) -> int:
	"""Generate one fixed-time TTS chunk.

	The chunk keeps local timing, so when chunks are concatenated later,
	the full audio stays aligned with the original video timeline.
	"""
	import numpy as np
	import soundfile as sf

	chunk_audio_out.parent.mkdir(parents=True, exist_ok=True)

	# If a chunk has no subtitle, create pure silence.
	# Use the previous sample_rate if known, otherwise use 24000 as a safe fallback.
	sr_final = sample_rate or 24000
	chunk_duration = max(0.1, chunk_end - chunk_start)
	full_chunk = np.zeros(int(chunk_duration * sr_final), dtype=np.float32)

	if not chunk_segments:
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

		if sample_rate is None and local_index == 0:
			sr_final = sr
			chunk_duration = max(0.1, chunk_end - chunk_start)
			full_chunk = np.zeros(int(chunk_duration * sr_final), dtype=np.float32)
		elif sr != sr_final:
			raise ValueError(f"Sample rate không đồng nhất: {sr} != {sr_final}")

		# Limit this voice line to before the next subtitle in the same chunk.
		# If this is the last line in the chunk, limit it to chunk_end.
		if local_index + 1 < len(chunk_segments):
			slot_end = chunk_segments[local_index + 1].start - 0.08
		else:
			slot_end = chunk_end

		available_duration = max(0.35, slot_end - segment.start)
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
	chunk_paths: List[Path],
	audio_out: Path,
) -> None:
	"""Concatenate all fixed-time TTS chunks into one full TTS WAV."""
	import numpy as np
	import soundfile as sf

	if not chunk_paths:
		raise ValueError("Không có chunk audio nào để ghép lại.")

	wav_list = []
	final_sr = None

	for chunk_path in chunk_paths:
		if not chunk_path.exists():
			raise FileNotFoundError(f"Thiếu chunk audio: {chunk_path}")

		wav, sr = sf.read(str(chunk_path), dtype="float32")
		if final_sr is None:
			final_sr = sr
		elif sr != final_sr:
			raise ValueError(f"Sample rate chunk không đồng nhất: {sr} != {final_sr}")

		wav_list.append(wav)

	full_audio = np.concatenate(wav_list)
	full_audio = np.clip(full_audio, -1.0, 1.0)
	audio_out.parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(audio_out), full_audio, final_sr)
	logging.info("Rebuilt full TTS audio from chunks: %s", audio_out)


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
) -> None:
	"""Generate TTS by fixed time chunks and rebuild full audio.

	This enables review/regeneration workflows:
	- First run: generate chunk_000.wav, chunk_001.wav, ... then rebuild full WAV.
	- If chunk N is bad: pass rerun_chunk=N to regenerate only that chunk,
	  then rebuild the full WAV and mux it into the video again.
	"""
	if rerun_chunk is not None and rerun_chunk < 0:
		raise ValueError("--rerun-tts-chunk phải >= 0.")

	chunks = build_fixed_time_tts_chunks(
		segments=segments,
		chunk_minutes=chunk_minutes,
		video_path=video_path,
	)
	chunks_dir.mkdir(parents=True, exist_ok=True)

	if rerun_chunk is not None and rerun_chunk >= len(chunks):
		raise ValueError(
			f"Chunk {rerun_chunk} không tồn tại. Video này chỉ có chunk 0 đến {len(chunks) - 1}."
		)

	chunk_infos = []
	chunk_paths: List[Path] = []
	chunks_to_generate = []

	for chunk_index, chunk_start, chunk_end, chunk_segments in chunks:
		chunk_path = chunks_dir / f"{audio_out.stem}_chunk_{chunk_index:03d}.wav"
		chunk_paths.append(chunk_path)
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
		model = load_qwen_tts_model(tts_model_name, device, attn_implementation)
		sample_rate = None

		for chunk_index, chunk_start, chunk_end, chunk_segments, chunk_path, should_generate in chunk_infos:
			if not should_generate:
				logging.info("Chunk %03d đã tồn tại, bỏ qua generate: %s", chunk_index, chunk_path)
				continue

			logging.info(
				"Generating TTS chunk %03d / %03d | %.2fs → %.2fs | %d segment(s)",
				chunk_index,
				len(chunks) - 1,
				chunk_start,
				chunk_end,
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
			)
	else:
		logging.info("All TTS chunks already exist. Rebuilding full WAV without loading TTS model.")

	rebuild_full_tts_audio_from_chunks(chunk_paths=chunk_paths, audio_out=audio_out)

def mux_audio_into_video_replace(video_in: Path, audio_in: Path, video_out: Path) -> None:
	"""Replace the video's original audio with generated TTS audio."""
	ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
	video_out.parent.mkdir(parents=True, exist_ok=True)

	command = [
		ffmpeg_path,
		"-y",
		"-i",
		str(video_in),
		"-i",
		str(audio_in),
		"-map",
		"0:v:0",
		"-map",
		"1:a:0",
		"-c:v",
		"copy",
		"-c:a",
		"aac",
		"-b:a",
		"192k",
		str(video_out),
	]
	run_command(command)


def mux_audio_into_video_mix(video_in: Path, audio_in: Path, video_out: Path) -> None:
	"""Mix original video audio with generated TTS audio."""
	ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
	video_out.parent.mkdir(parents=True, exist_ok=True)

	command = [
		ffmpeg_path,
		"-y",
		"-i",
		str(video_in),
		"-i",
		str(audio_in),
		"-filter_complex",
		"[0:a:0][1:a:0]amix=inputs=2:duration=first:dropout_transition=2[aout]",
		"-map",
		"0:v:0",
		"-map",
		"[aout]",
		"-c:v",
		"copy",
		"-c:a",
		"aac",
		"-b:a",
		"192k",
		str(video_out),
	]
	run_command(command)


# =========================
# Main process
# =========================


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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Transcribe/translate videos, burn subtitles, generate Qwen TTS audio, and mux it into video."
	)
	parser.add_argument(
		"--root",
		default=".",
		help="Project root folder. Default: current folder.",
	)
	parser.add_argument(
		"--video",
		default=None,
		help="Process only one video file inside input/. Example: lecture01.mp4",
	)
	parser.add_argument(
		"--model",
		default=DEFAULT_MODEL_PATH,
		help="Path to faster-whisper or Hugging Face Whisper model folder.",
	)
	parser.add_argument(
		"--translation-model",
		default=None,
		help="Optional path to a VinAI vi2en text translation model. Requires --task translate.",
	)
	parser.add_argument(
		"--task",
		choices=["translate", "transcribe"],
		default="translate",
		help="translate = output English subtitles, transcribe = keep original language.",
	)
	parser.add_argument(
		"--language",
		default="vi",
		help="Source language hint for faster-whisper. Use 'vi' for Vietnamese. Use empty string to auto-detect.",
	)
	parser.add_argument(
		"--device",
		choices=["cuda", "cpu"],
		default="cuda",
		help="Device for inference.",
	)
	parser.add_argument(
		"--compute-type",
		default="int8",
		help="faster-whisper compute type. Common values: int8, float16, float32.",
	)
	parser.add_argument(
		"--translation-batch-size",
		type=int,
		default=8,
		help="Number of subtitle segments translated together by VinAI. Default: 8.",
	)
	parser.add_argument(
		"--overwrite-srt",
		action="store_true",
		help="Regenerate SRT even if it already exists.",
	)
	parser.add_argument(
		"--skip-burn",
		action="store_true",
		help="Only generate SRT, do not create output video or TTS audio.",
	)

	# TTS options
	parser.add_argument(
		"--enable-tts",
		action="store_true",
		help="Generate Qwen TTS audio and mux it into the subtitled video.",
	)
	parser.add_argument(
		"--overwrite-tts",
		action="store_true",
		help="Regenerate TTS WAV even if it already exists.",
	)
	parser.add_argument(
		"--tts-mode",
		choices=["timed", "simple"],
		default="timed",
		help="Used only in full TTS generation mode. timed = place each TTS line at subtitle timestamp; simple = one continuous voice-over.",
	)
	parser.add_argument(
		"--tts-generation-mode",
		choices=["chunked", "full"],
		default="chunked",
		help="chunked = generate fixed 5-minute TTS chunks and rebuild full audio; full = old behavior, generate one full WAV first.",
	)
	parser.add_argument(
		"--rerun-tts-chunk",
		type=int,
		default=None,
		help="Regenerate only one TTS chunk by index, e.g. 3. Works with --tts-generation-mode chunked.",
	)
	parser.add_argument(
		"--tts-model",
		default=DEFAULT_TTS_MODEL,
		help="Qwen TTS model id or local model folder.",
	)
	parser.add_argument(
		"--tts-language",
		default="English",
		help="TTS language for Qwen. Use English if your subtitles are translated to English.",
	)
	parser.add_argument(
		"--tts-speaker",
		default="Aiden",
		help="Qwen CustomVoice speaker. Examples: Ryan, Aiden, Vivian, Serena.",
	)
	parser.add_argument(
		"--tts-instruct",
		default=(
			"Speak clearly and professionally in a calm teaching voice. "
			"Use a steady medium pace and neutral tone. "
			"Do not laugh, act, dramatize, or add emotions. "
			"Read the text exactly."
		),
		help="Natural language instruction for the TTS voice style.",
	)
	parser.add_argument(
		"--tts-attn-implementation",
		choices=["auto", "flash_attention_2", "sdpa", "eager"],
		default="auto",
		help="Attention implementation for Qwen TTS. Use flash_attention_2 only if installed correctly.",
	)
	parser.add_argument(
		"--audio-mode",
		choices=["replace", "mix"],
		default="replace",
		help="replace = replace original audio with TTS; mix = mix original audio and TTS.",
	)
	parser.add_argument(
		"--no-split-tts-audio",
		action="store_true",
		help="Do not split the generated TTS WAV into review chunks. By default, TTS WAV is split into 5-minute chunks when --enable-tts is used.",
	)
	parser.add_argument(
		"--tts-chunk-minutes",
		type=int,
		default=5,
		help="Chunk length in minutes for generated TTS WAV review files. Default: 5.",
	)
	parser.add_argument(
		"--tts-max-speedup",
		type=float,
		default=1.15,
		help="Maximum speed-up ratio used to fit a TTS sentence into its timing slot. Default: 1.15.",
	)

	return parser.parse_args()


def main() -> None:
	setup_logging()
	args = parse_args()

	root = Path(args.root).resolve()
	model_path = Path(args.model).expanduser().resolve()
	translation_model_path = (
		Path(args.translation_model).expanduser().resolve()
		if args.translation_model
		else None
	)
	language = args.language.strip() or None

	paths = ProjectPaths.from_root(root)
	paths.create_dirs()

	if not model_path.exists():
		raise FileNotFoundError(f"Không tìm thấy model folder: {model_path}")

	if translation_model_path is not None:
		if not translation_model_path.exists():
			raise FileNotFoundError(f"Không tìm thấy translation model folder: {translation_model_path}")
		if args.task != "translate":
			raise ValueError("--translation-model chỉ được dùng cùng --task translate.")
		detect_translation_model_type(translation_model_path)

	if args.enable_tts and args.task == "transcribe" and args.tts_language.lower() == "english":
		logging.warning(
			"Bạn đang dùng --task transcribe nhưng --tts-language English. "
			"Nếu video gốc là tiếng Việt, Qwen CustomVoice có thể không đọc tiếng Việt tốt. "
			"Nên dùng --task translate để tạo subtitle tiếng Anh rồi TTS tiếng Anh."
		)

	if args.tts_chunk_minutes <= 0:
		raise ValueError("--tts-chunk-minutes phải lớn hơn 0.")
	if args.rerun_tts_chunk is not None and args.rerun_tts_chunk < 0:
		raise ValueError("--rerun-tts-chunk phải >= 0.")
	if args.tts_max_speedup < 1.0:
		raise ValueError("--tts-max-speedup phải >= 1.0.")

	videos = find_videos(paths.input_dir, args.video)
	logging.info("Found %d video(s).", len(videos))

	for video_path in videos:
		try:
			process_video(
				video_path=video_path,
				model_path=model_path,
				translation_model_path=translation_model_path,
				paths=paths,
				task=args.task,
				language=language,
				device=args.device,
				compute_type=args.compute_type,
				translation_batch_size=args.translation_batch_size,
				overwrite_srt=args.overwrite_srt,
				skip_burn=args.skip_burn,
				enable_tts=args.enable_tts,
				overwrite_tts=args.overwrite_tts,
				tts_mode=args.tts_mode,
				tts_generation_mode=args.tts_generation_mode,
				rerun_tts_chunk=args.rerun_tts_chunk,
				tts_model=args.tts_model,
				tts_language=args.tts_language,
				tts_speaker=args.tts_speaker,
				tts_instruct=args.tts_instruct,
				tts_attn_implementation=args.tts_attn_implementation,
				audio_mode=args.audio_mode,
				split_tts_audio=args.enable_tts and not args.no_split_tts_audio,
				tts_chunk_minutes=args.tts_chunk_minutes,
				tts_max_speedup=args.tts_max_speedup,
			)
		except Exception as exc:
			logging.exception("Failed: %s | Error: %s", video_path.name, exc)


if __name__ == "__main__":
	main()

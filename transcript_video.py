"""
Professional video transcription + subtitle burning pipeline.

Folder structure:
	project/
	├── input/       # put original videos here
	├── subtitles/   # generated .srt files
	├── output/      # videos with hard subtitles
	└── temp/        # temporary extracted audio

Example:
	python transcript_video.py
	python transcript_video.py --video my_video.mp4
	python transcript_video.py --model "C:/Users/<USERNAME>/.faster-whisper-large-v3"
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

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
	output_dir: Path
	temp_dir: Path

	@classmethod
	def from_root(cls, root: Path) -> "ProjectPaths":
		return cls(
			root=root,
			input_dir=root / "input",
			subtitle_dir=root / "subtitles",
			output_dir=root / "output",
			temp_dir=root / "temp",
		)

	def create_dirs(self) -> None:
		for folder in [self.input_dir, self.subtitle_dir, self.output_dir, self.temp_dir]:
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


def escape_subtitle_path_for_ffmpeg(path: Path) -> str:
	"""Escape subtitle path so FFmpeg subtitles filter can read Windows paths."""
	escaped = str(path.resolve()).replace("\\", "/")
	escaped = escaped.replace(":", r"\:")
	escaped = escaped.replace("'", r"\'")
	return escaped


def run_command(command: List[str], *, hide_output: bool = False) -> None:
	"""Run a subprocess command with error checking."""
	kwargs = {"check": True}
	if hide_output:
		kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})
	subprocess.run(command, **kwargs)




def ensure_ffmpeg_available_for_transformers() -> None:
	"""Make imageio-ffmpeg's bundled ffmpeg visible to Transformers/audio loaders.

	Transformers can load audio from a filename, but internally it searches for
	an `ffmpeg` executable in PATH. `imageio_ffmpeg.get_ffmpeg_exe()` gives us
	a valid ffmpeg executable, but it is not automatically added to PATH.
	"""
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

	command = [
		ffmpeg_path,
		"-y",
		"-i",
		str(video_in),
		"-vf",
		f"subtitles='{srt_escaped}'",
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
	"""Read extracted WAV with Python stdlib and return data for HF pipeline.

	This bypasses Transformers' filename audio loader, so it does not need
	ffmpeg in PATH after the audio has already been extracted by imageio-ffmpeg.
	"""
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
		path for path in input_dir.iterdir()
		if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
	)
	if not videos:
		raise FileNotFoundError(f"Không có video nào trong folder: {input_dir}")
	return videos


# =========================
# Transcription engines
# =========================

def detect_model_type(model_path: Path) -> str:
	"""Detect whether the model folder is faster-whisper or Hugging Face Whisper."""
	if (model_path / "model.bin").exists():
		return "faster-whisper"
	if (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists():
		return "huggingface"
	raise ValueError(f"Không nhận diện được định dạng model tại: {model_path}")


def get_model_filename_suffix(model_path: Path) -> str:
	"""Return the short model name used as a subtitle filename suffix."""
	return MODEL_FILENAME_SUFFIXES[detect_model_type(model_path)]


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
# Main process
# =========================

def process_video(
	video_path: Path,
	model_path: Path,
	paths: ProjectPaths,
	task: str,
	language: Optional[str],
	device: str,
	compute_type: str,
	overwrite_srt: bool,
	skip_burn: bool,
) -> None:
	"""Generate SRT and optionally burn subtitles into one video."""
	model_suffix = get_model_filename_suffix(model_path)
	srt_path = paths.subtitle_dir / f"{video_path.stem}_{model_suffix}.srt"
	output_path = paths.output_dir / f"{video_path.stem}_ENG_SUB.mp4"

	logging.info("Processing: %s", video_path.name)

	if srt_path.exists() and not overwrite_srt:
		logging.info("SRT đã tồn tại, bỏ qua transcription: %s", srt_path)
	else:
		logging.info("Step 1: Transcribing/translating...")
		segments = transcribe_video(
			video_path=video_path,
			model_path=model_path,
			paths=paths,
			task=task,
			language=language,
			device=device,
			compute_type=compute_type,
		)
		logging.info("Step 2: Writing SRT: %s", srt_path)
		write_srt(segments, srt_path)

	if skip_burn:
		logging.info("Skip burn enabled. Done after SRT generation.")
		return

	logging.info("Step 3: Burning subtitles to video: %s", output_path)
	burn_subtitles(video_path, srt_path, output_path)
	logging.info("DONE: %s", output_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Transcribe/translate videos from input/ and burn subtitles to output/."
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
		"--overwrite-srt",
		action="store_true",
		help="Regenerate SRT even if it already exists.",
	)
	parser.add_argument(
		"--skip-burn",
		action="store_true",
		help="Only generate SRT, do not create output video.",
	)
	return parser.parse_args()


def main() -> None:
	setup_logging()
	args = parse_args()

	root = Path(args.root).resolve()
	model_path = Path(args.model).expanduser().resolve()
	language = args.language.strip() or None

	paths = ProjectPaths.from_root(root)
	paths.create_dirs()

	if not model_path.exists():
		raise FileNotFoundError(f"Không tìm thấy model folder: {model_path}")

	videos = find_videos(paths.input_dir, args.video)
	logging.info("Found %d video(s).", len(videos))

	for video_path in videos:
		try:
			process_video(
				video_path=video_path,
				model_path=model_path,
				paths=paths,
				task=args.task,
				language=language,
				device=args.device,
				compute_type=args.compute_type,
				overwrite_srt=args.overwrite_srt,
				skip_burn=args.skip_burn,
			)
		except Exception as exc:
			logging.error("Failed: %s | Error: %s", video_path.name, exc)


if __name__ == "__main__":
	main()

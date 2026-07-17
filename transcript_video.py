"""
Video transcription -> LLM-edited SRT -> burn subtitles -> optional chunked Qwen TTS.

Workflow designed for manual LLM correction/translation:

1) Generate Vietnamese SRT only:
   python transcript_video_llm_srt_workflow.py --mode make-srt --video Report.mp4 --language vi

   Default ASR model location:
   model/.faster-whisper-large-v3
   or
   model/faster-whisper-large-v3

   Output:
   subtitles/Report_vi_raw.srt

2) Send that SRT to an LLM, fix/translate it, then save the edited file, for example:
   subtitles/Report_llm_en.srt

3) Burn the LLM-edited SRT into the video:
   python transcript_video_llm_srt_workflow.py --mode burn-srt --video Report.mp4 \
	   --srt-input subtitles/Report_llm_en.srt

4) Burn + generate Qwen TTS from the same LLM-edited SRT:
   python transcript_video_llm_srt_workflow.py --mode burn-srt --video Report.mp4 \
	   --srt-input subtitles/Report_llm_en.srt --enable-tts \
	   --tts-language English --tts-speaker Aiden --tts-attn-implementation sdpa

   Default TTS model location:
   model/Qwen3-TTS-12Hz-1.7B-CustomVoice

Folder structure:
	project/
	├── input/       # original videos
	├── model/       # local faster-whisper and Qwen TTS models
	├── subtitles/   # raw Vietnamese SRT and LLM-edited SRT
	├── audio/       # generated TTS WAV and review chunks
	├── output/      # final videos
	└── temp/        # reserved for temporary files
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import imageio_ffmpeg


# =========================
# Configuration
# =========================

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

# Default models are resolved from the project model/ folder.
# You can still override them with --model or --tts-model.
DEFAULT_ASR_MODEL_CANDIDATES = (
	# ".faster-whisper-large-v3",
	"faster-whisper-large-v3",
	# "faster-whisper-large-v3-ct2",
	# "whisper-large-v3",
)
DEFAULT_TTS_MODEL_CANDIDATES = (
	"Qwen3-TTS-12Hz-1.7B-CustomVoice",
	# "Qwen-Qwen3-TTS-12Hz-1.7B-CustomVoice",
)

SUPPORTED_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
COMMAND_OUTPUT_LIMIT = 4000

DEFAULT_SUBTITLE_STYLE = (
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
	"Alignment=2,"
	"MarginL=10,"
	"MarginR=10,"
	"MarginV=25,"
	"Encoding=1"
)


@dataclass
class SubtitleSegment:
	"""A normalized subtitle segment."""

	start: float
	end: float
	text: str


@dataclass
class ProjectPaths:
	"""All important folders in the project."""

	root: Path
	input_dir: Path
	model_dir: Path
	subtitle_dir: Path
	audio_dir: Path
	output_dir: Path
	temp_dir: Path

	@classmethod
	def from_root(cls, root: Path) -> "ProjectPaths":
		return cls(
			root=root,
			input_dir=root / "input",
			model_dir=root / "model",
			subtitle_dir=root / "subtitles",
			audio_dir=root / "audio",
			output_dir=root / "output",
			temp_dir=root / "temp",
		)

	def create_dirs(self) -> None:
		for folder in [
			self.input_dir,
			self.model_dir,
			self.subtitle_dir,
			self.audio_dir,
			self.output_dir,
			self.temp_dir,
		]:
			folder.mkdir(parents=True, exist_ok=True)


# =========================
# Utility functions
# =========================


def setup_logging(level: str = "INFO") -> None:
	"""Configure console logging so VS Code terminal shows what the script is doing."""
	normalized_level = (level or "INFO").upper()
	if normalized_level not in SUPPORTED_LOG_LEVELS:
		normalized_level = "INFO"

	logging.basicConfig(
		level=getattr(logging, normalized_level),
		format="[%(asctime)s] [%(levelname)s] %(message)s",
		datefmt="%H:%M:%S",
		stream=sys.stdout,
		force=True,
	)


def package_version(package_name: str) -> str:
	"""Return installed package version, or a clear fallback string."""
	try:
		return importlib_metadata.version(package_name)
	except Exception:
		return "not installed / unknown"


def ctranslate2_cuda_supported_compute_types() -> str:
	"""Return supported CUDA compute types for CTranslate2 if available."""
	try:
		import ctranslate2

		if hasattr(ctranslate2, "get_supported_compute_types"):
			return ", ".join(ctranslate2.get_supported_compute_types("cuda"))
	except Exception as exc:
		return f"unavailable ({exc})"
	return "unavailable"


def log_environment_diagnostics(paths: "ProjectPaths", args: argparse.Namespace) -> None:
	"""Print the runtime environment so CPU/GPU problems are visible immediately."""
	logging.info("================ Runtime diagnostics ================")
	logging.info("Script file: %s", Path(__file__).resolve())
	logging.info("Working dir: %s", Path.cwd())
	logging.info("Project root: %s", paths.root)
	logging.info("Python executable: %s", sys.executable)
	logging.info("python on PATH: %s", shutil.which("python") or "not found")
	logging.info("Conda env: %s", os.environ.get("CONDA_DEFAULT_ENV", "not active/unknown"))
	logging.info("Platform: %s | Python: %s", platform.platform(), platform.python_version())
	logging.info("Mode: %s | requested device: %s | log level: %s", args.mode, args.device, args.log_level)
	logging.info("faster-whisper: %s", package_version("faster-whisper"))
	logging.info("ctranslate2: %s", package_version("ctranslate2"))
	logging.info("qwen-tts: %s", package_version("qwen-tts"))
	logging.info("torch: %s", package_version("torch"))

	try:
		import torch

		logging.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
		logging.info("torch.version.cuda: %s", torch.version.cuda)
		if torch.cuda.is_available():
			logging.info("torch GPU[0]: %s", torch.cuda.get_device_name(0))
			logging.info("torch CUDA capability: %s", torch.cuda.get_device_capability(0))
	except Exception as exc:
		logging.warning("Cannot import/check torch CUDA: %s", exc)

	logging.info("CTranslate2 CUDA device count: %d", ctranslate2_cuda_device_count())
	logging.info("CTranslate2 CUDA compute types: %s", ctranslate2_cuda_supported_compute_types())
	logging.info("FFmpeg executable: %s", imageio_ffmpeg.get_ffmpeg_exe())
	logging.info("=====================================================")


def ctranslate2_cuda_device_count() -> int:
	"""Return available CUDA device count for faster-whisper/CTranslate2."""
	try:
		import ctranslate2

		if hasattr(ctranslate2, "get_cuda_device_count"):
			return int(ctranslate2.get_cuda_device_count())
	except Exception as exc:
		logging.debug("Cannot check CTranslate2 CUDA devices: %s", exc)
	return 0


def torch_cuda_available() -> bool:
	"""Return whether PyTorch can see CUDA for Qwen TTS."""
	try:
		import torch

		return bool(torch.cuda.is_available())
	except Exception as exc:
		logging.debug("Cannot check torch CUDA availability: %s", exc)
		return False


def choose_asr_device(requested_device: str) -> str:
	"""Resolve ASR device for faster-whisper. Force mode fails loudly instead of silently using CPU."""
	requested_device = (requested_device or "auto").lower()

	if requested_device == "cpu":
		logging.info("ASR device forced: cpu")
		return "cpu"

	cuda_count = ctranslate2_cuda_device_count()

	if requested_device == "cuda":
		if cuda_count <= 0:
			raise RuntimeError(
				"Bạn yêu cầu faster-whisper chạy GPU bằng --device cuda, "
				"nhưng CTranslate2 không thấy CUDA device nào. "
				"Kiểm tra NVIDIA driver, CUDA-enabled ctranslate2/faster-whisper, "
				"và đúng Python environment trong VS Code."
			)
		logging.info("ASR device: cuda | CTranslate2 CUDA devices: %d", cuda_count)
		return "cuda"

	if cuda_count > 0:
		logging.info("ASR device auto-selected: cuda | CTranslate2 CUDA devices: %d", cuda_count)
		return "cuda"

	logging.warning(
		"ASR auto-selected CPU vì CTranslate2 không thấy CUDA device. "
		"Nếu máy có NVIDIA GPU, hãy kiểm tra environment/cài đặt CUDA của faster-whisper."
	)
	return "cpu"


def choose_tts_device(requested_device: str) -> str:
	"""Resolve Qwen TTS device. Force mode fails loudly instead of silently using CPU."""
	requested_device = (requested_device or "auto").lower()

	if requested_device == "cpu":
		logging.info("TTS device forced: cpu")
		return "cpu"

	cuda_ok = torch_cuda_available()

	if requested_device == "cuda":
		if not cuda_ok:
			raise RuntimeError(
				"Bạn yêu cầu Qwen TTS chạy GPU bằng --device cuda, "
				"nhưng PyTorch không thấy CUDA. "
				"Kiểm tra torch CUDA build và đúng Python environment trong VS Code."
			)
		logging.info("TTS device: cuda")
		return "cuda"

	if cuda_ok:
		logging.info("TTS device auto-selected: cuda")
		return "cuda"

	logging.warning(
		"TTS auto-selected CPU vì PyTorch không thấy CUDA. "
		"Nếu máy có NVIDIA GPU, hãy cài PyTorch bản CUDA trong đúng environment."
	)
	return "cpu"


def format_timestamp(seconds: Optional[float]) -> str:
	"""Convert seconds to SRT timestamp format: HH:MM:SS,mmm."""
	if seconds is None or seconds < 0:
		seconds = 0.0

	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	secs = int(seconds % 60)
	millis = int(round((seconds - int(seconds)) * 1000))

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


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
	"""Write text through a temporary file, then atomically replace the target file."""
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	tmp_path.write_text(content, encoding=encoding)
	tmp_path.replace(path)


def command_to_text(command: Sequence[str]) -> str:
	"""Return a readable command string for logs/errors."""
	return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def run_command(command: Sequence[str], *, hide_output: bool = False) -> None:
	"""Run a subprocess command and preserve useful diagnostics on failure."""
	logging.info("Running command: %s", command_to_text(command))
	started_at = time.perf_counter()
	completed = subprocess.run(
		list(command),
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		encoding="utf-8",
		errors="replace",
	)
	elapsed = time.perf_counter() - started_at

	if completed.returncode == 0:
		logging.info("Command finished in %.1fs", elapsed)
		if not hide_output and completed.stdout.strip():
			logging.debug(completed.stdout.strip())
		if not hide_output and completed.stderr.strip():
			logging.debug(completed.stderr.strip()[-COMMAND_OUTPUT_LIMIT:])
		return

	output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
	if len(output) > COMMAND_OUTPUT_LIMIT:
		output = output[-COMMAND_OUTPUT_LIMIT:]

	raise RuntimeError(
		f"Command failed with exit code {completed.returncode}: {command_to_text(command)}\n"
		f"Last command output:\n{output.strip()}"
	)


def escape_subtitle_path_for_ffmpeg(path: Path) -> str:
	"""Escape subtitle path so FFmpeg subtitles filter can read Windows paths."""
	escaped = str(path.resolve()).replace("\\", "/")
	escaped = escaped.replace(":", r"\:")
	escaped = escaped.replace("'", r"\'")
	return escaped


def resolve_existing_path(path_text: str, fallback_dirs: Sequence[Path]) -> Path:
	"""Resolve a user-provided path, trying fallback folders for relative names."""
	raw_path = Path(path_text).expanduser()
	candidates = [raw_path]

	if not raw_path.is_absolute():
		candidates.extend(folder / raw_path for folder in fallback_dirs)

	for candidate in candidates:
		resolved = candidate.resolve()
		if resolved.exists():
			return resolved

	tried = "\n".join(f"- {candidate.resolve()}" for candidate in candidates)
	raise FileNotFoundError(f"Không tìm thấy file: {path_text}\nĐã thử:\n{tried}")


def resolve_existing_model_path(
	model_text: str,
	paths: ProjectPaths,
	*,
	label: str,
) -> Path:
	"""Resolve a model path, preferring the project model/ folder for relative names."""
	raw_path = Path(model_text).expanduser()
	candidates = [raw_path]

	if not raw_path.is_absolute():
		candidates = [paths.model_dir / raw_path, paths.root / raw_path, raw_path]

	for candidate in candidates:
		resolved = candidate.resolve()
		if resolved.exists():
			logging.info("Using %s model: %s", label, resolved)
			return resolved

	tried = "\n".join(f"- {candidate.resolve()}" for candidate in candidates)
	raise FileNotFoundError(f"Không tìm thấy {label} model: {model_text}\nĐã thử:\n{tried}")


def find_default_faster_whisper_model(paths: ProjectPaths) -> Path:
	"""Find the default faster-whisper model inside project/model/."""
	for folder_name in DEFAULT_ASR_MODEL_CANDIDATES:
		candidate = paths.model_dir / folder_name
		if (candidate / "model.bin").exists():
			logging.info("Using default ASR model from model folder: %s", candidate.resolve())
			return candidate.resolve()

	if paths.model_dir.exists():
		for candidate in sorted(paths.model_dir.iterdir()):
			if candidate.is_dir() and (candidate / "model.bin").exists():
				logging.info("Auto-detected ASR model from model folder: %s", candidate.resolve())
				return candidate.resolve()

	expected = "\n".join(f"- {paths.model_dir / name}" for name in DEFAULT_ASR_MODEL_CANDIDATES)
	raise FileNotFoundError(
		"Không tìm thấy faster-whisper model mặc định trong folder model/.\n"
		"Hãy đặt model vào một trong các folder sau:\n"
		f"{expected}\n"
		"Hoặc truyền đường dẫn thủ công bằng --model."
	)


def resolve_faster_whisper_model_path(model_text: Optional[str], paths: ProjectPaths) -> Path:
	"""Resolve the ASR model path. If omitted, use project/model/ by default."""
	if model_text:
		model_path = resolve_existing_model_path(model_text, paths, label="ASR")
	else:
		model_path = find_default_faster_whisper_model(paths)

	validate_faster_whisper_model(model_path)
	return model_path


def looks_like_huggingface_model_id(text: str) -> bool:
	"""Return True for model ids such as Qwen/Qwen3-TTS-..."""
	return "/" in text and not any(char in text for char in "\\:")


def find_default_qwen_tts_model(paths: ProjectPaths) -> Path:
	"""Find the default Qwen TTS model inside project/model/."""
	for folder_name in DEFAULT_TTS_MODEL_CANDIDATES:
		candidate = paths.model_dir / folder_name
		if candidate.exists() and candidate.is_dir():
			logging.info("Using default TTS model from model folder: %s", candidate.resolve())
			return candidate.resolve()

	if paths.model_dir.exists():
		for candidate in sorted(paths.model_dir.iterdir()):
			name = candidate.name.lower()
			if candidate.is_dir() and "qwen" in name and "tts" in name:
				logging.info("Auto-detected TTS model from model folder: %s", candidate.resolve())
				return candidate.resolve()

	expected = "\n".join(f"- {paths.model_dir / name}" for name in DEFAULT_TTS_MODEL_CANDIDATES)
	raise FileNotFoundError(
		"Không tìm thấy Qwen TTS model mặc định trong folder model/.\n"
		"Hãy đặt model vào một trong các folder sau:\n"
		f"{expected}\n"
		"Hoặc truyền đường dẫn thủ công bằng --tts-model."
	)


def resolve_qwen_tts_model_name(tts_model_text: Optional[str], paths: ProjectPaths) -> str:
	"""Resolve Qwen TTS model. If omitted, use project/model/ by default."""
	if not tts_model_text:
		return str(find_default_qwen_tts_model(paths))

	try:
		return str(resolve_existing_model_path(tts_model_text, paths, label="TTS"))
	except FileNotFoundError:
		# Keep explicit Hugging Face model ids working, but do not use them as the default.
		if looks_like_huggingface_model_id(tts_model_text):
			logging.info("Using explicit Hugging Face TTS model id: %s", tts_model_text)
			return tts_model_text
		raise


def find_videos(input_dir: Path, selected_video: Optional[str] = None) -> List[Path]:
	"""Find videos. selected_video can be a filename in input/ or a direct path."""
	if selected_video:
		selected = Path(selected_video).expanduser()
		candidates = [selected]
		if not selected.is_absolute():
			candidates.insert(0, input_dir / selected_video)

		for video_path in candidates:
			video_path = video_path.resolve()
			if video_path.exists():
				if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
					raise ValueError(f"File không phải định dạng video được hỗ trợ: {video_path.name}")
				return [video_path]

		raise FileNotFoundError(
			f"Không tìm thấy video '{selected_video}'. Đặt file trong {input_dir} "
			"hoặc truyền đường dẫn đầy đủ/relative path hợp lệ."
		)

	videos = sorted(
		path
		for path in input_dir.iterdir()
		if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
	)
	if not videos:
		raise FileNotFoundError(f"Không có video nào trong folder: {input_dir}")
	return videos


def get_single_video_for_burn(input_dir: Path, selected_video: Optional[str]) -> Path:
	"""Burn mode needs exactly one video because one SRT must match one video."""
	videos = find_videos(input_dir, selected_video)
	if len(videos) != 1:
		raise ValueError(
			"--mode burn-srt cần đúng 1 video. "
			"Hãy truyền --video để tránh burn nhầm SRT vào video khác."
		)
	return videos[0]


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


# =========================
# SRT reading/writing + filtering
# =========================


BAD_PHRASES = [
	"subtitles by the amara.org community",
	"subtitle by the amara.org community",
	"subtitles by amara.org community",
	"amara.org community",
	"subscribe to la la school",
	"please subscribe to la la school",
	"la la school channel",
	"thanks for watching",
	"thank you for watching",
	"see you next time",
	"cảm ơn các bạn đã theo dõi",
	"cảm ơn bạn đã theo dõi",
	"hẹn gặp lại",
	"đừng quên đăng ký kênh",
	"nhớ đăng ký kênh",
]


def normalize_text_for_filter(text: str) -> str:
	"""Normalize text for hallucination filtering."""
	text = text.strip().lower()
	text = re.sub(r"\s+", " ", text)
	text = re.sub(r"[^\w\sÀ-ỹ]", "", text)
	return text


def is_bad_hallucination_text(text: str) -> bool:
	"""Remove known hallucinated outro/subtitle-credit phrases."""
	normalized = normalize_text_for_filter(text)
	if not normalized:
		return True

	for phrase in BAD_PHRASES:
		phrase_norm = normalize_text_for_filter(phrase)
		if phrase_norm and phrase_norm in normalized:
			return True

	return False


def remove_repeated_hallucination_segments(
	segments: Iterable[SubtitleSegment],
	max_same_text_count: int = 3,
	short_segment_seconds: float = 4.0,
) -> List[SubtitleSegment]:
	"""Remove suspicious repeated short subtitle segments."""
	cleaned: List[SubtitleSegment] = []
	repeat_count_by_text: dict[str, int] = {}

	for segment in segments:
		text = (segment.text or "").strip()
		normalized = normalize_text_for_filter(text)
		duration = max(0.0, segment.end - segment.start)

		if not normalized:
			continue

		repeat_count_by_text[normalized] = repeat_count_by_text.get(normalized, 0) + 1

		if duration <= short_segment_seconds and repeat_count_by_text[normalized] > max_same_text_count:
			logging.warning(
				"Removed repeated hallucination: %.2f --> %.2f | %s",
				segment.start,
				segment.end,
				text,
			)
			continue

		cleaned.append(segment)

	return cleaned


def fix_too_short_or_invalid_timing(segments: Iterable[SubtitleSegment]) -> List[SubtitleSegment]:
	"""Drop invalid timestamp segments and keep SRT timing safe."""
	cleaned: List[SubtitleSegment] = []
	last_end = 0.0

	for segment in sorted(segments, key=lambda item: item.start):
		if segment.end <= segment.start:
			logging.warning(
				"Removed invalid timing: %.2f --> %.2f | %s",
				segment.start,
				segment.end,
				segment.text,
			)
			continue

		start = max(segment.start, last_end)
		end = max(segment.end, start + 0.2)
		last_end = end
		cleaned.append(SubtitleSegment(start=start, end=end, text=segment.text))

	return cleaned


def post_process_segments(segments: Iterable[SubtitleSegment]) -> List[SubtitleSegment]:
	"""Main subtitle post-processing pipeline for raw Whisper output."""
	filtered: List[SubtitleSegment] = []

	for segment in segments:
		text = (segment.text or "").strip()
		if not text:
			continue

		if is_bad_hallucination_text(text):
			logging.warning(
				"Removed known hallucination: %.2f --> %.2f | %s",
				segment.start,
				segment.end,
				text,
			)
			continue

		filtered.append(SubtitleSegment(segment.start, segment.end, text))

	filtered = remove_repeated_hallucination_segments(filtered)
	filtered = fix_too_short_or_invalid_timing(filtered)
	return filtered


def write_srt(segments: Iterable[SubtitleSegment], srt_path: Path) -> None:
	"""Write subtitle segments to an .srt file after raw Whisper cleanup."""
	segments = post_process_segments(segments)
	blocks: List[str] = []

	for index, segment in enumerate(segments, start=1):
		text = (segment.text or "").strip()
		if not text:
			continue

		blocks.append(
			f"{index}\n"
			f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
			f"{text}"
		)

	atomic_write_text(srt_path, "\n\n".join(blocks) + "\n", encoding="utf-8")
	logging.info("Wrote %d subtitle segment(s): %s", len(blocks), srt_path)


def read_srt(srt_path: Path) -> List[SubtitleSegment]:
	"""Read an existing SRT file into SubtitleSegment objects without changing its text."""
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
		for line_index, line in enumerate(lines):
			if "-->" in line:
				timing_line_index = line_index
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

	if not segments:
		raise ValueError(f"SRT không có subtitle hợp lệ: {srt_path}")

	return segments


# =========================
# Video processing
# =========================


def burn_subtitles(video_in: Path, srt_in: Path, video_out: Path) -> None:
	"""Burn hard subtitles into a video using FFmpeg."""
	if not video_in.exists():
		raise FileNotFoundError(f"Không tìm thấy video để burn subtitle: {video_in}")
	if not srt_in.exists():
		raise FileNotFoundError(f"Không tìm thấy subtitle để burn: {srt_in}")

	video_out.parent.mkdir(parents=True, exist_ok=True)
	ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
	srt_escaped = escape_subtitle_path_for_ffmpeg(srt_in)

	command = [
		ffmpeg_path,
		"-y",
		"-i",
		str(video_in),
		"-vf",
		# f"subtitles='{srt_escaped}':force_style='{DEFAULT_SUBTITLE_STYLE}'",
		f"subtitles='{srt_escaped}':force_style='MarginV=25'",
		"-c:a",
		"copy",
		str(video_out),
	]
	run_command(command)


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
		"-shortest",
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
		"-shortest",
		str(video_out),
	]
	run_command(command)


# =========================
# Transcription
# =========================


def validate_faster_whisper_model(model_path: Path) -> None:
	"""Validate that the ASR model folder is a faster-whisper model."""
	if not model_path.exists():
		raise FileNotFoundError(f"Không tìm thấy model folder: {model_path}")
	if not (model_path / "model.bin").exists():
		raise ValueError(
			f"Không nhận diện được model faster-whisper tại: {model_path}. "
			"Bản này chỉ giữ faster-whisper để workflow gọn và đúng mục tiêu."
		)


def transcribe_video_to_segments(
	video_path: Path,
	model_path: Path,
	language: Optional[str],
	device: str,
	compute_type: str,
) -> List[SubtitleSegment]:
	"""Transcribe video to original-language segments using faster-whisper."""
	from faster_whisper import WhisperModel

	validate_faster_whisper_model(model_path)

	logging.info("Engine: faster-whisper")
	logging.info("ASR model path: %s", model_path)
	logging.info("ASR video input: %s", video_path)
	logging.info("ASR language: %s | device: %s | compute_type: %s", language or "auto", device, compute_type)

	started_at = time.perf_counter()
	logging.info("Loading faster-whisper model...")
	model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
	logging.info("Model loaded in %.1fs", time.perf_counter() - started_at)

	logging.info("Starting transcription with VAD filter...")
	started_at = time.perf_counter()
	segments_iter, info = model.transcribe(
		str(video_path),
		task="transcribe",
		language=language,
		beam_size=5,
		condition_on_previous_text=False,
		vad_filter=True,
		vad_parameters=dict(
			min_silence_duration_ms=500,
			speech_pad_ms=300,
		),
		no_speech_threshold=0.6,
		compression_ratio_threshold=2.4,
		log_prob_threshold=-1.0,
	)

	logging.info(
		"Detected language: %s | probability: %.3f",
		getattr(info, "language", "unknown"),
		float(getattr(info, "language_probability", 0.0) or 0.0),
	)

	output_segments: List[SubtitleSegment] = []
	for index, item in enumerate(segments_iter, start=1):
		text = item.text or ""
		output_segments.append(SubtitleSegment(item.start, item.end, text))
		if index <= 5 or index % 25 == 0:
			logging.info(
				"ASR segment %d | %.2fs -> %.2fs | %s",
				index,
				item.start,
				item.end,
				text.strip()[:100],
			)

	logging.info("Transcription finished: %d segment(s) in %.1fs", len(output_segments), time.perf_counter() - started_at)
	return output_segments


# =========================
# Qwen TTS
# =========================


def load_qwen_tts_model(tts_model_name: str, device: str, attn_implementation: str):
	"""Load Qwen TTS model lazily so the script can still run without TTS."""
	import torch
	from qwen_tts import Qwen3TTSModel

	if device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError(
			"Bạn yêu cầu Qwen TTS chạy GPU nhưng torch.cuda.is_available() = False. "
			"Hãy cài PyTorch bản CUDA đúng môi trường Python, hoặc chạy --device cpu."
		)

	if device == "cuda":
		logging.info("Qwen TTS device: cuda | GPU: %s", torch.cuda.get_device_name(0))
	else:
		logging.info("Qwen TTS device: cpu")

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


def fit_wav_to_available_duration(
	wav,
	sample_rate: int,
	available_duration: float,
	max_speedup: float = 1.15,
	fade_out_seconds: float = 0.04,
):
	"""Fit a generated TTS waveform into an available time slot."""
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
			segment for segment in valid_segments if chunk_start <= segment.start < chunk_end
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
	"""Generate one fixed-time TTS chunk with a safe tail after the chunk boundary."""
	import numpy as np
	import soundfile as sf

	chunk_audio_out.parent.mkdir(parents=True, exist_ok=True)

	gap_seconds = 0.08
	min_slot_seconds = 0.35
	chunk_tail_seconds = max(0.0, float(chunk_tail_seconds))

	global_segments = sorted(
		all_segments if all_segments is not None else chunk_segments,
		key=lambda item: item.start,
	)

	def find_next_global_start(current_segment: SubtitleSegment) -> Optional[float]:
		for candidate in global_segments:
			if candidate.start > current_segment.start + 1e-6:
				return candidate.start
		return None

	sr_final = sample_rate or 24000
	base_chunk_duration = max(0.1, chunk_end - chunk_start)
	chunk_duration_with_tail = max(0.1, chunk_end - chunk_start + chunk_tail_seconds)
	full_chunk = np.zeros(int(chunk_duration_with_tail * sr_final), dtype=np.float32)

	if not chunk_segments:
		silent_chunk = np.zeros(int(base_chunk_duration * sr_final), dtype=np.float32)
		sf.write(str(chunk_audio_out), silent_chunk, sr_final)
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
				full_chunk = np.zeros(int(chunk_duration_with_tail * sr_final), dtype=np.float32)
			elif sr != sr_final:
				raise ValueError(f"Sample rate không đồng nhất: {sr} != {sr_final}")
		elif sr != sr_final:
			raise ValueError(f"Sample rate không đồng nhất: {sr} != {sr_final}")

		next_global_start = find_next_global_start(segment)
		if next_global_start is not None:
			slot_end = next_global_start - gap_seconds
		else:
			slot_end = video_duration if video_duration and video_duration > 0 else segment.end + chunk_tail_seconds

		slot_end = min(slot_end, chunk_end + chunk_tail_seconds)
		available_duration = max(min_slot_seconds, slot_end - segment.start)
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
	"""Rebuild full TTS WAV by placing every chunk at its original timeline position."""
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
	"""Generate TTS by fixed time chunks and rebuild full audio."""
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
		should_generate = overwrite_all_chunks or rerun_chunk == chunk_index or not chunk_path.exists()
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
		chunk_infos=[
			(chunk_path, chunk_start, chunk_end)
			for (_chunk_index, chunk_start, chunk_end, _segments, chunk_path, _should_generate) in chunk_infos
		],
		audio_out=audio_out,
		expected_total_duration=expected_total_duration,
	)


# =========================
# Main workflow
# =========================


def make_vietnamese_srt(
	video_path: Path,
	model_path: Path,
	paths: ProjectPaths,
	language: Optional[str],
	device: str,
	compute_type: str,
	overwrite_srt: bool,
	srt_output: Optional[str],
) -> None:
	"""Step 1: generate raw Vietnamese SRT only."""
	if srt_output:
		srt_path = (paths.root / srt_output).resolve() if not Path(srt_output).is_absolute() else Path(srt_output).resolve()
	else:
		srt_path = paths.subtitle_dir / f"{video_path.stem}_vi_raw.srt"

	logging.info("Processing: %s", video_path.name)
	logging.info("Input video path: %s", video_path)
	logging.info("Output raw Vietnamese SRT: %s", srt_path)

	if srt_path.exists() and not overwrite_srt:
		logging.info("SRT đã tồn tại, bỏ qua transcription: %s", srt_path)
		return

	logging.info("Step 1: Transcribing Vietnamese SRT...")
	segments = transcribe_video_to_segments(
		video_path=video_path,
		model_path=model_path,
		language=language,
		device=device,
		compute_type=compute_type,
	)

	logging.info("Step 2: Writing Vietnamese raw SRT: %s", srt_path)
	write_srt(segments, srt_path)
	logging.info("DONE. Hãy đưa file này cho LLM sửa/dịch, rồi chạy --mode burn-srt với --srt-input.")


def burn_llm_srt_workflow(
	video_path: Path,
	srt_input: Path,
	paths: ProjectPaths,
	enable_tts: bool,
	overwrite_tts: bool,
	rerun_tts_chunk: Optional[int],
	tts_model: str,
	tts_language: str,
	tts_speaker: str,
	tts_instruct: str,
	tts_attn_implementation: str,
	device: str,
	audio_mode: str,
	tts_chunk_minutes: int,
	tts_max_speedup: float,
	tts_chunk_tail_seconds: float,
) -> None:
	"""Step 2: burn the LLM-edited SRT and optionally generate TTS from it."""
	srt_tag = srt_input.stem
	subtitled_output_path = paths.output_dir / f"{video_path.stem}_{srt_tag}_sub.mp4"
	tts_audio_path = paths.audio_dir / f"{video_path.stem}_{srt_tag}_tts.wav"
	tts_chunks_dir = paths.audio_dir / f"{video_path.stem}_{srt_tag}_tts_chunks"
	final_tts_output_path = paths.output_dir / f"{video_path.stem}_{srt_tag}_sub_tts.mp4"

	logging.info("Processing video: %s", video_path.name)
	logging.info("Using LLM-edited SRT: %s", srt_input)

	logging.info("Step 1: Reading SRT...")
	segments = read_srt(srt_input)
	logging.info("Loaded %d subtitle segment(s) from SRT.", len(segments))
	logging.info("Subtitled output path: %s", subtitled_output_path)
	if enable_tts:
		logging.info("TTS WAV path: %s", tts_audio_path)
		logging.info("TTS chunks dir: %s", tts_chunks_dir)
		logging.info("Final TTS video path: %s", final_tts_output_path)

	logging.info("Step 2: Burning LLM-edited SRT to video: %s", subtitled_output_path)
	burn_subtitles(video_path, srt_input, subtitled_output_path)

	if not enable_tts:
		logging.info("TTS disabled. DONE: %s", subtitled_output_path)
		return

	logging.info("Step 3: Generating/rebuilding chunked Qwen TTS from LLM-edited SRT: %s", tts_audio_path)
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

	logging.info("Step 4: Muxing TTS audio into video: %s", final_tts_output_path)
	if audio_mode == "mix":
		mux_audio_into_video_mix(subtitled_output_path, tts_audio_path, final_tts_output_path)
	else:
		mux_audio_into_video_replace(subtitled_output_path, tts_audio_path, final_tts_output_path)

	logging.info("DONE: %s", final_tts_output_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Two-step workflow: generate Vietnamese SRT first, then burn an LLM-edited SRT into video."
	)
	parser.add_argument(
		"--mode",
		choices=["make-srt", "burn-srt"],
		default="make-srt",
		help="make-srt = generate Vietnamese SRT only. burn-srt = use an LLM-edited SRT to burn/TTS.",
	)
	parser.add_argument(
		"--root",
		default=".",
		help="Project root folder. Default: current folder.",
	)
	parser.add_argument(
		"--log-level",
		choices=list(SUPPORTED_LOG_LEVELS),
		default="INFO",
		help="Logging level. Use DEBUG when troubleshooting.",
	)
	parser.add_argument(
		"--video",
		default=None,
		help="Video filename inside input/, or a valid relative/absolute path.",
	)
	parser.add_argument(
		"--model",
		default=None,
		help=(
			"Path to faster-whisper model folder. Used only in --mode make-srt. "
			"Default: auto-detect inside project model/ folder."
		),
	)
	parser.add_argument(
		"--language",
		default="vi",
		help="Source language hint for faster-whisper. Default: vi.",
	)
	parser.add_argument(
		"--device",
		choices=["auto", "cuda", "cpu"],
		default="auto",
		help=(
			"Inference device. auto = prefer GPU when CUDA is available; "
			"cuda = force GPU and raise an error if CUDA is not usable; cpu = force CPU."
		),
	)
	parser.add_argument(
		"--compute-type",
		default="int8",
		help="faster-whisper compute type. Common values: int8, float16, float32.",
	)
	parser.add_argument(
		"--overwrite-srt",
		action="store_true",
		help="Regenerate Vietnamese SRT even if it already exists.",
	)
	parser.add_argument(
		"--srt-output",
		default=None,
		help="Optional output path for raw Vietnamese SRT. Use only with one selected video.",
	)
	parser.add_argument(
		"--srt-input",
		default=None,
		help="LLM-edited SRT path. Required in --mode burn-srt. Can be inside subtitles/.",
	)

	parser.add_argument(
		"--enable-tts",
		action="store_true",
		help="Generate chunked Qwen TTS from --srt-input and mux it into the subtitled video.",
	)
	parser.add_argument(
		"--overwrite-tts",
		action="store_true",
		help="Regenerate all TTS chunks/WAV even if they already exist.",
	)
	parser.add_argument(
		"--rerun-tts-chunk",
		type=int,
		default=None,
		help="Regenerate only one TTS chunk by index, e.g. 3.",
	)
	parser.add_argument(
		"--tts-model",
		default=None,
		help=(
			"Qwen TTS model id or local model folder. "
			"Default: auto-detect inside project model/ folder."
		),
	)
	parser.add_argument(
		"--tts-language",
		default="English",
		help="TTS language for Qwen. Use English if your LLM SRT is English.",
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
		"--tts-chunk-minutes",
		type=int,
		default=5,
		help="Chunk length in minutes for generated TTS review files. Default: 5.",
	)
	parser.add_argument(
		"--tts-max-speedup",
		type=float,
		default=1.15,
		help="Maximum speed-up ratio used to fit a TTS sentence into its timing slot. Default: 1.15.",
	)
	parser.add_argument(
		"--tts-chunk-tail-seconds",
		type=float,
		default=10.0,
		help="Extra safety tail after each chunk boundary, in seconds. Default: 10.",
	)

	return parser.parse_args()


def validate_runtime_options(args: argparse.Namespace, video_count: int) -> None:
	"""Validate CLI options early so failures are clear before long model loading."""
	if args.mode == "make-srt":
		if args.srt_output and video_count != 1:
			raise ValueError("--srt-output chỉ dùng khi bạn chọn đúng 1 video bằng --video.")

	if args.mode == "burn-srt" and not args.srt_input:
		raise ValueError("--mode burn-srt cần --srt-input, tức file SRT đã được LLM sửa/dịch.")

	if args.tts_chunk_minutes <= 0:
		raise ValueError("--tts-chunk-minutes phải lớn hơn 0.")
	if args.rerun_tts_chunk is not None and args.rerun_tts_chunk < 0:
		raise ValueError("--rerun-tts-chunk phải >= 0.")
	if args.tts_max_speedup < 1.0:
		raise ValueError("--tts-max-speedup phải >= 1.0.")
	if args.tts_chunk_tail_seconds < 0:
		raise ValueError("--tts-chunk-tail-seconds phải >= 0.")


def main() -> None:
	args = parse_args()
	setup_logging(args.log_level)

	root = Path(args.root).resolve()
	paths = ProjectPaths.from_root(root)
	paths.create_dirs()

	log_environment_diagnostics(paths, args)

	failed_count = 0

	if args.mode == "make-srt":
		videos = find_videos(paths.input_dir, args.video)
		validate_runtime_options(args, video_count=len(videos))
		model_path = resolve_faster_whisper_model_path(args.model, paths)
		language = args.language.strip() or None
		asr_device = choose_asr_device(args.device)
		logging.info("Found %d video(s).", len(videos))

		for video_path in videos:
			try:
				make_vietnamese_srt(
					video_path=video_path,
					model_path=model_path,
					paths=paths,
					language=language,
					device=asr_device,
					compute_type=args.compute_type,
					overwrite_srt=args.overwrite_srt,
					srt_output=args.srt_output,
				)
			except Exception as exc:
				failed_count += 1
				logging.exception("Failed: %s | Error: %s", video_path.name, exc)

	else:
		video_path = get_single_video_for_burn(paths.input_dir, args.video)
		validate_runtime_options(args, video_count=1)
		srt_input = resolve_existing_path(args.srt_input, [paths.subtitle_dir, paths.root])
		tts_model = resolve_qwen_tts_model_name(args.tts_model, paths) if args.enable_tts else ""
		tts_device = choose_tts_device(args.device) if args.enable_tts else "cpu"

		try:
			burn_llm_srt_workflow(
				video_path=video_path,
				srt_input=srt_input,
				paths=paths,
				enable_tts=args.enable_tts,
				overwrite_tts=args.overwrite_tts,
				rerun_tts_chunk=args.rerun_tts_chunk,
				tts_model=tts_model,
				tts_language=args.tts_language,
				tts_speaker=args.tts_speaker,
				tts_instruct=args.tts_instruct,
				tts_attn_implementation=args.tts_attn_implementation,
				device=tts_device,
				audio_mode=args.audio_mode,
				tts_chunk_minutes=args.tts_chunk_minutes,
				tts_max_speedup=args.tts_max_speedup,
				tts_chunk_tail_seconds=args.tts_chunk_tail_seconds,
			)
		except Exception as exc:
			failed_count += 1
			logging.exception("Failed: %s | Error: %s", video_path.name, exc)

	if failed_count:
		raise SystemExit(f"Finished with {failed_count} failed task(s). Check logs above.")


if __name__ == "__main__":
	main()

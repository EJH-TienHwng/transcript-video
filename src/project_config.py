from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
DEFAULT_MODEL_PATH = "models/faster-whisper-large-v3"
MODEL_FILENAME_SUFFIXES = {"faster-whisper": "faster", "huggingface": "huggingface"}
TRANSLATION_MODEL_FILENAME_SUFFIXES = {"vinai-translate": "vinai"}
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


@dataclass
class SubtitleSegment:
	start: float
	end: float
	text: str


@dataclass
class ProjectPaths:
	root: Path
	input_dir: Path
	subtitle_dir: Path
	audio_dir: Path
	output_dir: Path
	temp_dir: Path

	@classmethod
	def from_root(cls, root: Path) -> "ProjectPaths":
		data_root = root / "data"
		return cls(root, data_root / "input", data_root / "subtitles", data_root / "audio", data_root / "output", data_root / "temp")

	def create_dirs(self) -> None:
		for folder in (self.input_dir, self.subtitle_dir, self.audio_dir, self.output_dir, self.temp_dir):
			folder.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def configure_binary_path(root: Path) -> None:
	bin_dir = str((root / "bin").resolve())
	current_path = os.environ.get("PATH", "")
	if bin_dir not in current_path.split(os.pathsep):
		os.environ["PATH"] = bin_dir + os.pathsep + current_path
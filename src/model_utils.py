from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .project_config import (
    MODEL_FILENAME_SUFFIXES,
    TRANSLATION_MODEL_FILENAME_SUFFIXES,
)


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

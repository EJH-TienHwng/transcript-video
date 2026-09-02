from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import (
    MODEL_FILENAME_SUFFIXES,
    TRANSLATION_MODEL_FILENAME_SUFFIXES,
)


def read_transformers_model_config(model_path: Path) -> dict[str, Any]:
    """Read a local Transformers config file when one is available."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model config: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a JSON object: {config_path}")
    return config


def has_transformers_model_weights(model_path: Path) -> bool:
    """Return whether a local Transformers model folder contains model weights."""
    known_files = (
        "model.safetensors",
        "pytorch_model.bin",
    )
    return any((model_path / name).is_file() for name in known_files) or any(
        path.is_file()
        for pattern in ("model-*.safetensors", "pytorch_model-*.bin")
        for path in model_path.glob(pattern)
    )


def detect_model_type(model_path: Path) -> str:
    """Detect whether the ASR model folder is faster-whisper or Hugging Face Whisper."""
    if (model_path / "model.bin").exists():
        return "faster-whisper"

    config = read_transformers_model_config(model_path)
    if has_transformers_model_weights(model_path) and config.get("model_type") == "whisper":
        return "huggingface"

    if has_transformers_model_weights(model_path) and config.get("model_type") == "mbart":
        raise ValueError(
            f"The model at {model_path} is a text translation model. "
            "Pass it through --translation-model and use a Whisper model for --model."
        )

    raise ValueError(f"Could not detect the model format at: {model_path}")


def detect_translation_model_type(model_path: Path) -> str:
    """Detect a supported local text translation model."""
    config = read_transformers_model_config(model_path)
    if has_transformers_model_weights(model_path) and config.get("model_type") == "mbart":
        return "vinai-translate"

    raise ValueError(f"Could not detect a supported translation model at: {model_path}")


def get_model_filename_suffix(model_path: Path, translation_model_path: Path | None = None) -> str:
    """Return the short model name used as a subtitle filename suffix."""
    suffix = MODEL_FILENAME_SUFFIXES[detect_model_type(model_path)]
    if translation_model_path is not None:
        translation_type = detect_translation_model_type(translation_model_path)
        suffix += f"_{TRANSLATION_MODEL_FILENAME_SUFFIXES[translation_type]}"
    return suffix

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
FASTER_WHISPER_COMPUTE_TYPES = {
    "auto",
    "default",
    "int8",
    "int8_float16",
    "int8_float32",
    "int8_bfloat16",
    "int16",
    "float16",
    "float32",
    "bfloat16",
}
DEFAULT_CONFIG_PATH = Path("configs/transcription.toml")
DEFAULT_MODEL_PATH = "models/faster-whisper-large-v3"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_FILENAME_SUFFIXES = {"faster-whisper": "faster", "huggingface": "huggingface"}
TRANSLATION_MODEL_FILENAME_SUFFIXES = {"vinai-translate": "vinai"}


@dataclass(slots=True)
class SubtitleSegment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class ProjectSettings:
    root: str = "."
    video: str | None = None
    model: str = DEFAULT_MODEL_PATH
    translation_model: str | None = None


@dataclass(slots=True)
class HardwareSettings:
    device: str = "cuda"
    compute_type: str = "int8_float16"
    video_encoder: str = "auto"


@dataclass(slots=True)
class TranscriptionSettings:
    task: str = "transcribe"
    language: str = "vi"
    translation_batch_size: int = 8
    overwrite_srt: bool = False
    skip_burn: bool = False


@dataclass(slots=True)
class TTSSettings:
    enabled: bool = False
    overwrite: bool = False
    mode: str = "timed"
    generation_mode: str = "chunked"
    rerun_chunk: int | None = None
    model: str = DEFAULT_TTS_MODEL
    language: str = "English"
    speaker: str = "Aiden"
    instruct: str = (
        "Speak clearly and professionally in a calm teaching voice. "
        "Use a steady medium pace and neutral tone. "
        "Do not laugh, act, dramatize, or add emotions. "
        "Read the text exactly."
    )
    attn_implementation: str = "auto"
    audio_mode: str = "replace"
    split_audio: bool = True
    chunk_minutes: int = 5
    max_speedup: float = 1.15
    chunk_tail_seconds: float = 10.0
    context_max_sentences: int = 4
    context_max_chars: int = 450
    context_break_seconds: float = 3.0


@dataclass(slots=True)
class RunSettings:
    project: ProjectSettings
    hardware: HardwareSettings
    transcription: TranscriptionSettings
    tts: TTSSettings

    @classmethod
    def defaults(cls) -> RunSettings:
        return cls(ProjectSettings(), HardwareSettings(), TranscriptionSettings(), TTSSettings())


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    input_dir: Path
    subtitle_dir: Path
    audio_dir: Path
    output_dir: Path
    temp_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        data_root = root / "data"
        return cls(
            root=root,
            input_dir=data_root / "input",
            subtitle_dir=data_root / "subtitles",
            audio_dir=data_root / "audio",
            output_dir=data_root / "output",
            temp_dir=data_root / "temp",
        )

    def create_dirs(self) -> None:
        for folder in (
            self.input_dir,
            self.subtitle_dir,
            self.audio_dir,
            self.output_dir,
            self.temp_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)


_SECTION_TYPES = {
    "project": ProjectSettings,
    "hardware": HardwareSettings,
    "transcription": TranscriptionSettings,
    "tts": TTSSettings,
}


def _load_section(
    raw: dict[str, Any], section: str, settings_type: type[Any], base: Any | None = None
) -> Any:
    values = raw.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Config section [{section}] must be a TOML table.")

    allowed = {field.name for field in fields(settings_type)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in [{section}]: {', '.join(unknown)}")

    merged = asdict(base) if base is not None else {}
    merged.update(values)
    try:
        return settings_type(**merged)
    except TypeError as exc:
        raise ValueError(f"Invalid values in config section [{section}].") from exc


def _migrate_legacy_hardware_settings(raw: dict[str, Any]) -> None:
    """Move pre-hardware-section settings without breaking saved run profiles."""
    transcription = raw.get("transcription")
    if not isinstance(transcription, dict):
        return

    hardware = raw.setdefault("hardware", {})
    if not isinstance(hardware, dict):
        return

    for key in ("device", "compute_type"):
        if key not in transcription:
            continue
        if key in hardware:
            raise ValueError(
                f"Config defines '{key}' in both [hardware] and legacy [transcription]."
            )
        hardware[key] = transcription.pop(key)


def load_run_settings(config_path: Path, base: RunSettings | None = None) -> RunSettings:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    try:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Run config is not valid TOML: {config_path}") from exc

    _migrate_legacy_hardware_settings(raw)
    unknown_sections = sorted(set(raw) - set(_SECTION_TYPES))
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {', '.join(unknown_sections)}")

    previous = base or RunSettings.defaults()
    return RunSettings(
        project=_load_section(raw, "project", ProjectSettings, previous.project),
        hardware=_load_section(raw, "hardware", HardwareSettings, previous.hardware),
        transcription=_load_section(
            raw, "transcription", TranscriptionSettings, previous.transcription
        ),
        tts=_load_section(raw, "tts", TTSSettings, previous.tts),
    )


def _toml_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def save_run_settings(settings: RunSettings, config_path: Path) -> Path:
    config_path = config_path.expanduser().resolve()
    lines: list[str] = []

    for section in _SECTION_TYPES:
        lines.append(f"[{section}]")
        for key, value in asdict(getattr(settings, section)).items():
            if value is not None:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return current


def configure_binary_path(root: Path) -> None:
    bin_dir = str((root / "bin").resolve())
    current_path = os.environ.get("PATH", "")
    if bin_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + current_path

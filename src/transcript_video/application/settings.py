from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import (
    DEFAULT_CONFIG_PATH,
    FASTER_WHISPER_COMPUTE_TYPES,
    RunSettings,
    _migrate_legacy_hardware_settings,
    load_run_settings,
)


@dataclass(slots=True)
class ResolvedSettings:
    settings: RunSettings
    sources: dict[str, str]
    config_path: Path | None
    profile_path: Path | None


def resolve_settings(
    *,
    config_path: Path | None = DEFAULT_CONFIG_PATH,
    profile: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    require_config: bool = False,
) -> ResolvedSettings:
    settings = RunSettings.defaults()
    sources = {
        f"{section}.{key}": "default"
        for section, values in asdict(settings).items()
        for key in values
    }
    loaded_config: Path | None = None
    if config_path is not None:
        candidate = config_path.expanduser().resolve()
        if candidate.is_file():
            settings = load_run_settings(candidate, settings)
            loaded_config = candidate
            _mark_file_sources(sources, candidate, "config")
        elif require_config:
            raise FileNotFoundError(f"Run config not found: {candidate}")

    profile_path = _resolve_profile(profile, loaded_config)
    if profile_path is not None:
        settings = load_run_settings(profile_path, settings)
        _mark_file_sources(sources, profile_path, "profile")

    for dotted_name, value in (overrides or {}).items():
        if value is None:
            continue
        section, name = dotted_name.split(".", 1)
        setattr(getattr(settings, section), name, value)
        sources[dotted_name] = "command line"
    validate_settings(settings)
    return ResolvedSettings(settings, sources, loaded_config, profile_path)


def _resolve_profile(profile: str | Path | None, config_path: Path | None) -> Path | None:
    if profile is None:
        return None
    candidate = Path(profile).expanduser()
    if candidate.suffix or candidate.parent != Path("."):
        resolved = candidate.resolve()
    else:
        base = config_path.parent if config_path else Path("configs").resolve()
        resolved = base / "profiles" / f"{candidate.name}.toml"
    if not resolved.is_file():
        raise FileNotFoundError(f"Profile not found: {resolved}")
    return resolved


def _mark_file_sources(sources: dict[str, str], path: Path, source: str) -> None:
    import tomllib

    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    _migrate_legacy_hardware_settings(raw)
    for section, values in raw.items():
        if isinstance(values, dict):
            for key in values:
                sources[f"{section}.{key}"] = source


def validate_settings(settings: RunSettings) -> None:
    settings.subtitle_style.validate()
    defaults = asdict(RunSettings.defaults())
    for section, values in asdict(settings).items():
        for key, value in values.items():
            default = defaults[section][key]
            name = f"{section}.{key}"
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false.")
            if isinstance(default, float) and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number.")
    project, hardware, transcription, tts = (
        settings.project,
        settings.hardware,
        settings.transcription,
        settings.tts,
    )
    strings = {
        "project.root": project.root,
        "project.model": project.model,
        "hardware.compute_type": hardware.compute_type,
        "tts.model": tts.model,
        "tts.language": tts.language,
        "tts.speaker": tts.speaker,
        "tts.instruct": tts.instruct,
    }
    for name, value in strings.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")
    if project.video is not None and not isinstance(project.video, str):
        raise ValueError("project.video must be a string or null.")
    if project.translation_model is not None and (
        not isinstance(project.translation_model, str) or not project.translation_model.strip()
    ):
        raise ValueError("project.translation_model must be a non-empty string or null.")
    if not isinstance(transcription.language, str):
        raise ValueError("transcription.language must be a string.")
    choices = {
        "transcription.task": (transcription.task, {"translate", "transcribe"}),
        "hardware.device": (hardware.device, {"cuda", "cpu"}),
        "hardware.compute_type": (hardware.compute_type, FASTER_WHISPER_COMPUTE_TYPES),
        "hardware.video_encoder": (hardware.video_encoder, {"auto", "h264_nvenc", "libx264"}),
        "tts.mode": (tts.mode, {"timed", "simple"}),
        "tts.generation_mode": (tts.generation_mode, {"chunked", "full"}),
        "tts.attn_implementation": (
            tts.attn_implementation,
            {"auto", "flash_attention_2", "sdpa", "eager"},
        ),
        "tts.audio_mode": (tts.audio_mode, {"replace", "mix"}),
    }
    for name, (value, allowed) in choices.items():
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}.")
    if not isinstance(transcription.translation_batch_size, int) or isinstance(
        transcription.translation_batch_size, bool
    ):
        raise ValueError("transcription.translation_batch_size must be an integer.")
    if transcription.translation_batch_size < 1:
        raise ValueError("transcription.translation_batch_size must be at least 1.")
    if not isinstance(tts.chunk_minutes, int) or isinstance(tts.chunk_minutes, bool):
        raise ValueError("tts.chunk_minutes must be an integer.")
    if not isinstance(tts.context_max_sentences, int) or isinstance(
        tts.context_max_sentences, bool
    ):
        raise ValueError("tts.context_max_sentences must be an integer.")
    if not isinstance(tts.context_max_chars, int) or isinstance(tts.context_max_chars, bool):
        raise ValueError("tts.context_max_chars must be an integer.")
    if not isinstance(tts.max_speedup, int | float) or isinstance(tts.max_speedup, bool):
        raise ValueError("tts.max_speedup must be a number.")
    if not isinstance(tts.chunk_tail_seconds, int | float) or isinstance(
        tts.chunk_tail_seconds, bool
    ):
        raise ValueError("tts.chunk_tail_seconds must be a number.")
    if not isinstance(tts.context_break_seconds, int | float) or isinstance(
        tts.context_break_seconds, bool
    ):
        raise ValueError("tts.context_break_seconds must be a number.")
    for name, minimum in (
        ("chunk_minutes", 1),
        ("max_speedup", 1),
        ("chunk_tail_seconds", 0),
        ("context_max_sentences", 1),
        ("context_max_chars", 1),
        ("context_break_seconds", 0),
    ):
        if getattr(tts, name) < minimum:
            raise ValueError(f"tts.{name} must be at least {minimum}.")
    if tts.rerun_chunk is not None and (
        isinstance(tts.rerun_chunk, bool)
        or not isinstance(tts.rerun_chunk, int)
        or tts.rerun_chunk < 0
    ):
        raise ValueError("tts.rerun_chunk must be an integer at least 0.")
    if tts.verify_final_audio and tts.mode == "simple" and tts.generation_mode == "full":
        raise ValueError("tts.verify_final_audio requires timed or chunked generation.")

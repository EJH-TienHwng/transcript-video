from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import (
    DEFAULT_CONFIG_PATH,
    FASTER_WHISPER_COMPUTE_TYPES,
    HardwareSettings,
    ProjectPaths,
    ProjectSettings,
    RunSettings,
    TranscriptionSettings,
    TTSSettings,
    configure_binary_path,
    load_run_settings,
    save_run_settings,
    setup_logging,
)


def _resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _config_defaults(argv: Sequence[str]) -> tuple[Path, RunSettings]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known, _unknown = parser.parse_known_args(argv)
    config_path = known.config.expanduser()

    if config_path.is_file():
        return config_path, load_run_settings(config_path)
    if "--config" in argv or any(arg.startswith("--config=") for arg in argv):
        raise FileNotFoundError(f"Run config not found: {config_path.resolve()}")
    return config_path, RunSettings.defaults()


def _build_parser(config_path: Path, defaults: RunSettings) -> argparse.ArgumentParser:
    project = defaults.project
    hardware = defaults.hardware
    transcription = defaults.transcription
    tts = defaults.tts

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe or translate videos, burn subtitles, generate Qwen TTS audio, "
            "and mux the result."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help=f"TOML run config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        default=None,
        help="Save the effective settings to a reusable TOML file before running.",
    )
    parser.add_argument("--root", default=project.root, help="Project root folder.")
    parser.add_argument(
        "--video",
        default=project.video,
        help="Process one video inside data/input; omit to process all videos.",
    )
    parser.add_argument(
        "--model",
        default=project.model,
        help="Path to a faster-whisper or Hugging Face Whisper model folder.",
    )
    parser.add_argument(
        "--translation-model",
        default=project.translation_model,
        help="Optional VinAI vi2en model folder; requires --task translate.",
    )
    parser.add_argument(
        "--task",
        choices=["translate", "transcribe"],
        default=transcription.task,
        help="Keep the source language or translate speech directly to English.",
    )
    parser.add_argument(
        "--language",
        default=transcription.language,
        help="Source language hint; use an empty string for auto-detection.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=hardware.device,
        help="Inference device. CUDA is the GPU-accelerated default.",
    )
    parser.add_argument(
        "--compute-type",
        choices=sorted(FASTER_WHISPER_COMPUTE_TYPES),
        default=hardware.compute_type,
        help="faster-whisper compute type, such as int8, float16, or float32.",
    )
    parser.add_argument(
        "--video-encoder",
        choices=["auto", "h264_nvenc", "libx264"],
        default=hardware.video_encoder,
        help="FFmpeg video encoder. 'auto' prefers NVIDIA NVENC and falls back to libx264.",
    )
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=transcription.translation_batch_size,
        help="Number of subtitle segments translated in one batch.",
    )
    parser.add_argument(
        "--overwrite-srt",
        action=argparse.BooleanOptionalAction,
        default=transcription.overwrite_srt,
        help="Regenerate an existing SRT file.",
    )
    parser.add_argument(
        "--skip-burn",
        action=argparse.BooleanOptionalAction,
        default=transcription.skip_burn,
        help="Stop after writing the SRT file.",
    )

    tts_group = parser.add_argument_group("text-to-speech")
    tts_group.add_argument(
        "--enable-tts",
        action=argparse.BooleanOptionalAction,
        default=tts.enabled,
        help="Generate Qwen TTS audio and mux it into the video.",
    )
    tts_group.add_argument(
        "--overwrite-tts",
        action=argparse.BooleanOptionalAction,
        default=tts.overwrite,
        help="Regenerate existing TTS audio.",
    )
    tts_group.add_argument(
        "--tts-mode",
        choices=["timed", "simple"],
        default=tts.mode,
        help="Place speech at subtitle timestamps or generate one continuous voice-over.",
    )
    tts_group.add_argument(
        "--tts-generation-mode",
        choices=["chunked", "full"],
        default=tts.generation_mode,
        help="Generate reviewable time chunks or one full WAV file.",
    )
    tts_group.add_argument(
        "--rerun-tts-chunk",
        type=int,
        default=tts.rerun_chunk,
        help="Regenerate one zero-based chunk index in chunked mode.",
    )
    tts_group.add_argument("--tts-model", default=tts.model, help="Qwen TTS model ID or path.")
    tts_group.add_argument("--tts-language", default=tts.language, help="Qwen TTS language.")
    tts_group.add_argument("--tts-speaker", default=tts.speaker, help="Qwen voice name.")
    tts_group.add_argument("--tts-instruct", default=tts.instruct, help="Voice style instruction.")
    tts_group.add_argument(
        "--tts-attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default=tts.attn_implementation,
        help="Attention implementation used by Qwen TTS.",
    )
    tts_group.add_argument(
        "--audio-mode",
        choices=["replace", "mix"],
        default=tts.audio_mode,
        help="Replace or mix the original audio track.",
    )
    tts_group.add_argument(
        "--split-tts-audio",
        action=argparse.BooleanOptionalAction,
        default=tts.split_audio,
        help="Create review chunks from generated TTS audio.",
    )
    tts_group.add_argument(
        "--tts-chunk-minutes",
        type=int,
        default=tts.chunk_minutes,
        help="Length of each TTS review chunk in minutes.",
    )
    tts_group.add_argument(
        "--tts-max-speedup",
        type=float,
        default=tts.max_speedup,
        help="Maximum speed-up used to fit speech into a subtitle slot.",
    )
    tts_group.add_argument(
        "--tts-chunk-tail-seconds",
        type=float,
        default=tts.chunk_tail_seconds,
        help="Safety tail after each fixed TTS chunk boundary.",
    )
    return parser


def _settings_from_args(args: argparse.Namespace) -> RunSettings:
    return RunSettings(
        project=ProjectSettings(
            root=args.root,
            video=args.video,
            model=args.model,
            translation_model=args.translation_model,
        ),
        hardware=HardwareSettings(
            device=args.device,
            compute_type=args.compute_type,
            video_encoder=args.video_encoder,
        ),
        transcription=TranscriptionSettings(
            task=args.task,
            language=args.language,
            translation_batch_size=args.translation_batch_size,
            overwrite_srt=args.overwrite_srt,
            skip_burn=args.skip_burn,
        ),
        tts=TTSSettings(
            enabled=args.enable_tts,
            overwrite=args.overwrite_tts,
            mode=args.tts_mode,
            generation_mode=args.tts_generation_mode,
            rerun_chunk=args.rerun_tts_chunk,
            model=args.tts_model,
            language=args.tts_language,
            speaker=args.tts_speaker,
            instruct=args.tts_instruct,
            attn_implementation=args.tts_attn_implementation,
            audio_mode=args.audio_mode,
            split_audio=args.split_tts_audio,
            chunk_minutes=args.tts_chunk_minutes,
            max_speedup=args.tts_max_speedup,
            chunk_tail_seconds=args.tts_chunk_tail_seconds,
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, RunSettings]:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    config_path, defaults = _config_defaults(raw_argv)
    _validate_settings(defaults)
    args = _build_parser(config_path, defaults).parse_args(raw_argv)
    settings = _settings_from_args(args)
    _validate_settings(settings)
    return args, settings


def _validate_settings(settings: RunSettings) -> None:
    project = settings.project
    hardware = settings.hardware
    transcription = settings.transcription
    tts = settings.tts

    def require_string(
        value: object,
        name: str,
        *,
        optional: bool = False,
        allow_empty: bool = False,
    ) -> None:
        if optional and value is None:
            return
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            qualifier = "a non-empty string or null" if optional else "a non-empty string"
            if allow_empty:
                qualifier = "a string"
            raise ValueError(f"{name} must be {qualifier}.")

    def require_bool(value: object, name: str) -> None:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")

    def require_int(value: object, name: str, minimum: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")

    def require_number(value: object, name: str, minimum: float) -> None:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number.")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")

    require_string(project.root, "project.root")
    require_string(project.video, "project.video", optional=True)
    require_string(project.model, "project.model")
    require_string(project.translation_model, "project.translation_model", optional=True)
    require_string(transcription.language, "transcription.language", allow_empty=True)
    require_string(hardware.compute_type, "hardware.compute_type")
    require_string(tts.model, "tts.model")
    require_string(tts.language, "tts.language")
    require_string(tts.speaker, "tts.speaker")
    require_string(tts.instruct, "tts.instruct")

    for value, name in (
        (transcription.overwrite_srt, "transcription.overwrite_srt"),
        (transcription.skip_burn, "transcription.skip_burn"),
        (tts.enabled, "tts.enabled"),
        (tts.overwrite, "tts.overwrite"),
        (tts.split_audio, "tts.split_audio"),
    ):
        require_bool(value, name)

    if transcription.task not in {"translate", "transcribe"}:
        raise ValueError("transcription.task must be 'translate' or 'transcribe'.")
    if hardware.device not in {"cuda", "cpu"}:
        raise ValueError("hardware.device must be 'cuda' or 'cpu'.")
    if hardware.compute_type not in FASTER_WHISPER_COMPUTE_TYPES:
        raise ValueError("hardware.compute_type is not supported by faster-whisper.")
    if hardware.video_encoder not in {"auto", "h264_nvenc", "libx264"}:
        raise ValueError("hardware.video_encoder must be 'auto', 'h264_nvenc', or 'libx264'.")
    require_int(transcription.translation_batch_size, "transcription.translation_batch_size", 1)
    if tts.mode not in {"timed", "simple"}:
        raise ValueError("tts.mode must be 'timed' or 'simple'.")
    if tts.generation_mode not in {"chunked", "full"}:
        raise ValueError("tts.generation_mode must be 'chunked' or 'full'.")
    if tts.attn_implementation not in {"auto", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError("tts.attn_implementation is not supported.")
    if tts.audio_mode not in {"replace", "mix"}:
        raise ValueError("tts.audio_mode must be 'replace' or 'mix'.")
    require_int(tts.chunk_minutes, "tts.chunk_minutes", 1)
    if tts.rerun_chunk is not None:
        require_int(tts.rerun_chunk, "tts.rerun_chunk", 0)
    require_number(tts.max_speedup, "tts.max_speedup", 1.0)
    require_number(tts.chunk_tail_seconds, "tts.chunk_tail_seconds", 0.0)


def main() -> None:
    setup_logging()
    args, settings = parse_args()

    if args.save_config is not None:
        saved_path = save_run_settings(settings, args.save_config)
        logging.info("Saved reusable run config: %s", saved_path)

    from .processing.media import find_videos
    from .processing.models import detect_translation_model_type
    from .processing.pipeline import process_video

    project = settings.project
    transcription = settings.transcription
    tts = settings.tts
    root = Path(project.root).expanduser().resolve()
    configure_binary_path(root)
    model_path = _resolve_from_root(root, project.model)
    translation_model_path = (
        _resolve_from_root(root, project.translation_model) if project.translation_model else None
    )

    paths = ProjectPaths.from_root(root)
    paths.create_dirs()

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    if translation_model_path is not None:
        if not translation_model_path.is_dir():
            raise FileNotFoundError(f"Translation model folder not found: {translation_model_path}")
        if transcription.task != "translate":
            raise ValueError("project.translation_model requires transcription.task = 'translate'.")
        detect_translation_model_type(translation_model_path)

    if tts.enabled and transcription.task == "transcribe" and tts.language.lower() == "english":
        logging.warning(
            "TTS is configured for English while transcription keeps the source language. "
            "Use task='translate' when the source video is not already in English."
        )

    videos = find_videos(paths.input_dir, project.video)
    logging.info("Found %d video(s).", len(videos))

    failed_videos = []
    for video_path in videos:
        try:
            process_video(
                video_path=video_path,
                model_path=model_path,
                translation_model_path=translation_model_path,
                paths=paths,
                settings=settings,
            )
        except Exception as exc:
            failed_videos.append(video_path.name)
            logging.exception("Failed: %s | Error: %s", video_path.name, exc)

    if failed_videos:
        raise RuntimeError(
            f"Failed to process {len(failed_videos)}/{len(videos)} video(s): "
            + ", ".join(failed_videos)
        )

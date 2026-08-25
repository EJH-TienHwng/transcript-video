from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .media import find_videos
from .model_utils import detect_translation_model_type
from .pipeline import process_video
from .project_config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TTS_MODEL,
    ProjectPaths,
    configure_binary_path,
    setup_logging,
)


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
		default="transcribe",
		help="transcribe = keep original language. Use this first for Vietnamese SRT. translate = output English subtitles directly.",
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
	parser.add_argument(
		"--tts-chunk-tail-seconds",
		type=float,
		default=10.0,
		help="Extra safety tail after each chunk boundary, in seconds. Prevents losing audio that starts before a chunk boundary and ends after it. Default: 10.",
	)

	return parser.parse_args()


def main() -> None:
	setup_logging()
	args = parse_args()

	root = Path(args.root).resolve()
	configure_binary_path(root)
	model_path = (root / args.model if not Path(args.model).expanduser().is_absolute() else Path(args.model)).expanduser().resolve()
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
	if args.tts_chunk_tail_seconds < 0:
		raise ValueError("--tts-chunk-tail-seconds phải >= 0.")

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
				tts_chunk_tail_seconds=args.tts_chunk_tail_seconds,
			)
		except Exception as exc:
			logging.exception("Failed: %s | Error: %s", video_path.name, exc)

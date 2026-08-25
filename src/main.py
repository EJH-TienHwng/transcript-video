"""
Professional video transcription + subtitle + TTS voice-over pipeline.

Boundary-tail fixed version: chunk buffers keep their tail even after sample_rate is known.

Folder structure:
	project/
	├── data/audio/       # generated .wav TTS audio files and 5-minute review chunks
	├── data/input/       # put original videos here
	├── data/output/      # final videos
	├── data/subtitles/   # generated .srt files
	└── data/temp/        # temporary extracted audio

Basic usage:
	python -m src.main
	python -m src.main --video my_video.mp4

Generate clean Vietnamese subtitles first, then send the SRT to an LLM for translation:
	python -m src.main --video my_video.mp4 --task transcribe --language vi --skip-burn

Generate subtitles + Qwen TTS audio + final video with TTS audio:
	python -m src.main --video my_video.mp4 --task translate --language vi --enable-tts

By default, when TTS is enabled, the generated TTS WAV is also split into 5-minute chunks:
	data/audio/<video_name>_tts_chunks/<video_name>_tts_part_000.wav

Use a local Whisper model folder:
	python -m src.main --model "C:/Users/<USERNAME>/.faster-whisper-large-v3"

Install needed packages:
	pip install imageio-ffmpeg faster-whisper
	pip install -U qwen-tts soundfile numpy

Optional for Hugging Face Whisper:
	pip install transformers torch accelerate safetensors
"""

from .cli import main


if __name__ == "__main__":
    main()


# python transcript_video_old.py --video "Report.mp4" --model "C:\Users\AHG5HC\.faster-whisper-large-v3" --task transcribe --language vi --enable-tts --tts-model "C:\Users\AHG5HC\Documents\Code\transcript_video_eng\model\Qwen3-TTS-12Hz-1.7B-CustomVoice" --tts-language English --tts-speaker Aiden --tts-generation-mode chunked --audio-mode replace --tts-attn-implementation sdpa
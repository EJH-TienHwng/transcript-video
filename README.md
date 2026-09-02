# Transcript Video

Local tools for video transcription, subtitle rendering, Qwen TTS voice-over, and compiling multiple processed videos into a training course.

## Features

- Transcribe with local faster-whisper or Hugging Face Whisper models.
- Translate speech directly with Whisper or translate Vietnamese subtitle text with VinAI.
- Write and reuse SRT files.
- Burn subtitles with the FFmpeg binary supplied by `imageio-ffmpeg`.
- Generate full or fixed-time Qwen TTS audio.
- Replace or mix the original audio track.
- Build a course video with a table of contents, session cards, and MP4 chapters.
- Store repeatable run settings in TOML instead of maintaining long shell commands.

## Project layout

```text
transcript-video/
├── configs/
│   ├── transcription.toml       # default reusable processing settings
│   └── courses/                 # course-builder JSON configurations
├── data/
│   ├── input/                   # source videos
│   ├── subtitles/               # generated or manually edited SRT files
│   ├── audio/                   # generated TTS audio and review chunks
│   ├── output/                  # rendered videos
│   ├── temp/                    # temporary extracted audio
│   └── compilation/             # course-builder working files and outputs
├── docs/                        # subtitle-editing prompts and documentation
├── scripts/                     # manual smoke-test utilities
├── src/transcript_video/
│   ├── cli.py                   # primary command-line interface
│   ├── config.py                # reusable TOML settings and shared domain types
│   ├── processing/              # transcription, subtitles, media, models, and TTS
│   └── course/                  # course configuration, rendering, timeline, and TUI
├── tests/                       # dependency-light automated tests
├── pyproject.toml               # package metadata and direct dependencies
├── ruff.toml                    # lint and format policy
└── uv.lock                      # fully resolved reproducible environment
```

The package uses the standard Python `src` layout. `src` is a container directory; the importable package is `transcript_video`.

## Requirements

- Windows
- Python 3.12
- An NVIDIA GPU is recommended for long videos
- Local Whisper model files under `models/` or an explicit model path

The locked PyTorch packages target CUDA 12.4.

## Install with uv

Install uv if it is not already available:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create the Python 3.12 environment and install the locked dependencies:

```powershell
uv sync
```

`uv` reads `.python-version`, downloads Python 3.12 when necessary, creates `.venv`, and installs from `uv.lock`.

## Reusable run configuration

The main command automatically loads [configs/transcription.toml](configs/transcription.toml). Edit that file once, then run:

```powershell
uv run transcript-video
```

The file contains three sections:

- `[project]`: root directory, selected video, ASR model, and optional translation model.
- `[transcription]`: task, language, device, compute type, batch size, and subtitle behavior.
- `[tts]`: voice, model, generation mode, timing, chunking, and audio behavior.

Use another saved configuration:

```powershell
uv run transcript-video --config configs/my-run.toml
```

CLI arguments override values loaded from TOML:

```powershell
uv run transcript-video --config configs/my-run.toml --video lesson-02.mp4 --device cpu
```

Save the effective combination of config values and CLI overrides:

```powershell
uv run transcript-video `
  --video lesson-01.mp4 `
  --task translate `
  --enable-tts `
  --tts-speaker Aiden `
  --save-config configs/english-dub.toml
```

Boolean options support both forms, for example `--enable-tts` and `--no-enable-tts`, or `--split-tts-audio` and `--no-split-tts-audio`.

## Model layout

```text
models/
├── faster-whisper-large-v3/
│   ├── config.json
│   └── model.bin
├── whisper-large-v3/
│   ├── config.json
│   └── model.safetensors
└── vinai-translate-vi2en-v2/
    ├── config.json
    └── pytorch_model.bin
```

Single-file and sharded Transformers weights are supported. Model artifacts are intentionally excluded from Git.

## Common commands

Process every supported video in `data/input` using the default TOML settings:

```powershell
uv run transcript-video
```

Generate only an SRT:

```powershell
uv run transcript-video --video lesson.mp4 --skip-burn
```

Transcribe Vietnamese, translate with VinAI, and generate English TTS:

```powershell
uv run transcript-video `
  --video lesson.mp4 `
  --task translate `
  --translation-model models/vinai-translate-vi2en-v2 `
  --enable-tts
```

Show every available override:

```powershell
uv run transcript-video --help
```

## Course builder

Create a course configuration interactively:

```powershell
uv run transcript-course-config
```

Build a course from an existing JSON configuration:

```powershell
uv run transcript-course --config configs/courses/training_course.json
```

Relative paths in course configurations are resolved from the repository root discovered through `pyproject.toml`.

## Quality checks

Ruff is the project's formatter and linter. It is locked in the `dev` dependency group.

```powershell
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
```

Run the automated tests:

```powershell
uv run python -m unittest discover -s tests -v
```

For a lightweight quality-only environment without the AI/runtime stack:

```powershell
uv sync --only-dev
$env:PYTHONPATH = "src"
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python -m unittest discover -s tests -v
```

## Outputs

| Artifact | Path |
| --- | --- |
| Subtitle | `data/subtitles/<video>_<model>.srt` |
| Hard-subtitled video | `data/output/<video>_vi-dub_en-sub.mp4` |
| TTS audio | `data/audio/<video>_tts.wav` |
| TTS video | `data/output/<video>_en-dub_en-sub.mp4` |

Existing SRT and TTS artifacts are reused unless the corresponding overwrite setting is enabled.

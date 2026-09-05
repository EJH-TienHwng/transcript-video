# Transcript Video Documentation

[English](README.md) · [Tiếng Việt](README.vi.md)

Transcript Video is a local, GPU-first pipeline for speech transcription, optional Vietnamese-to-English translation, subtitle rendering, Qwen TTS voice-over, and course-video compilation.

> This English guide is the canonical documentation. The Vietnamese version is available through the language link above.

## Command overview

```text
transcript-video
├── process [VIDEO]
├── inspect VIDEO
├── doctor
├── config show|validate
└── course create|build|tui
```

Global options must appear before the command: `-q`, `-v`, `-vv`, `--no-color`, `--log-file PATH`, and `--json`. Typer also provides `--install-completion` and `--show-completion`.

The default terminal log is concise. `-v` adds diagnostics and `-vv` adds source locations and tracebacks. Detailed DEBUG logs always rotate at `logs/transcript-video.log` unless `--log-file` overrides the path. `NO_COLOR` and `--no-color` are both honored.

## Everyday workflows

```powershell
uv run transcript-video process
uv run transcript-video process lesson.mp4 --profile gpu-tts
uv run transcript-video process lesson.mp4 --dry-run
uv run transcript-video process lesson.mp4 --force transcription --force tts
uv run transcript-video doctor
uv run transcript-video inspect data/input/lesson.mp4
uv run transcript-video config show --sources
uv run transcript-video config validate --profile gpu-tts
```

Dry-run performs configuration, input-path, and execution-plan validation, but does not create directories, save config, load AI models, or start FFmpeg encoding. `--force` is repeatable and accepts `transcription`, `translation`, `tts`, and `render`; the older overwrite flags remain supported.

Profiles are partial TOML files in `configs/profiles/<name>.toml` (or an explicit TOML path). Effective values resolve in this order: defaults, base config, profile, then command-line overrides.

Read-only commands support machine output by placing `--json` before the command:

```powershell
uv run transcript-video --json inspect data/input/lesson.mp4
uv run transcript-video --json doctor
```

Exit codes are `0` for success, `1` for a runtime/readiness failure, `2` for invalid CLI usage, and `130` for user cancellation.

## Course wizard and full TUI

`transcript-video course create` launches the lightweight Questionary wizard. Its review loop can edit titles and numbers, reorder sessions, remove sessions, and go back without restarting.

`transcript-video course tui` launches the Textual application. It provides Course Metadata, Session Editor, and Review/Build screens. Video metadata and course builds run in background workers. Keyboard shortcuts include `Ctrl+S` to save, `Esc` to go back, `A/E/Delete` to add/edit/remove, `U/D` to reorder, and `Q` to quit. Unsaved changes require confirmation.

## Architecture

Presentation code lives in `cli.py`, `ui/`, `course/wizard.py`, and `tui/`. Application services in `application/` resolve settings, diagnostics, and inspection. `events.py` defines UI-independent pipeline stages and observers. `process_runner.py` owns subprocess execution, ffprobe JSON, FFmpeg `-progress pipe:1` parsing, failure diagnostics, and cancellation. Processing and course modules contain the media/model business logic.

## Developer workflow

```powershell
uv sync
just format
just lint
just test
just check
uv run pre-commit install
```

Pytest markers are `integration`, `gpu`, and `slow`; `just test-fast` excludes all three. Ruff is both formatter and linter. The pre-commit configuration runs Ruff plus TOML/YAML and whitespace checks.

## Contents

- [Transcript Video Documentation](#transcript-video-documentation)
  - [Command overview](#command-overview)
  - [Everyday workflows](#everyday-workflows)
  - [Course wizard and full TUI](#course-wizard-and-full-tui)
  - [Architecture](#architecture)
  - [Developer workflow](#developer-workflow)
  - [Contents](#contents)
  - [Capabilities](#capabilities)
  - [Project structure](#project-structure)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [GPU acceleration](#gpu-acceleration)
    - [Workload map](#workload-map)
  - [Configuration](#configuration)
  - [Transcription workflow](#transcription-workflow)
  - [TTS workflow](#tts-workflow)
  - [Course builder](#course-builder)
  - [Outputs](#outputs)
  - [Quality checks](#quality-checks)
  - [Troubleshooting](#troubleshooting)
    - [CUDA is unavailable](#cuda-is-unavailable)
    - [NVENC falls back to libx264](#nvenc-falls-back-to-libx264)
    - [CUDA out of memory](#cuda-out-of-memory)
    - [CPU activity is still visible](#cpu-activity-is-still-visible)
    - [Existing artifacts are unexpectedly reused](#existing-artifacts-are-unexpectedly-reused)

## Capabilities

- Local ASR with faster-whisper or Hugging Face Whisper.
- Direct Whisper translation or a separate VinAI Vietnamese-to-English translation stage.
- SRT generation, validation, cleanup, and reuse.
- Hard-subtitle rendering with FFmpeg.
- Qwen3-TTS simple, timed, full, and reviewable fixed-chunk generation modes.
- Original-audio replacement or mixing.
- Course compilation with TOC pages, session cards, normalized videos, and MP4 chapters.
- Reusable TOML run profiles and JSON course definitions.
- CUDA inference and automatic NVIDIA NVENC video encoding when available.

## Project structure

```text
transcript-video/
├── assets/                         # course card/TOC images
├── configs/
│   ├── transcription.toml          # default processing profile
│   └── courses/                    # course-builder JSON profiles
├── data/
│   ├── input/                      # source videos
│   ├── subtitles/                  # SRT files
│   ├── audio/                      # TTS WAV files and review chunks
│   ├── output/                     # rendered videos
│   ├── temp/                       # temporary ASR audio
│   └── compilation/                # course work files and final courses
├── docs/                           # English/Vietnamese docs and editing prompts
├── scripts/                        # manual smoke-test utilities
├── src/transcript_video/
│   ├── cli.py                      # main CLI
│   ├── application/                # settings, inspection, diagnostics
│   ├── ui/                         # Rich console, logging, progress
│   ├── tui/                        # full Textual course application
│   ├── events.py                   # UI-independent pipeline events
│   ├── process_runner.py           # subprocess/FFmpeg/ffprobe boundary
│   ├── processing/                 # ASR, translation, subtitles, media, TTS
│   └── course/                     # course builder, cards, timeline, wizard
├── tests/                          # dependency-light automated tests
├── pyproject.toml                  # package metadata and direct dependencies
├── ruff.toml                       # lint/format policy
├── justfile                        # developer task runner
├── .pre-commit-config.yaml         # local commit checks
└── uv.lock                         # reproducible dependency lock
```

## Installation

### Requirements

- Windows and Python 3.12.
- An NVIDIA GPU is strongly recommended.
- A recent NVIDIA driver compatible with the locked CUDA 12.4 PyTorch wheels.
- Local Whisper model files and, when used, local VinAI/Qwen model files.

Install uv:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create `.venv` and install the exact locked environment:

```powershell
uv sync
```

Verify the CLI:

```powershell
uv run transcript-video --help
uv run transcript-video course --help
uv run transcript-video course create --help
```

## GPU acceleration

The default profile is GPU-first:

```toml
[hardware]
device = "cuda"
compute_type = "int8_float16"
video_encoder = "auto"
```

### Workload map

| Workload | Preferred execution | Notes |
| --- | --- | --- |
| faster-whisper ASR | CUDA INT8/FP16 | `compute_type = "int8_float16"` keeps non-quantized work in FP16 while reducing model memory. |
| Hugging Face Whisper | CUDA FP16 | The Transformers pipeline is placed on GPU 0. |
| VinAI translation | CUDA FP16 | Model weights and token batches are moved to CUDA. |
| Qwen3-TTS | CUDA FP16 | Model uses `device_map="cuda:0"`. |
| Subtitle/course H.264 encoding | NVIDIA NVENC | `auto` probes `h264_nvenc` with a real one-frame encode. |
| Subtitle/libass filter | CPU | FFmpeg's standard subtitle renderer is CPU-only. NVENC still handles final H.264 encoding. |
| Scale/pad/fps filters | CPU | These filters feed NVENC; GPU encoding removes the largest avoidable CPU encode load. |
| AAC, WAV, NumPy timing/mixing | CPU | These operations are lightweight or incur more transfer overhead when moved to GPU. |
| Pillow course-card drawing | CPU | Image generation is small compared with video encoding. |
| Stream copy/mux/chapter metadata | Neither compute-heavy | Encoded packets are copied without video re-encoding. |

`video_encoder = "auto"` is recommended. At first use, the project asks FFmpeg to perform a tiny NVENC encode. It selects `h264_nvenc` only when the encoder, NVIDIA driver, and physical GPU path all work. Otherwise it logs the reason and uses `libx264`.

FFmpeg selection order is `TRANSCRIPT_VIDEO_FFMPEG`, then `ffmpeg` from `PATH` (including `bin/ffmpeg.exe` after project startup), then the `imageio-ffmpeg` bundled binary. The bundled binary may not contain NVENC. To enable GPU video encoding, place an NVENC-capable Windows FFmpeg build at `bin/ffmpeg.exe`, add it to `PATH`, or set:

```powershell
$env:TRANSCRIPT_VIDEO_FFMPEG = "C:\path\to\ffmpeg.exe"
```

Available policies:

- `auto`: prefer NVENC, safely fall back to libx264.
- `h264_nvenc`: request NVENC; still fall back if the runtime probe fails.
- `libx264`: explicitly use CPU encoding.

You can override the profile for one run:

```powershell
uv run transcript-video --device cuda --compute-type int8_float16 --video-encoder h264_nvenc
```

To confirm GPU activity during a real job:

```powershell
nvidia-smi -l 1
```

CUDA fallback protects portability, but a fallback warning means the AI workload is running much more slowly. Check the troubleshooting section instead of accepting it for production workloads.

## Configuration

The main command automatically loads [the default profile](../configs/transcription.toml). Relative paths are resolved from `project.root`.

```toml
[project]
root = "."
model = "models/faster-whisper-large-v3"
# video = "lesson-01.mp4"
# translation_model = "models/vinai-translate-vi2en-v2"

[hardware]
device = "cuda"
compute_type = "int8_float16"
video_encoder = "auto"

[transcription]
task = "transcribe"
language = "vi"
translation_batch_size = 8
overwrite_srt = false
skip_burn = false

[tts]
enabled = false
overwrite = false
mode = "timed"
generation_mode = "chunked"
model = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
language = "English"
speaker = "Aiden"
instruct = "Speak clearly and professionally..."
attn_implementation = "auto"
audio_mode = "replace"
split_audio = true
chunk_minutes = 5
max_speedup = 1.15
chunk_tail_seconds = 10.0
context_max_sentences = 4
context_max_chars = 450
context_break_seconds = 3.0
```

CLI values override TOML values without modifying the file:

```powershell
uv run transcript-video --video lesson-02.mp4 --task translate --enable-tts
```

Save the effective profile, including overrides:

```powershell
uv run transcript-video `
  --video lesson-02.mp4 `
  --task translate `
  --enable-tts `
  --save-config configs/lesson-02.toml
```

Reuse it later:

```powershell
uv run transcript-video --config configs/lesson-02.toml
```

## Transcription workflow

Place videos in `data/input`, set a valid model path, then run:

```powershell
uv run transcript-video
```

Process only one input video:

```powershell
uv run transcript-video --video lesson.mp4
```

Generate/reuse the SRT without rendering video:

```powershell
uv run transcript-video --video lesson.mp4 --skip-burn
```

Regenerate an existing SRT:

```powershell
uv run transcript-video --video lesson.mp4 --overwrite-srt
```

Use an empty language string for Whisper auto-detection:

```powershell
uv run transcript-video --language ""
```

For separate VinAI translation, Whisper first transcribes the source language, then VinAI translates subtitle batches:

```powershell
uv run transcript-video `
  --video lesson.mp4 `
  --task translate `
  --translation-model models/vinai-translate-vi2en-v2
```

Existing SRT files are reused unless `overwrite_srt` is enabled.

## TTS workflow

Enable English voice-over:

```powershell
uv run transcript-video --video lesson.mp4 --task translate --enable-tts
```

The recommended `chunked` mode creates fixed review windows, preserves a configurable safety tail at chunk boundaries, and reconstructs the final WAV by timeline overlay. Regenerate one zero-based chunk after reviewing it:

```powershell
uv run transcript-video --rerun-tts-chunk 3
```

Relevant options:

- `tts.mode = "timed"`: place each generated line at its subtitle time.
- `tts.mode = "simple"`: generate a continuous voice-over.
- `tts.generation_mode = "chunked"`: generate reusable review chunks.
- `tts.generation_mode = "full"`: generate the complete track in one pass.
- `tts.audio_mode = "replace"`: replace source audio.
- `tts.audio_mode = "mix"`: mix source and TTS audio.
- `tts.max_speedup`: maximum pitch-preserving FFmpeg `atempo` speed-up used to fit a line.
- `tts.chunk_tail_seconds`: extra boundary room for lines near a chunk end.
- `tts.context_max_sentences` / `context_max_chars`: bound each contextual Qwen generation.
- `tts.context_break_seconds`: only gaps larger than this force a new acoustic context.

Timed TTS aligns each contextual generation with the configured faster-whisper model, then
places each extracted sentence at its original SRT start. Problems requiring manual inspection
are written to `data/audio/<video>_tts_review.jsonl`; speech is never silently truncated.

Run the isolated manual TTS check:

```powershell
uv run python scripts/tts_smoke_test.py --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

## Course builder

Create a JSON profile interactively:

```powershell
uv run transcript-video course create
```

Build the course:

```powershell
uv run transcript-video course build --config configs/courses/training_course.json
```

Each course profile supports:

- ordered session videos;
- course and session titles;
- theme image and optional custom font;
- TOC pagination and duration;
- resolution, frame rate, bitrates, and audio sample rate;
- `render.video_encoder` with `auto`, `h264_nvenc`, or `libx264`;
- optional MP4 chapter metadata.

All relative JSON paths are resolved from the repository root. The default `auto` video encoder also applies to TOC cards and normalized session videos.

## Outputs

| Artifact | Default location |
| --- | --- |
| SRT | `data/subtitles/<video>_<model>.srt` |
| Hard-subtitled video | `data/output/<video>_vi-dub_en-sub.mp4` |
| Full TTS WAV | `data/audio/<video>_tts.wav` |
| TTS review chunks | `data/audio/<video>_tts_chunks/` |
| TTS timing/alignment review log | `data/audio/<video>_tts_review.jsonl` |
| Final TTS video | `data/output/<video>_en-dub_en-sub.mp4` |
| Course work/final files | `data/compilation/` |

## Quality checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

Apply formatting:

```powershell
uv run ruff format .
uv run ruff check --fix .
```

VS Code is configured to use the `.venv` interpreter and Ruff format-on-save.

## Troubleshooting

### CUDA is unavailable

Run:

```powershell
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If `nvidia-smi` works but PyTorch reports `False`, recreate the locked environment with `uv sync` and verify that another Python environment is not active.

### NVENC falls back to libx264

The runtime probe failed. Common causes are an old NVIDIA driver, an FFmpeg build without NVENC (including some `imageio-ffmpeg` binaries), unavailable GPU resources, or execution through a non-NVIDIA machine. Install an NVENC-capable FFmpeg in `bin/ffmpeg.exe` or select it with `TRANSCRIPT_VIDEO_FFMPEG`. `auto` keeps the job working with CPU encode. The AI stages can still use CUDA independently.

### CUDA out of memory

- Close other GPU-heavy programs.
- Reduce `translation_batch_size`.
- Keep `compute_type = "int8_float16"` for lower faster-whisper VRAM use.
- Generate TTS in chunked mode.
- Use a smaller model when available.

The GTX 1650 Ti has limited VRAM, so loading Whisper, VinAI, and Qwen concurrently should be avoided. The pipeline loads them stage by stage rather than intentionally keeping every model resident.

### CPU activity is still visible

This is expected. Media decoding, libass subtitle rasterization, FFmpeg filters, AAC audio, Pillow card drawing, file I/O, and NumPy waveform assembly still use CPU. The expensive model inference and supported H.264 encoding paths are the parts assigned to GPU.

### Existing artifacts are unexpectedly reused

Use `--overwrite-srt` or `--overwrite-tts`. For chunked TTS, use `--rerun-tts-chunk INDEX` to regenerate only the required chunk.

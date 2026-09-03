# Transcript Video

Local, GPU-first tools for transcription, subtitle rendering, Qwen TTS voice-over, and compiling processed sessions into a training course.

**Documentation:** [English](docs/README.md) · [Tiếng Việt](docs/README.vi.md)

## Highlights

- One Typer executable with Rich output: `transcript-video`.
- Reusable TOML configuration and profiles instead of long repeated commands.
- Real FFmpeg machine-readable progress support and a shared subprocess runner.
- Questionary course wizard plus a full three-screen Textual editor.
- Environment diagnostics, media inspection, dry runs, rotating file logs, shell completion, and JSON output.
- CUDA inference and automatic NVIDIA NVENC selection with a tested CPU encoder fallback.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. The locked PyTorch build targets CUDA 12.4.

```powershell
uv sync
uv run transcript-video doctor
```

Place source videos in `data/input` and local model files under `models`, or configure absolute paths.

## Basic CLI

```powershell
# Process every configured video
uv run transcript-video process

# Process one video and translate its speech
uv run transcript-video process lesson.mp4 --task translate

# Inspect without loading a model or encoding
uv run transcript-video inspect data/input/lesson.mp4

# Validate and show the effective config with its sources
uv run transcript-video config validate
uv run transcript-video config show --sources

# Print the complete plan without writing files
uv run transcript-video process lesson.mp4 --dry-run
```

The pre-0.3 form (`transcript-video --video lesson.mp4 ...`) remains accepted. The old `transcript-course` and `transcript-course-config` executables are deprecated wrappers.

## Course tools

```powershell
# Guided Questionary flow
uv run transcript-video course create

# Full Textual application
uv run transcript-video course tui

# Non-interactive build
uv run transcript-video course build --config configs/courses/training_course.json
```

## Configuration and profiles

The default settings live in [`configs/transcription.toml`](configs/transcription.toml). Named profiles live in `configs/profiles/<name>.toml`; resolution order is `defaults < config < profile < CLI`.

```powershell
uv run transcript-video process lesson.mp4 --profile gpu-tts
uv run transcript-video process lesson.mp4 --force transcription --force render
```

## Developer workflow

```powershell
just install
just format
just check
uv run pre-commit install
```

Without `just`, use `uv run ruff format .`, `uv run ruff check .`, and `uv run pytest`.

See the [full English guide](docs/README.md) for logging, profiles, completion, output names, GPU setup, architecture, and troubleshooting.

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

Use the Python version pinned in [`.python-version`](.python-version) (currently 3.14) and [uv](https://docs.astral.sh/uv/). The configured PyTorch build targets CUDA 13.2 and uses local wheels from `wheels/`.

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

# Process an explicit list in the given order
uv run transcript-video process lesson1.mp4 lesson2.mp4 lesson3.mp4

# Inspect without loading a model or encoding
uv run transcript-video inspect data/input/lesson.mp4

# Validate and show the effective config with its sources
uv run transcript-video config validate
uv run transcript-video config show --sources

# Print the complete plan without writing files
uv run transcript-video process lesson.mp4 --dry-run
```

Positional inputs accept names inside `data/input/` or absolute paths. Repeated paths are processed once, keeping the first occurrence. Distinct inputs with the same filename stem are rejected to prevent output/cache collisions. With no positional inputs, `project.video` is used when configured; otherwise `data/input/` is scanned.

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

The wizard offers batch selection with cached video metadata, automatic titles/numbers,
Recommended or Custom settings, and a final review before saving. Use Space to select videos;
checklist order becomes the initial course order, then use Move up/down in the review.
Remove and overwrite require confirmation. Metadata failure does not block setup.

## Progress and diagnostics

Status comes from semantic pipeline events; diagnostic logs are separate. Interactive terminals
show overall/video/stage progress; redirected output automatically uses throttled lines.

| Mode | Terminal output |
| --- | --- |
| Default | Progress, important warnings, TTS quality summary, result |
| `-q` | Warnings/errors and final result; no live progress |
| `-v` | Normal UI plus INFO diagnostics/recovery details |
| `-vv` | DEBUG commands/source locations and failure tracebacks, without locals |
| `--plain` | Line-based status even in an interactive terminal |

Global options go **before** the command. `NO_COLOR` and `--no-color` apply to the CLI and wizard;
terminals that cannot encode the status symbols use ASCII equivalents.

```powershell
uv run transcript-video --plain process one.mp4 two.mp4 --events-json logs/events-batch.jsonl
uv run transcript-video -q process lesson.mp4
uv run transcript-video -vv process lesson.mp4
uv run transcript-video course build --config configs/courses/training_course.json --events-json logs/events-course.jsonl
```

`--events-json` creates a **new** JSONL file; an existing destination is rejected. It never mixes
human output into that file. Dry-run writes neither event files nor diagnostic logs.
`--json` retains its separate semantics for read-only commands.

The global log rotates at 10 MiB with five backups. Process, course build/create and TUI
invocations also have a run ID and individual DEBUG text/JSONL logs:

```text
logs/transcript-video.log
logs/runs/<timestamp>-<random>_process.log
logs/runs/<timestamp>-<random>_process.jsonl
```

`--log-file PATH` still selects the **global rotating log**. Run files go in `PATH.parent/runs/`.
Run logs are not automatically pruned; keep or remove them according to your retention needs.
See [event schema and architecture](docs/observability.md) for integration details and limitations.

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

## TTS review reports

WAVs stay in `data/audio/`. Machine-readable JSONL and indented JSON reports live together:

```text
data/report/tts/<video>_tts_review.jsonl
data/report/tts/<video>_tts_review.pretty.json
data/report/tts/<video>_tts_chunks/<video>_tts_chunk_000.review.jsonl
data/report/tts/<video>_tts_chunks/<video>_tts_chunk_000.review.pretty.json
```

JSONL keeps one object per line; read `.pretty.json` for manual inspection. Chunk cache validation,
rebuild and reruns use the JSONL in the report directory. Old metadata alongside audio is left
untouched and triggers one safe regeneration; new caches are reused normally. `--force tts`
regenerates every chunk. Direct Python TTS calls default to the current project report directory;
pass `review_log_path` (or `review_path` for a single chunk) for a different project.

TTS protects a 180 ms tail budget and the next sentence's onset, regenerating unsafe boundaries.
Placement leaves at least 120 ms between complete sentence waveforms. Speech is never shortened
by cropping to subtitle timestamps; overflow can shift subsequent sentences and is reported.
These are tested safety budgets, not a guarantee that Whisper identifies every acoustic ending.

`doctor` reads `.python-version`, checks the report directory, and warns if the version pin is
missing or malformed instead of crashing.

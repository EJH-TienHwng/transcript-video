# Transcript Video

Local, GPU-first tools for Vietnamese transcription, externally translated English subtitles, Qwen TTS voice-over, and compiling processed sessions into a training course.

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

## Subtitle workflow

1. Run `uv run transcript-video process VIDEO`. Whisper creates or reuses the application-owned Vietnamese source SRT at `data/subtitles/source/<stem>_vi_<backend>.srt`.
2. Send that SRT to an external LLM with [`docs/prompts/optimal_prompt.md`](docs/prompts/optimal_prompt.md).
3. Save the edited English result as `data/subtitles/translated/<stem>_en.srt`, or pass it with `--translated-srt PATH`.
4. Rerun `process`. Timed/chunked TTS is placed first, then the derived `data/subtitles/retimed/<stem>_en_retimed.srt` is burned before the final mux. Disabled/simple TTS burns the translated SRT directly.

Files under `translated/` are user-owned and are never overwritten. If the English SRT is missing, processing stops cleanly after the source SRT and prints the handoff paths without loading Qwen.

Retiming changes timing only. It keeps each translated start unless TTS is meaningfully later, and keeps each translated end unless TTS runs longer. A cue may end earlier only to hand off to the next retimed cue, never before its own narration ends. Comparisons use SRT millisecond precision, and the TTS review records original, actual placement, final retimed values, shifts, and reasons.

## Basic CLI

```powershell
# Process every configured video
uv run transcript-video process

# Generate/reuse the Vietnamese source SRT only
uv run transcript-video process lesson.mp4 --profile srt

# Generate and verify final sentence audio with the CPU Whisper verifier
uv run transcript-video process lesson.mp4 --profile tts-review

# Use an English SRT stored outside the canonical translated directory
uv run transcript-video process lesson.mp4 --translated-srt edits/lesson_en.srt

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

Positional inputs accept names inside `data/input/` or absolute paths. Repeated paths are processed once, keeping the first occurrence. Distinct inputs with the same filename stem are rejected to prevent output collisions. With no positional inputs, `project.video` is used when configured; otherwise `data/input/` is scanned. `--translated-srt` is limited to single-video runs; batches resolve each canonical English SRT independently.

The pre-0.3 form (`transcript-video --video lesson.mp4 ...`) remains accepted. The old `transcript-course` and `transcript-course-config` executables are deprecated wrappers.

## Speed up waiting intervals

Normal processing remains unchanged unless speed-up is enabled. To create the normal final video
and an additional speed-up artifact in one run:

```powershell
uv run transcript-video process Analysis.mp4 --speedup
```

Each input uses one shared `data/speedup/<stem>.speedup.toml`. If that file is missing, normal processing still succeeds, a commented template is created once, and speed-up encoding is deferred. An existing empty or comment-only file explicitly means that the video was reviewed and needs no speed-up:

```toml
# Reviewed: no speed-up required.
```

A malformed file or invalid/overlapping interval remains a configuration error. To configure speed-up, add intervals:

```toml
[[segment]]
start = "00:04:32.500"
end = "00:05:50.000"
speed = 10
label = "Running build..."
```

Allowed factors are `2`, `5`, and `10`. Timestamps accept `HH:MM:SS`, milliseconds with `.` or
`,`, and `MM:SS`. Intervals are sorted but must not overlap or exceed the final video duration.
The optional label appears above `Speed up ×N` in the top-right overlay. White text uses a thin
black character outline without a background box.

After adjusting timestamps, regenerate only the additional artifact without running Whisper,
subtitle rendering, TTS, or normal muxing:

```powershell
uv run transcript-video speedup Analysis.mp4
uv run transcript-video speedup Analysis.mp4 Build.mp4 Configuration.mp4
uv run transcript-video speedup Analysis.mp4 --spec custom.toml
```

Batch runs resolve one conventional spec per video, continue after individual failures, and exit
non-zero after the summary if any video failed. `--spec` is accepted only with one video.

Normal outputs are `Analysis_vi-dub_en-sub.mp4` (Vietnamese audio with English subtitles) and, when TTS is enabled, `Analysis_en-dub_en-sub.mp4` (English TTS with English subtitles). `process --speedup` applies the same spec independently to every normal output produced by that run, creating both corresponding `_speedup.mp4` files when TTS is enabled. The standalone command discovers either or both existing canonical normal outputs without consulting the current TTS setting.

`--speedup-spec PATH` selects one custom spec for `process` and implies `--speedup`; it is rejected for multi-video runs. `[speedup] enabled = false` is the default. `process --dry-run --speedup` reports missing, zero-segment, or configured spec state and planned outputs without creating templates or media.

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
Review offers Add session using the same selection, duplicate checks and metadata cache.
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
uv run transcript-video process lesson.mp4 --profile tts-review --tts-speaker Ryan
uv run transcript-video process lesson.mp4 --force transcription
```

`--force transcription` regenerates only the Vietnamese source SRT. It never modifies the
translated English SRT. Existing source SRTs are otherwise reused solely by file existence;
model/config changes do not invalidate them. TTS generation reached by a normal run reuses
valid output and chunks. Use `--force tts` for a full rebuild; `--rerun-tts-chunk INDEX` remains
an explicit selective debugging control.

Dry-run checks video paths/extensions, stem collisions, the local ASR model format and
paths, settings, explicit binary paths and output path conflicts without loading weights or running
FFmpeg. It separately reports source/translated subtitle existence and whether each video reaches
the translation handoff. It creates no artifacts, directories, logs or event files, including with
`--save-config`. Inspection works without model folders; unresolved predictions are labeled explicitly.
Human inspection and configuration validation use tables; `--json` also works for `config validate`.
Doctor separates configured encoder, NVENC listing/runtime and software fallback checks, reports
Python/PyTorch/CUDA/ASR compute support, model paths, storage and directory usability as PASS/WARN/FAIL.

The Textual editor has three screens: metadata/settings, sessions, and review/build. Its browser
lists supported files in `data/input` and `data/output` without recursion; Enter fills the form,
Add confirms, and manual paths remain available. Metadata is probed in a worker and cached by
path/mtime/size. Edit retains the original until Save changes succeeds; Cancel edit (`Ctrl+E`)
keeps it intact. Save or cancel an edit before remove/reorder/review. Session numbers and unknown
JSON fields are preserved. Corrupt/invalid JSON aborts launch with a path-bearing error and is
never silently replaced. Saves validate through the shared course API and replace JSON atomically.

Settings include cards, chapters, TOC and advanced render/font fields. Review shows per-session
and total durations plus estimated output duration including cards/TOC. Progress shows completed
course stages, actual stage/FFmpeg counts, session counts and the log; unknown totals stay busy.
`Ctrl+S` saves, `Esc` goes back, `A/E/Delete` adds/edits/removes, `U/D` reorders, `Ctrl+B` builds on
review and `Q` quits (letter bindings apply outside inputs). During build, navigation/save/quit
are guarded until the worker finishes. Textual `--no-color` uses grayscale; fullscreen cursor
control remains necessary. CLI and wizard retain normal no-color output.

## Developer workflow

```powershell
just install
just format
just check
uv run pre-commit install
```

Hosted CI runs on Windows and Linux with `.github/requirements-test.txt` and an editable
`--no-deps` install. This runs the mocked/unit/Pilot suite without CUDA wheels or model runtimes.
The normal `uv sync` environment and local PyTorch wheels are unchanged. Useful recipes also
include `just wizard`, `just test-tui`, `just test-fast` and `just coverage`.

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

JSONL keeps one object per line; read `.pretty.json` for manual inspection. Normal runs reuse valid
TTS chunks; review metadata retains sentence ranges for assembly and explicit selective reruns.
Old provenance metadata is ignored and may be removed manually. Direct Python TTS calls default to the current project report directory;
pass `review_log_path` (or `review_path` for a single chunk) for a different project.

TTS protects a 180 ms tail budget and the next sentence's onset, regenerating unsafe boundaries.
Placement leaves at least 120 ms between complete sentence waveforms. Speech is never shortened
by cropping to subtitle timestamps; overflow can shift subsequent sentences and is reported.
These are tested safety budgets, not a guarantee that Whisper identifies every acoustic ending.

`doctor` reads `.python-version`, checks the report directory, and warns if the version pin is
missing or malformed instead of crashing.

Review schema **v4** adds optional final-WAV ASR text verification, normalized words, coverage,
missing/added words, first/last matches, and a `word_error_ratio` from SequenceMatcher edits
(not minimum-edit WER). `missing_final_word` identifies an unmatched expected final word.
`possible_truncated_tail` is a 20 ms RMS heuristic (absolute >0.01 and >25% of sentence RMS),
only a manual-review hint. It can flag valid unpadded fricatives. No pronunciation, pitch,
prosody or voice-consistency score is inferred from ASR agreement.

Use `--profile tts-review` or `tts.verify_final_audio = true`; CLI overrides include
`--verify-final-audio` / `--no-verify-final-audio`. The review profile uses the user-owned English
SRT and English TTS; voice/instruction remain inherited from the base config. Verification reads each sentence's sample
range from the published final WAV without an expected-text ASR prompt. Timed full generation
verifies once; chunked runs verify the rebuilt WAV. The existing
faster-whisper CPU aligner is reused. Missing/failed verification is explicitly flagged;
it never counts as a text pass. Full simple/untimed generation cannot enable this option.

Every applied speed-up above 1 + 1e-6 emits a sentence REVIEW; quality summaries include
`speed_adjusted`. To review the fastest sentences first:

```powershell
Get-Content data/report/tts/lesson_tts_review.pretty.json -Raw |
    ConvertFrom-Json | Sort-Object applied_speedup -Descending |
    Select-Object subtitle_index, text, applied_speedup, review_reason
```

`data/report/tts/<video>_tts_review.duration.json` records source video, final WAV and muxed
output durations, audio-minus-source `tts_duration_delta`, and output-minus-source
`duration_delta`. Unknown measurements are null. If mux fails, the audio measurements remain
with null output duration. Speech longer than the source is preserved; no `-shortest` is added.
Listen to flagged sentence ranges and compare adjacent sentences for pronunciation, register,
technical terms, pauses and naturalness. Mocked tests do not establish real speech quality.

## Artifact ownership and safety

Vietnamese ASR writes only under `data/subtitles/source/`. English SRTs under
`data/subtitles/translated/` are external, user-owned inputs: the pipeline only resolves, reads,
validates, and reports them. There is no artifact provenance/fingerprint cache and no automatic
invalidation. Old `*.provenance.json` files are ignored and may be removed manually.

SRT, generated/rebuilt WAV, rendered/muxed/course MP4 and reports use unique
same-directory `.<name>.<id>.partial.<extension>` files, then `os.replace`. Writers close the
temporary file before replacement on Windows. Failure/Ctrl+C keeps the previous final file;
temporary files are cleaned in `finally`. Hard-killed orphan partials are excluded from media
discovery; they are not swept automatically because another process may own them.

Batch execution owns a scoped model runtime. Each backend loads once per unchanged
configuration; previous GPU weights are parked on CPU before another GPU workload loads,
including Qwen's separate speech codec. The CPU aligner remains reusable. Cleanup releases
the runtime after success or failure; single-video and batch input ordering stay unchanged.
This trades host RAM and CPU/GPU transfers for avoiding repeated weight loads. It cannot make
a model fit on an undersized GPU, and real model-transfer performance remains unbenchmarked.

## Subtitle appearance and optional attention

The optional `[subtitle_style]` table preserves the existing default `MarginV=25`; unset
fields retain libass defaults. Supported keys are `font_name`, `font_size`, `primary_color`,
`outline_color`, `border_style` (1/3), `outline`, `shadow`, `alignment` (1–9), and
`margin_vertical`, `margin_left`, `margin_right`. Colors use `&HAABBGGRR`. Font names allow
Unicode letters/numbers, spaces, dots, parentheses, hyphens and underscores; ASS/filter
delimiters are rejected to prevent injection. Numbers must be finite and nonnegative; font
size must be positive, margins/enums integer. NVENC settings are unchanged.

```toml
[subtitle_style]
font_name = "Arial"
font_size = 18
primary_color = "&H00FFFFFF"
outline = 1.5
margin_vertical = 25
```

FlashAttention 2 remains optional. An explicit `tts.attn_implementation = "flash_attention_2"`
checks CUDA capability, package import/ABI and a tiny FP16 kernel before loading Qwen, with
WARN + SDPA fallback when unavailable. Doctor reports the optional check. See the
[validation notes](docs/quality-validation.md) for this machine's compatibility limits.
No dependency, CUDA wheel version or SoX requirement was added.

Python consistency is checked by `uv run python scripts/check_python_version.py`, pytest and
CI against the canonical `.python-version`, the packaging minor-version interval and CI pin.

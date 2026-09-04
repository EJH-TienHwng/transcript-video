You are working on the current `dev` branch of the `transcript-video` repository.

The repository has already undergone a major DX/CLI/TUI refactor. DO NOT start over and DO NOT replace the architecture wholesale.

Before modifying anything, read the current implementation carefully, especially:

* `src/transcript_video/cli.py`
* `src/transcript_video/events.py`
* `src/transcript_video/process_runner.py`
* `src/transcript_video/application/`
* `src/transcript_video/ui/`
* `src/transcript_video/processing/`
* `src/transcript_video/course/`
* `src/transcript_video/course/wizard.py`
* `src/transcript_video/tui/app.py`
* `tests/`
* `pyproject.toml`
* `justfile`
* `.pre-commit-config.yaml`
* README/docs

The current architecture is fundamentally good and must be preserved:

```text
CLI / Wizard / Textual TUI
            ↓
      Application layer
            ↓
      Processing / Course core
```

Core processing must remain independent from Rich, Typer, Textual, Questionary, and terminal-specific presentation.

The objective of this task is to COMPLETE and HARDEN the existing QoL implementation, fixing the remaining behavioral and UX gaps.

Do not create placeholder implementations.

Do not claim a feature is complete unless it is functional and tested.

---

# 1. Pipeline event lifecycle

The current `PipelineEvent` abstraction is a good foundation but is not expressive enough.

Add an explicit event/lifecycle type such as:

```python
class PipelineEventType(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
```

Update `PipelineEvent` so UI code never needs to infer semantic state from free-form `message` strings.

Desired shape conceptually:

```python
@dataclass(frozen=True, slots=True)
class PipelineEvent:
    stage: PipelineStage
    type: PipelineEventType
    message: str = ""
    current: float | None = None
    total: float | None = None
    details: Mapping[str, object] | None = None
```

Adjust the exact implementation to match the existing codebase.

Emit correct lifecycle events for:

* input discovery
* transcription
* translation
* subtitle writing
* TTS
* FFmpeg render
* mux
* cached/reused stages
* failure
* completion

Do not encode state only in strings such as `"Reusing cached subtitles"`.

---

# 2. Implement real Rich progress

The existing Rich observer currently behaves mostly like a spinner.

Upgrade it into real stage-aware progress.

Use appropriate Rich components such as:

```text
SpinnerColumn
TextColumn
BarColumn
TaskProgressColumn
TimeRemainingColumn
TimeElapsedColumn
```

Desired behavior:

```text
✓ Transcription     cached
✓ Translation       cached
✓ Subtitles         complete
→ TTS               ━━━━━━━━━━━━━╺━━━━━━ 67%
○ Render            waiting
```

For stages without measurable progress, use an indeterminate spinner.

NEVER invent fake percentages.

Maintain separate stage state instead of representing the entire pipeline as one generic task.

---

# 3. FFmpeg percentage progress

The repository already parses machine-readable FFmpeg progress using `-progress pipe:1`.

Complete the implementation.

Before invoking FFmpeg, determine total input duration using the existing ffprobe abstraction when possible.

Emit:

```python
current = elapsed_seconds
total = duration_seconds
```

and include speed when available.

Desired CLI:

```text
Rendering video    ━━━━━━━━━━━━━━━╺━━━━ 76%  1.82x  ETA 00:41
```

Desired Textual TUI:

```text
Rendering video
████████████████░░░░ 76%
```

Handle correctly:

* unknown duration
* malformed FFmpeg progress
* missing speed
* FFmpeg failure
* Ctrl+C
* subprocess cleanup
* Windows behavior

If duration cannot be determined, use indeterminate progress rather than a fake percentage.

---

# 4. Translation and TTS progress

Where real progress exists, expose it through `PipelineEvent`.

For example, if translation processes N subtitle segments/batches:

```python
current = completed_segments
total = total_segments
```

For chunked TTS:

```python
current = completed_chunks
total = total_chunks
```

For Whisper/transcription, do not invent a percentage unless the backend exposes sufficiently reliable progress.

Use a spinner when necessary.

---

# 5. Fix verbosity semantics

The intended behavior is:

```text
-q
    warnings/errors and essential result only

default
    concise polished application UX
    not noisy diagnostic logging

-v
    INFO diagnostic/runtime information

-vv
    DEBUG information
    external commands
    detailed paths
    tracebacks/source information
```

Currently `-v` and `-vv` are not sufficiently differentiated.

Fix logging level mapping.

Presentation output and diagnostic logging must remain separate concepts.

---

# 6. Structured subprocess errors

The terminal must not dump hundreds of lines of FFmpeg stderr by default.

Refactor `ProcessExecutionError` or equivalent so it stores structured fields:

```text
command
returncode
stdout
stderr
label/tool
```

Its normal `str()` representation should be short.

Example:

```text
FFmpeg exited with code 1
```

Full stderr must be written to DEBUG/file logging.

Allow UI code to optionally show only a short tail or relevant extracted error.

---

# 7. Improve application error UX

Normal CLI errors should answer:

1. What failed?
2. At which stage?
3. What is the likely cause if confidently known?
4. What can the developer try?
5. Where is the detailed log?

Example:

```text
✗ Video rendering failed

FFmpeg could not use `h264_nvenc`.

Try:
  transcript-video process lesson.mp4 --video-encoder libx264

Detailed log:
  logs/transcript-video.log

Run with -vv for full diagnostics.
```

Do not make unreliable guesses.

Add small exception types/application error metadata where useful.

With `-vv`, show Rich traceback.

Without `-vv`, do not dump traceback.

---

# 8. Initial process summary

Before a real processing run, print a concise Rich configuration summary.

Example:

```text
╭─ Transcript Video ───────────────────╮
│ Input        lesson.mp4              │
│ Model        faster-whisper-large-v3 │
│ Device       CUDA / int8_float16     │
│ Task         transcribe              │
│ Translation  VinAI                   │
│ TTS          Aiden / timed           │
│ Encoder      h264_nvenc              │
╰──────────────────────────────────────╯
```

Only show relevant enabled features.

Respect:

```text
-q
--no-color
--json where applicable
```

---

# 9. Better final run summary

Collect real stage timings and useful metrics.

Example:

```text
╭─ Completed ──────────────────────────╮
│ Video          lesson.mp4            │
│ Subtitles      184 segments          │
│ Transcription  2m 18s                │
│ Translation    31s                   │
│ TTS            1m 04s                │
│ Render         52s                   │
│ Total          4m 45s                │
│                                      │
│ SRT   data/subtitles/...             │
│ MP4   data/output/...                │
╰──────────────────────────────────────╯
```

Only show metrics that really exist.

Do not fabricate data.

---

# 10. Complete `--no-color`

`--no-color` must consistently affect non-TUI terminal presentation:

* Rich console output
* Rich logging
* errors
* summaries
* progress
* Wizard presentation
* help output if technically possible through Typer/Rich configuration

Also honor the standard `NO_COLOR` environment variable.

Piped/redirected output must not contain unnecessary ANSI sequences.

For Textual TUI, either:

1. implement a clean monochrome theme when `--no-color` is supplied, OR
2. explicitly document and test that `--no-color` applies to non-fullscreen output only.

Prefer option 1 if it can be implemented cleanly.

Add automated ANSI/no-color tests.

---

# 11. Wizard cleanup and redesign

The current Questionary implementation is correctly called a Wizard now, but it still contains old/plain output and incomplete flow.

Keep Questionary.

Do NOT convert the Wizard into Textual.

Use the shared Rich console/theme for presentation.

Fix all known issues:

* replace raw decorative `print()` output with Rich presentation
* fix `_review_sessions`
* remove unreachable/dead review code
* preserve Ctrl+C behavior
* support `--no-color`
* make validation errors colored/readable
* prevent duplicate session numbers when EDITING, not only when initially creating
* never allow the Wizard to save a config that `load_course_config()` rejects

The review loop must support:

```text
Add session
Edit session
Move up
Move down
Remove session
Continue
Cancel/back where sensible
```

A user who finishes initial video selection and notices a missing video must not have to restart the Wizard.

Render the session list as a Rich table where appropriate.

---

# 12. Fix Textual destructive edit bug

Current session editing removes the selected session before the edit is confirmed.

Fix this.

Maintain something such as:

```python
editing_index: int | None
```

When editing:

* populate fields
* keep the original item in the list
* show `Save changes`
* mutate the original item only after confirmation
* preserve its position
* allow cancelling edit without data loss

Add an automated test for this behavior.

---

# 13. Textual video browser

Upgrade Session Editor from requiring users to manually type every video path.

Add an available-video panel/browser on the left and course-session table/list on the right.

Concept:

```text
┌ Available videos ───────┬ Course sessions ───────────┐
│ intro.mp4               │ 01 Introduction            │
│ can.mp4                 │ 02 CAN Basics              │
│ lin.mp4                 │ 03 LIN Basics              │
└─────────────────────────┴────────────────────────────┘
```

Use Textual widgets that best fit the implementation, such as:

* `DirectoryTree`
* `DataTable`
* `ListView`

Do not scan arbitrary huge directories unnecessarily.

Allow manual path input as an escape hatch if useful.

---

# 14. Textual metadata panel

When a video is selected/highlighted, show useful metadata using ffprobe:

```text
Duration       12:42
Resolution     1920 × 1080
FPS            30
Video codec    H.264
Audio codec    AAC
Sample rate    48 kHz
Channels       stereo
Size           428 MiB

Artifacts
✓ subtitle
✓ TTS audio
✗ final output
```

Reuse a centralized media inspection abstraction.

Do not execute ffprobe on the Textual UI thread.

Continue using Textual workers.

Cache metadata where appropriate so moving selection does not repeatedly probe unchanged files.

---

# 15. Textual progress

Replace the build screen's log-only progress with actual Textual `ProgressBar` widgets/state.

Still keep a log panel for diagnostic/status text.

Use `TextualProgressObserver`.

The TUI must consume the same semantic pipeline events used by Rich CLI progress.

Do not redirect Rich progress into Textual.

---

# 16. Textual review improvements

Review must show:

* course title
* output path
* sessions
* session video paths
* per-video duration if known
* total duration if known
* save
* build
* back

Keep keyboard bindings visible through Textual Footer.

---

# 17. Textual configuration fields

The current Metadata screen exposes only a small subset of `CourseConfig`.

Add access to useful existing course settings without inventing new fields.

At minimum expose important fields already supported by `CourseConfig`, such as relevant:

* width
* height
* FPS
* encoder
* bitrate
* card duration
* theme
* output

Use a basic/advanced settings UI if showing everything at once would be cluttered.

---

# 18. Invalid Textual configuration handling

Do NOT silently replace corrupted JSON with a blank draft.

If an existing course config cannot be parsed:

* display a clear error/modal
* show the file path
* preserve the original file
* do not silently overwrite it

Add tests.

---

# 19. Centralize CourseConfig load/save

The Wizard, Textual UI, and course builder should not each independently invent serialization rules.

Reuse or extend the course configuration domain API.

Create clean helpers such as conceptually:

```text
load_course_config
save_course_config
draft_from_config
config_from_draft
```

Validate before writing.

Keep unknown/forward-compatible fields only if doing so is deliberate and safe.

---

# 20. Improve `inspect`

Normal human-readable `inspect` must NOT print raw ffprobe JSON.

Render a Rich panel/table containing useful fields:

```text
Video
  Duration
  Resolution
  FPS
  Codec
  Bitrate

Audio
  Codec
  Channels
  Sample rate

File
  Size

Artifacts
  Path
  Exists
```

Keep raw structured data available through:

```bash
transcript-video --json inspect VIDEO
```

Media metadata inspection must work even if the transcription model directory is missing.

Artifact prediction/discovery should be best-effort and must not prevent basic media inspection.

---

# 21. Improve `doctor`

Keep existing checks and add useful project-specific checks where possible:

* Python version
* config validity
* ffmpeg
* ffprobe
* subtitle filter
* configured encoder
* NVENC availability
* fallback libx264 availability
* PyTorch version
* PyTorch CUDA build
* torch CUDA availability
* GPU name
* faster-whisper/CTranslate2 compute support where safely detectable
* transcription model
* translation model when enabled
* TTS configuration/model/cache when enabled
* writable directories
* free disk space

Do not load huge models just to run `doctor`.

Distinguish:

```text
PASS
WARN
FAIL
```

Clearly distinguish:

```text
configured h264_nvenc unavailable
libx264 fallback available
```

instead of collapsing them into one generic encoder check.

---

# 22. Improve `config validate`

Do not only print:

```text
Valid configuration
```

Produce structured validation checks:

```text
✓ TOML syntax
✓ Known sections
✓ Known fields
✓ Enum values
✓ Hardware configuration
✓ Translation configuration
✓ TTS configuration

Configuration valid
```

Failures must identify exact setting/path.

Maintain non-zero exit code for invalid configuration.

---

# 23. Fix config source tracking

Verify configuration source tracking for legacy/migrated settings.

Normalization/migration must happen before source provenance is recorded.

Example:

A legacy:

```toml
[transcription]
device = "cpu"
```

that migrates to:

```text
hardware.device
```

must report the source of `hardware.device` as the config file rather than `default`.

Create tests.

---

# 24. Complete `--force`

Review the current `ForceTarget` implementation.

Current behavior must not advertise targets whose semantics are effectively no-ops.

Support:

```text
--force transcription
--force translation
--force tts
--force render
--force all
```

only when they have real defined behavior.

If render caching is not part of the architecture, either implement clean render-reuse semantics or remove/defer `render` rather than exposing a misleading option.

Keep old overwrite flags working for compatibility.

Document precedence.

---

# 25. Improve dry-run validation

`--dry-run` must continue to:

* not load heavy models
* not encode media
* not write artifacts
* not write logs unnecessarily
* not create output directories unnecessarily

But it should validate lightweight prerequisites:

* video exists
* model directory/configured path exists where required
* translation model path when enabled
* basic configuration values
* output path resolution

Do not silently replace missing model information with placeholders when that hides a real configuration mistake.

Add tests proving dry-run does not modify files.

---

# 26. Compatibility/deprecation output

Legacy executable wrappers must remain compatible where reasonable.

Deprecation warnings must actually be visible to normal users.

Do not rely solely on Python `DeprecationWarning`, which may be hidden.

Use stderr/Rich output such as:

```text
Warning: `transcript-course` is deprecated.
Use `transcript-video course build`.
```

Do not duplicate business logic.

---

# 27. Split oversized presentation modules where useful

After behavior is correct, reduce oversized files.

Prefer a CLI structure such as:

```text
cli/
├── app.py
├── process.py
├── config.py
├── doctor.py
├── inspect.py
└── course.py
```

and Textual structure such as:

```text
tui/
├── app.py
├── models.py
├── observers.py
├── screens/
└── widgets/
```

ONLY do this after tests cover current behavior.

Do not perform a risky cosmetic rewrite solely to match this exact tree.

---

# 28. Testing requirements

Expand automated tests substantially.

Cover at least:

## CLI

* root help
* process help
* course help
* config help
* version
* unknown option exit code
* `--no-color`
* `NO_COLOR`
* `-q`
* default verbosity
* `-v`
* `-vv`

## Config

* default/config/profile/CLI precedence
* source tracking
* legacy source tracking
* invalid TOML
* unknown key
* invalid enum
* profile resolution

## Dry run

* valid plan
* missing video
* missing model
* no files written
* `--save-config` does not write during dry-run

## Doctor

* fully healthy mocked environment
* missing ffmpeg
* missing ffprobe
* CUDA unavailable
* NVENC unavailable but libx264 available
* required failure exit code
* JSON output

## Inspect

* normal media
* JSON
* missing video
* missing transcription model must not prevent media metadata inspection
* artifact existence detection

## Events/progress

* STARTED
* PROGRESS
* COMPLETED
* SKIPPED
* FAILED
* current/total
* unknown-total progress

## FFmpeg

* parser
* elapsed time
* speed
* malformed progress
* unknown duration
* successful process
* failed process
* stderr preservation
* Ctrl+C/subprocess termination where testable

## Wizard domain logic

Move pure session manipulation into testable functions when appropriate.

Test:

* add
* edit
* duplicate number rejection
* reorder
* remove
* adding a session after initial review

## Textual

Use Textual pilot/test APIs.

Test:

* app smoke
* add
* edit without destructive removal
* cancel edit
* remove
* reorder
* save
* reload
* unsaved quit confirmation
* invalid JSON handling
* review
* build progress observer

No default test must require downloading/loading large AI models.

Mark GPU/slow/integration tests appropriately.

---

# 29. CI

Add a GitHub Actions quality workflow if feasible with the dependency architecture.

At minimum it should run lightweight quality checks:

```text
ruff format --check
ruff check
pytest -m "not gpu and not slow"
```

If the current heavyweight CUDA dependency design prevents sensible hosted CI, restructure the test/dependency installation only as much as necessary, or document the blocker accurately.

Do not pretend CI works if the environment cannot install the configured CUDA stack.

---

# 30. Developer tooling

Keep the existing `justfile` and pre-commit setup.

Add useful recipes if appropriate:

```text
wizard
coverage
test-unit
test-fast
```

Do not make hooks run GPU/model-heavy tests.

---

# 31. Optional dependency evaluation

This is lower priority.

After all functional work above is complete, evaluate whether heavyweight dependencies can safely be separated into extras such as:

```text
asr
tts
tui
dev
```

Do NOT destabilize the known-good CUDA/uv environment merely for aesthetic dependency separation.

If unsafe, leave it unchanged and explain why.

---

# 32. Documentation

Update README/docs to describe ONLY behavior that actually exists.

Especially verify documentation for:

* `-q/-v/-vv`
* `--no-color`
* FFmpeg percentage progress
* Wizard editing/add/back behavior
* Textual video browser
* doctor
* inspect
* config validation
* force targets
* compatibility commands

Do not overstate incomplete functionality.

---

# 33. Validation workflow

Work in phases.

After each substantial phase run:

```powershell
uv run ruff format .
uv run ruff check .
uv run pytest -m "not slow and not gpu"
```

At the end run the widest test suite the environment supports.

Also run CLI smoke tests manually:

```powershell
uv run transcript-video --help
uv run transcript-video process --help
uv run transcript-video config --help
uv run transcript-video course --help
uv run transcript-video doctor
```

If a test cannot run because the environment lacks GPU, FFmpeg, models, or another external dependency, say exactly which test was not run.

Never report "all tests pass" unless all tests were actually executed.

---

# 34. Priority order

Implement in this order:

## P0

1. event lifecycle
2. real Rich progress
3. FFmpeg total/percentage progress
4. structured subprocess errors
5. pretty error UX
6. verbosity semantics
7. Wizard correctness bugs
8. Textual destructive Edit bug
9. tests for these behaviors

## P1

 1. Wizard Rich/no-color UX
 2. Textual ProgressBar
 3. Textual video browser
 4. Textual metadata
 5. invalid JSON handling
 6. centralized CourseConfig persistence
 7. better inspect
 8. better doctor
 9. config validation/source fixes
10. process/final summaries
11. force semantics

## P2

 1. split oversized CLI/TUI modules
 2. CI
 3. optional dependency evaluation
 4. docs finalization

Do not perform P2 cosmetic refactoring before P0/P1 behavior is correct.

---

# 35. Final report

When finished provide:

## Fixed

Concrete bugs fixed.

## Completed QoL features

Features that are now genuinely functional.

## Architecture changes

Explain new event/progress/error abstractions.

## CLI behavior

Document exact commands/options.

## Wizard behavior

Document workflow.

## TUI behavior

Document screens/bindings.

## Tests executed

Give exact commands and results.

## Tests not executed

Give exact reason.

## Remaining limitations

Only real limitations.

## Files changed

Summarize important files.

Do not provide a vague success summary.

Begin by reading the CURRENT `dev` branch and confirming the existing architecture. Then implement these changes incrementally instead of rewriting the project.

xin chào
con cò bé bé
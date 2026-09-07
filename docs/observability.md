# Events, logs, and terminal UI

The core emits semantic events and diagnostic logging independently. It imports no Rich,
Questionary or Textual renderer. `event_scope()` attaches an observer to the current execution
context; `log_context()` scopes immutable run/video/stage/chunk/subtitle/operation fields.
Both reset their ContextVars on success and exceptions. Direct Python calls can pass an observer
or use a scope; calls without one remain silent except for standard library logging.

```text
pipeline / course builder / model loading / TTS
  ├─ emit / stage_context → PipelineEvent
  │   └─ CompositeObserver → RichProgressObserver / TextualObserver / JsonEventObserver
  └─ logging → ContextFilter → rotating text / run text / run JSONL
```

## Event schema v1

`PipelineEvent` keeps its original positional fields: stage, message, current, total, artifact,
details. New optional fields are kind, context and timestamp. `RecordingObserver` keeps the
original event objects. `to_dict()` returns JSON-compatible data; producers must put only small
JSON primitives/containers in details, never model objects, waveforms or exception instances.

Every event line has these fields:

| Field | Meaning |
| --- | --- |
| schema_version | Integer `1` |
| timestamp | ISO 8601 wall time, milliseconds, local UTC offset |
| kind | `start`, `progress`, `complete`, `review`, `warning`, `failure`, `artifact`, `reused` |
| severity | Derived: `normal`, `review`, `warning`, `error` |
| stage | Stable `PipelineStage` string |
| message | Human description; consumers should branch on kind/stage, not parse messages |
| current, total | Number or null; total is null when unknown |
| artifact | Path string or null |
| details | JSON object; stage-specific documented fields below |

Optional context fields are flattened: `run_id`, `video`, `chunk`, `subtitle`, `operation`.
Absent context fields are omitted. `stage` always identifies the event's stage, even when emitted
inside another operation. Chunk indexes are zero-based; subtitle indexes are one-based.
CLI run IDs combine local timestamp and ten random hex digits.

Example line (formatted here for reading; on disk it is one physical line):

```json
{
  "schema_version": 1,
  "timestamp": "2026-09-06T13:00:00.000+07:00",
  "run_id": "20260906-130000-0123456789",
  "video": "lesson.mp4",
  "stage": "tts",
  "chunk": 2,
  "operation": "chunks",
  "kind": "progress",
  "severity": "normal",
  "message": "TTS chunks ready",
  "current": 3,
  "total": 4,
  "artifact": null,
  "details": {"unit": "chunks", "sentences": 92, "total_sentences": 126}
}
```

## Lifecycle and progress

`stage_context()` emits START and COMPLETE (REUSED when `reused=True`), or FAILURE before re-raising.
REUSED is an additive v1 kind for validated subtitle/audio/chunk cache reuse; JSON fields are unchanged.
Chunk reuse uses operation=chunks and does not complete the enclosing TTS stage. Consumers should tolerate new kinds. Terminal lifecycle
events include `details.elapsed_seconds`, measured with a monotonic clock. Nested operations
such as `load_qwen`, `load_aligner`, `load_whisper`, `load_translation`, and `chunk` do not complete
the enclosing stage. RUN completion means all inputs were attempted; count VIDEO failures to
determine batch success. A failed video does not prevent later videos from running.

Process stages include run, video, transcribe, translate, subtitles, render, tts, mux.
Course stages include run, validate, normalize, toc, cards, concatenate, chapters. The final
course ARTIFACT uses stage `complete` and details `sessions`, `duration`, `chapters`, `category`.

FFmpeg PROGRESS uses operation `ffmpeg`, current output seconds and known media duration as total.
Details contain `unit=seconds`, `speed` (e.g. `2.6x`), and FFmpeg `state`. End-of-process success
comes from the enclosing COMPLETE event, not from percent reaching 100. Unknown durations have
no percentage/ETA. ETA is remaining output duration divided by a positive FFmpeg speed; it is
only an estimate and is omitted for invalid/missing speed.

TTS emits owned-sentence progress within context groups and ready chunk counts including reused
cache entries. A sentence crossing a fixed chunk boundary is counted by its owner, once.
`operation=quality` provides final metadata counts: sentences, aligned, regenerated, speed_adjusted, shifted,
overflow, failures, flagged. `failures` counts sentences with no generated waveform, not recovered
attempts. `flagged` counts unique report entries with a nonempty review_reason, including recovered
alignment cases. Counts overlap (a regenerated sentence can also overflow). Individual REVIEW
events carry subtitle context plus action, timing_shift, overflow_duration, required_speedup and
applied_speedup. Every speed-up above 1 + 1e-6 has a `speed_adjusted` reason even without overflow.
Review schema v4 adds text/tail QA; semantic event schema remains v1. Cache invalidation uses a
WARNING with `operation=cache`, `decision=invalidated` and a short reason; matching artifact/model
reuse emits REUSED. Mux REVIEW carries source/WAV/output durations and both duration deltas.
See [the README](../README.md#tts-review-reports) for fields and manual acceptance limits.

ARTIFACT events use `details.category`: Subtitles, Audio, Video, Review. They are emitted when the
artifact is available, including validated reused artifacts. The CLI collects these for its result.

## Diagnostic logs and error ownership

The console starts at WARNING; `-v` enables INFO and `-vv` DEBUG. Run files always capture package
DEBUG output. Noisy third-party loggers are limited to WARNING (INFO under -vv). The logging setup
owns only its tagged handlers, preserves external handlers, and restores prior logger levels on
close. Text files include only nonempty context fields. JSONL records contain schema_version,
timestamp, level, logger, message, optional context and optional formatted exception text.

Core code raises exceptions. The subprocess runner retains complete command, return code,
stdout and stderr in DEBUG logs while raising a short `ProcessExecutionError`. Its command, returncode, stdout, stderr, tool,
timeout and stage fields retain structured context; str() does not include stderr. Batch execution records each
exception and continues; the command renders its final failure list once. Single course failures
use one error panel. Tracebacks never include locals or environment dumps. Per-stage lifecycle
and review diagnostics are written to files without duplicating status on the console.

Important warnings use `warn()` to retain a WARNING diagnostic and publish a WARNING event;
with an observer active, the console logging handler suppresses its duplicate diagnostic line.
Without an observer, standard logging keeps its normal warning behavior.

The Textual worker forwards events using call_from_thread, with invocation context installed
inside the worker. It does not install a second FFmpeg callback. Its worker owns the single
failure log line; a separate main-thread callback updates Textual ProgressBar widgets for every
semantic event. Overall progress counts completed course stages, not estimated encoding work.
Other log events use the same formatting and progress throttling as CLI line mode.

## Terminal and storage limits

Rich Live refreshes at at most eight frames per second for progress; line mode emits progress
at approximately one-second intervals or ten percentage point changes. Lifecycle transitions
are immediate. JSON events are not throttled. Unknown-total work shows a spinner; known totals
show counts and a percentage. `--plain` and non-TTY output never start Live. NO_COLOR does not
remove interactive cursor control; use --plain as well when that is required.

The semantic palette is shared by Rich/Questionary and mapped to a Textual theme. Status symbols
fall back to ASCII for consoles such as cp1252. Rendering is tested at 80 and 120 columns.
The wizard remains linear: checkbox order → editable sessions → appearance → output → review.
Metadata is probed once per path per wizard invocation, with a ten-second timeout per probe;
unreadable metadata displays unavailable and suppresses duration estimates. Estimates include
TOC pages and session cards, never encode time.

Per-run files are intentionally not rotated/deleted automatically. There is no latest.log copy
or terminal hyperlink dependency. An existing --events-json destination is rejected rather than
overwritten. No run log, event file, project directory or artifact is created by process --dry-run.

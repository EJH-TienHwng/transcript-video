from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from transcript_video.context import EventContext, current_context, log_context
from transcript_video.events import (
    CompositeObserver,
    EventKind,
    JsonEventObserver,
    PipelineEvent,
    PipelineStage,
    RecordingObserver,
    emit,
    event_scope,
    ffmpeg_events,
    stage_context,
)
from transcript_video.ui.console import THEME
from transcript_video.ui.logging import (
    ContextFilter,
    JsonLogFormatter,
    close_logging,
    configure_logging,
)
from transcript_video.ui.progress import (
    LineThrottle,
    PipelineProgressState,
    RichProgressObserver,
    format_pipeline_event,
)


def test_lifecycle_context_serialization_and_reset(tmp_path, monkeypatch):
    clock = iter([10.0, 13.5, 20.0, 21.0])
    monkeypatch.setattr("transcript_video.events.time.perf_counter", lambda: next(clock))
    record = RecordingObserver()
    json_observer = JsonEventObserver(tmp_path / "events.jsonl")
    before = current_context.get()
    try:
        with event_scope(CompositeObserver(record, json_observer), run_id="run1", video="A.mp4"):
            with stage_context(PipelineStage.TTS, "Generating"):
                with log_context(chunk=2, subtitle=7):
                    emit(
                        PipelineStage.TTS,
                        "Recovery",
                        kind=EventKind.REVIEW,
                        details={"action": "regenerated"},
                    )
                emit(
                    PipelineStage.TTS, "Audio", kind=EventKind.ARTIFACT, artifact=Path("audio.wav")
                )
            with pytest.raises(ValueError, match="broken"), stage_context(PipelineStage.MUX, "Mux"):
                raise ValueError("broken")
    finally:
        json_observer.close()
    assert current_context.get() == before
    assert [e.kind for e in record.events] == [
        EventKind.START,
        EventKind.REVIEW,
        EventKind.ARTIFACT,
        EventKind.COMPLETE,
        EventKind.START,
        EventKind.FAILURE,
    ]
    assert record.events[3].details["elapsed_seconds"] == 3.5
    lines = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert lines == [event.to_dict() for event in record.events]
    assert lines[1]["chunk"] == 2 and lines[1]["subtitle"] == 7
    assert lines[1]["severity"] == "review" and lines[5]["severity"] == "error"
    assert "chunk" not in lines[3] and lines[2]["artifact"] == "audio.wav"
    with pytest.raises(FileExistsError):
        JsonEventObserver(tmp_path / "events.jsonl")


def test_log_record_context_and_json_formatter():
    def record():
        result = logging.LogRecord("test", logging.INFO, __file__, 1, "Value %d", (3,), None)
        ContextFilter().filter(result)
        return json.loads(JsonLogFormatter().format(result))

    with log_context(run_id="r", video="A", stage="tts"):
        with log_context(chunk=2):
            payload = record()
        assert "chunk" not in record()
    with log_context(video="B"):
        assert record()["video"] == "B" and "run_id" not in record()
    assert payload["message"] == "Value 3" and payload["chunk"] == 2
    assert payload["level"] == "INFO" and payload["timestamp"]


@pytest.mark.parametrize(
    "verbosity,visible",
    [(-1, (False, False)), (0, (False, False)), (1, (True, False)), (2, (True, True))],
)
def test_verbosity_and_owned_handlers(tmp_path, verbosity, visible):
    stream = StringIO()
    console = Console(file=stream, theme=THEME, no_color=True)
    external = logging.NullHandler()
    root = logging.getLogger()
    root.addHandler(external)
    try:
        with log_context(run_id="r", video="A", stage="tts", chunk=2):
            logs = configure_logging(console, verbosity, tmp_path / "global.log", run_id="r")
            configure_logging(console, verbosity, tmp_path / "global.log", run_id="r")
            logger = logging.getLogger("transcript_video.test")
            logger.debug("debug-marker")
            logger.info("info-marker")
            logger.warning("warning-marker")
        assert external in root.handlers
        assert stream.getvalue().count("warning-marker") == 1
        assert ("info-marker" in stream.getvalue(), "debug-marker" in stream.getvalue()) == visible
        entries = [json.loads(line) for line in logs.jsonl.read_text(encoding="utf-8").splitlines()]
        assert [e["message"] for e in entries] == ["debug-marker", "info-marker", "warning-marker"]
        assert all(e["run_id"] == "r" and e["chunk"] == 2 for e in entries)
        assert "run_id=r video=A stage=tts chunk=2" in logs.text.read_text(encoding="utf-8")
        assert "\x1b" not in stream.getvalue()
    finally:
        close_logging()
        root.removeHandler(external)


def test_dry_run_does_not_open_event_or_log_files(tmp_path, monkeypatch):
    model = tmp_path / "models/faster-whisper-large-v3"
    model.mkdir(parents=True)
    (model / "model.bin").touch()
    from transcript_video.cli import app

    video = tmp_path / "test.mp4"
    video.touch()
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    result = CliRunner().invoke(
        app,
        [
            "--no-color",
            "process",
            str(video),
            "--root",
            str(tmp_path),
            "--events-json",
            str(tmp_path / "events/output.jsonl"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert set(tmp_path.rglob("*")) == before
    assert "\x1b" not in result.output


def test_progress_reducer_preserves_parent_timings_and_batch_state():
    state = PipelineProgressState()
    with event_scope(run_id="r", video="A"):

        def apply(stage, kind, **kwargs):
            state.apply(PipelineEvent(stage, str(stage), kind=kind, **kwargs))

        apply(PipelineStage.RUN, EventKind.START, total=3)
        apply(PipelineStage.VIDEO, EventKind.START, details={"stages": ["tts", "mux"]})
        apply(PipelineStage.TTS, EventKind.START)
        with log_context(operation="load_qwen"):
            apply(PipelineStage.TTS, EventKind.COMPLETE, details={"elapsed_seconds": 2})
        assert state.stages["tts"].status == "active" and state.stages["mux"].status == "pending"
        apply(PipelineStage.TTS, EventKind.COMPLETE, details={"elapsed_seconds": 9})
        apply(PipelineStage.VIDEO, EventKind.FAILURE)
    assert state.completed == 1 and state.total == 3
    assert state.timings["A"]["tts"] == 9
    with log_context(video="B"):
        state.apply(PipelineEvent(PipelineStage.VIDEO, "B", kind=EventKind.START))
    assert state.stages == {} and state.videos["A"] == "failure"


@pytest.mark.parametrize("width", [80, 120])
def test_live_unknown_total_plain_and_no_color(width):
    stream = StringIO()
    console = Console(file=stream, theme=THEME, width=width, force_terminal=True, no_color=True)
    with RichProgressObserver(console) as observer:
        observer.notify(PipelineEvent(PipelineStage.RUN, "Start", total=3, kind=EventKind.START))
        with event_scope(video="Video"):
            observer.notify(
                PipelineEvent(
                    PipelineStage.VIDEO,
                    "Video",
                    kind=EventKind.START,
                    details={"stages": ["source subtitles", "translated subtitles", "render"]},
                )
            )
            observer.notify(PipelineEvent(PipelineStage.TTS, "Loading Qwen", kind=EventKind.START))
        console.print(observer.render())
        assert observer.progress.tasks[observer.stage_task].total is None
        assert "Video · TTS" in stream.getvalue()
        observer.notify(PipelineEvent(PipelineStage.TTS, "Chunks", current=1, total=4))
        observer.notify(PipelineEvent(PipelineStage.MUX, "Unknown duration", kind=EventKind.START))
        assert observer.progress.tasks[observer.stage_task].total is None
    assert "Loading Qwen" in stream.getvalue()
    plain_stream = StringIO()
    plain = RichProgressObserver(Console(file=plain_stream, no_color=True, theme=THEME), plain=True)
    assert plain.live is None
    plain.notify(PipelineEvent(PipelineStage.TTS, "started", kind=EventKind.START))
    assert "\x1b" not in plain_stream.getvalue()


def test_ffmpeg_percentage_speed_eta_and_throttle(monkeypatch):
    from transcript_video.process_runner import FFmpegProgress

    observer = RecordingObserver()
    with event_scope(observer):
        ffmpeg_events(PipelineStage.RENDER, "Render", 100)(FFmpegProgress(75, "2x", "continue"))
    event = observer.events[0]
    text = format_pipeline_event(event)
    assert "75%" in text and "2x" in text and "ETA 00:12" in text
    monkeypatch.setattr("transcript_video.ui.progress.time.monotonic", lambda: 1)
    throttle = LineThrottle()
    assert throttle.accepts(event) and not throttle.accepts(event)
    assert throttle.accepts(PipelineEvent(PipelineStage.RENDER, "done", kind=EventKind.COMPLETE))


def test_tts_quality_aggregation_uses_final_metadata_once(tmp_path):
    from transcript_video.processing.tts.core import log_tts_summary

    entries = [
        dict(
            subtitle_index=i,
            action="regenerated_individual_sentence" if i < 3 else "context_aligned",
            timing_shift=1 if i < 2 else 0,
            overflow_duration=1 if i == 0 else 0,
            review_reason="review" if i < 3 else "",
        )
        for i in range(6)
    ]
    record = RecordingObserver()
    progress = RichProgressObserver(Console(file=StringIO(), theme=THEME))
    with event_scope(CompositeObserver(record, progress), video="A"):
        log_tts_summary(6, entries, tmp_path / "review.jsonl")
    counts = progress.state.quality["A"]
    assert counts == dict(
        sentences=6,
        aligned=3,
        regenerated=3,
        speed_adjusted=0,
        shifted=2,
        overflow=1,
        failures=0,
        flagged=3,
    )
    assert len([e for e in record.events if e.context.operation == "quality"]) == 1
    assert [e.context.subtitle for e in record.events if e.context.operation == "review"] == [
        0,
        1,
        2,
    ]


def test_batch_continues_after_failure_and_json_remains_separate(tmp_path, monkeypatch):
    from transcript_video.application import processing
    from transcript_video.cli import app

    config = tmp_path / "run.toml"
    config.write_text(
        '[project]\nmodel="models/asr"\n[transcription]\nskip_burn=true\n', encoding="utf-8"
    )
    (tmp_path / "models/asr").mkdir(parents=True)
    (tmp_path / "models/asr/model.bin").touch()
    videos = [tmp_path / f"{name}.mp4" for name in "ABC"]
    for video in videos:
        video.touch()
    seen = []

    def process(video, *args):
        seen.append((video.name, current_context.get().video))
        if video.stem == "B":
            raise ValueError("intentional failure")
        with stage_context(PipelineStage.SUBTITLES, "Subtitles"):
            emit(
                PipelineStage.SUBTITLES,
                "Ready",
                kind=EventKind.ARTIFACT,
                artifact=video.with_suffix(".srt"),
                details={"category": "Subtitles"},
            )

    monkeypatch.setattr(processing, "process_video", process)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "--plain",
            "--no-color",
            "process",
            *map(str, videos),
            "--root",
            str(tmp_path),
            "--config",
            str(config),
            "--events-json",
            "events.jsonl",
        ],
    )
    assert result.exit_code == 1, result.exception
    assert seen == [(f"{name}.mp4", f"{name}.mp4") for name in "ABC"]
    assert "Videos: 2/3" in result.output and result.output.count("intentional failure") == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len({event["run_id"] for event in events}) == 1
    assert [e["video"] for e in events if e["stage"] == "video" and e["kind"] == "complete"] == [
        "A.mp4",
        "C.mp4",
    ]
    logs = list((tmp_path / "logs/runs").glob("*.jsonl"))
    assert len(logs) == 1 and "intentional failure" in logs[0].read_text(encoding="utf-8")
    assert current_context.get() == EventContext()


def test_process_batch_defers_missing_video_error_and_continues(tmp_path, monkeypatch):
    from transcript_video.application import processing
    from transcript_video.config import RunSettings

    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").touch()
    videos = [tmp_path / name for name in ("A.mp4", "Missing.mp4", "C.mp4")]
    videos[0].touch()
    videos[2].touch()
    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    settings.project.model = str(model)
    seen = []
    monkeypatch.setattr(processing, "process_video", lambda video, *args: seen.append(video.name))

    plan = processing.build_process_plan(settings, videos, defer_video_errors=True)
    summary = processing.execute_process_plan(plan)

    assert seen == ["A.mp4", "C.mp4"]
    assert summary.succeeded == 2 and summary.total == 3
    assert summary.failures[0].startswith("Missing.mp4: Video not found:")


def test_course_builder_emits_every_stage(tmp_path, monkeypatch):
    from transcript_video.course import builder
    from transcript_video.course.config import CourseConfig, SessionConfig

    videos = [tmp_path / f"{i}.mp4" for i in range(3)]
    for video in videos:
        video.touch()
    config = CourseConfig(
        "Course",
        tmp_path / "course.mp4",
        None,
        [SessionConfig(str(i), video, i + 1) for i, video in enumerate(videos)],
        work_dir=tmp_path / "work",
    )
    monkeypatch.setattr(builder, "get_media_duration_seconds", lambda path: 10.0)
    for name in (
        "normalize_session_video",
        "still_image_to_video",
        "render_session_card",
        "concatenate_videos",
        "add_chapter_metadata",
    ):
        monkeypatch.setattr(builder, name, Mock())
    monkeypatch.setattr(builder, "render_toc_pages", lambda *args: [tmp_path / "toc.png"])
    observer = RecordingObserver()
    builder.build_course(config, observer)
    completed = [e.stage.value for e in observer.events if e.kind == EventKind.COMPLETE]
    assert completed == ["validate", "normalize", "toc", "cards", "concatenate", "chapters", "run"]
    assert [
        e.current
        for e in observer.events
        if e.stage == PipelineStage.NORMALIZE and e.kind == EventKind.PROGRESS
    ] == [0, 1, 1, 2, 2, 3]
    artifact = next(e for e in observer.events if e.kind == EventKind.ARTIFACT)
    assert artifact.details["duration"] == 50 and artifact.details["sessions"] == 3


def test_textual_adapter_calls_main_thread_once():
    from transcript_video.tui.app import TextualObserver

    app, log = Mock(), Mock()
    observer = TextualObserver(app, log)
    event = PipelineEvent(PipelineStage.TOC, "Rendered", kind=EventKind.COMPLETE)
    observer.notify(event)
    app.call_from_thread.assert_called_once_with(
        log.write_line, f"{event.timestamp[11:19]} {format_pipeline_event(event)}"
    )


def test_windows_cp1252_symbols_never_break_processing():
    from io import BytesIO, TextIOWrapper

    buffer = BytesIO()
    stream = TextIOWrapper(buffer, encoding="cp1252")
    console = Console(file=stream, theme=THEME, no_color=True)
    observer = RichProgressObserver(console)
    observer.notify(PipelineEvent(PipelineStage.RUN, "Run", kind=EventKind.START, total=1))
    observer.notify(PipelineEvent(PipelineStage.VIDEO, "Video", kind=EventKind.COMPLETE))
    observer.summary(title="Complete", elapsed=1)
    stream.flush()
    assert b"OK" in buffer.getvalue()


def test_global_help_does_not_become_legacy_process_help():
    from transcript_video.cli import _normalize_legacy_argv

    assert _normalize_legacy_argv(["--no-color", "--plain", "--help"]) == [
        "--no-color",
        "--plain",
        "--help",
    ]


def test_important_warning_reaches_events_console_and_file_once(tmp_path):
    from transcript_video.events import warn

    stream = StringIO()
    console = Console(file=stream, theme=THEME, no_color=True)
    record = RecordingObserver()
    try:
        logs = configure_logging(console, 0, tmp_path / "global.log", run_id="r")
        with event_scope(
            CompositeObserver(RichProgressObserver(console), record), run_id="r", stage="render"
        ):
            warn(logging.getLogger("transcript_video.hardware"), "Fallback to %s", "CPU")
        assert stream.getvalue().count("Fallback to CPU") == 1
        assert record.events[0].kind == EventKind.WARNING
        entries = [json.loads(line) for line in logs.jsonl.read_text(encoding="utf-8").splitlines()]
        assert len([entry for entry in entries if entry["level"] == "WARNING"]) == 1
    finally:
        close_logging()


def test_missing_model_has_run_failure_event(tmp_path, monkeypatch):
    from transcript_video.cli import app

    video = tmp_path / "test.mp4"
    video.touch()
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "--no-color",
            "process",
            str(video),
            "--root",
            str(tmp_path),
            "--events-json",
            "events.jsonl",
        ],
    )
    assert result.exit_code == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["kind"] for event in events] == ["start", "failure"]
    assert result.output.count("Model folder not found") == 1


def test_subprocess_failure_retains_full_diagnostics_in_json_log(tmp_path):
    import sys

    from transcript_video.process_runner import ProcessExecutionError, run_process

    try:
        logs = configure_logging(
            Console(file=StringIO(), theme=THEME), 0, tmp_path / "global.log", run_id="r"
        )
        with pytest.raises(ProcessExecutionError, match="exited with code 7"):
            run_process(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('useful-stdout'); sys.stderr.write('first-line\\nlast-line'); sys.exit(7)",
                ]
            )
        data = [json.loads(line) for line in logs.jsonl.read_text(encoding="utf-8").splitlines()]
        failure = next(row for row in data if row["message"].startswith("Command failed:"))
        assert "returncode=7" in failure["message"]
        assert "useful-stdout" in failure["message"] and "first-line" in failure["message"]
    finally:
        close_logging()

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transcript_video.application.settings import resolve_settings
from transcript_video.cli import _normalize_legacy_argv, app
from transcript_video.events import PipelineEvent, PipelineStage, RecordingObserver
from transcript_video.process_runner import (
    ProcessExecutionError,
    parse_ffmpeg_progress,
    run_process,
)


def test_config_precedence_and_source_tracking(tmp_path: Path) -> None:
    config = tmp_path / "base.toml"
    config.write_text('[hardware]\ndevice = "cpu"\n', encoding="utf-8")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "gpu.toml").write_text('[hardware]\ndevice = "cuda"\n', encoding="utf-8")

    resolved = resolve_settings(
        config_path=config,
        profile="gpu",
        overrides={"hardware.device": "cpu"},
        require_config=True,
    )

    assert resolved.settings.hardware.device == "cpu"
    assert resolved.sources["hardware.device"] == "command line"


def test_recording_observer_preserves_semantic_event() -> None:
    observer = RecordingObserver()
    event = PipelineEvent(PipelineStage.RENDER, "Rendering")
    observer.notify(event)
    assert observer.events == [event]


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ({"out_time_us": "2500000", "progress": "continue"}, 2.5),
        ({"out_time": "01:02:03.5", "progress": "end"}, 3723.5),
        ({"out_time_us": "broken"}, None),
    ],
)
def test_ffmpeg_progress_parser(raw: dict[str, str], seconds: float | None) -> None:
    assert parse_ffmpeg_progress(raw).elapsed_seconds == seconds


def test_subprocess_runner_reports_stderr() -> None:
    with pytest.raises(ProcessExecutionError, match="intentional"):
        run_process(
            [sys.executable, "-c", "import sys; sys.stderr.write('intentional'); sys.exit(3)"]
        )


def test_cli_config_validate() -> None:
    result = CliRunner().invoke(
        app, ["config", "validate", "--config", "configs/transcription.toml"]
    )
    assert result.exit_code == 0
    assert "Valid configuration" in result.stdout


def test_legacy_cli_keeps_global_options_before_process() -> None:
    assert _normalize_legacy_argv(["--no-color", "-vv", "--video", "lesson.mp4"]) == [
        "--no-color",
        "-vv",
        "process",
        "--video",
        "lesson.mp4",
    ]


def test_dry_run_does_not_create_project_directories(tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.touch()
    result = CliRunner().invoke(
        app,
        ["process", str(video), "--root", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "Dry-run execution plan" in result.stdout
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_opens_metadata_screen() -> None:
    pytest.importorskip("textual")
    from transcript_video.tui.app import CourseApp, MetadataScreen

    async with CourseApp().run_test() as pilot:
        await pilot.pause()
        assert isinstance(pilot.app.screen, MetadataScreen)


@pytest.mark.parametrize(
    "names",
    [[], ["one.mp4"], ["three.mp4", "one.mp4", "two.mp4"], ["one.mp4", "one.mp4", "two.mp4"]],
)
def test_multi_video_cli_preserves_order_without_dry_run_writes(tmp_path, monkeypatch, names):
    from transcript_video.application import processing

    input_dir = tmp_path / "data/input"
    input_dir.mkdir(parents=True)
    for name in ("one.mp4", "two.mp4", "three.mp4"):
        (input_dir / name).touch()
    config = tmp_path / "run.toml"
    config.write_text("", encoding="utf-8")
    plans = []
    original = processing.build_process_plan

    def capture(*args):
        plan = original(*args)
        plans.append(plan)
        return plan

    monkeypatch.setattr(processing, "build_process_plan", capture)
    before = set(tmp_path.rglob("*"))
    result = CliRunner().invoke(
        app,
        ["process", *names, "--root", str(tmp_path), "--config", str(config), "--dry-run"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.exception
    expected = list(dict.fromkeys(names)) if names else ["one.mp4", "three.mp4", "two.mp4"]
    assert [path.name for path in plans[0].videos] == expected
    assert all(name in result.stdout for name in expected)
    assert set(tmp_path.rglob("*")) == before


def test_multi_absolute_paths_and_legacy_selection(tmp_path):
    from transcript_video.application.processing import build_process_plan
    from transcript_video.config import RunSettings

    first, second = tmp_path / "one video.mp4", tmp_path / "two.mp4"
    first.touch()
    second.touch()
    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    settings.project.video = str(first)
    assert build_process_plan(settings).videos == (first,)
    assert build_process_plan(settings, [second, first, second]).videos == (second, first)
    result = CliRunner().invoke(
        app,
        ["process", str(second), str(first), "--root", str(tmp_path), "--dry-run"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.exception
    assert "one video.mp4" in result.stdout and "two.mp4" in result.stdout
    legacy = CliRunner().invoke(
        app, ["process", "--video", str(first), "--root", str(tmp_path), "--dry-run"]
    )
    assert legacy.exit_code == 0
    assert not (tmp_path / "data").exists()


def test_multi_video_missing_and_conflicting_names_fail_before_writes(tmp_path):
    from transcript_video.application.processing import build_process_plan
    from transcript_video.config import RunSettings

    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    for folder in ("first", "second"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "same.mp4").touch()
    with pytest.raises(ValueError, match="share an output name"):
        build_process_plan(settings, [tmp_path / "first/same.mp4", tmp_path / "second/same.mp4"])
    result = CliRunner().invoke(
        app,
        [
            "process",
            str(tmp_path / "first/same.mp4"),
            str(tmp_path / "missing.mp4"),
            "--root",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "Video not found" in str(result.exception)
    assert not (tmp_path / "data").exists()


def test_explicit_video_list_is_executed_in_order(tmp_path, monkeypatch):
    from transcript_video.application import processing
    from transcript_video.config import RunSettings

    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    settings.project.model = "models/transcription"
    (tmp_path / settings.project.model).mkdir(parents=True)
    videos = [tmp_path / name for name in ("C.mp4", "A.mp4", "B.mp4")]
    for path in videos:
        path.touch()
    visited = []
    monkeypatch.setattr(processing, "process_video", lambda video, *args: visited.append(video))
    plan = processing.build_process_plan(settings, videos)
    result = processing.execute_process_plan(plan)
    assert visited == videos
    assert result.succeeded == result.total == 3
    assert plan.paths.report_dir.is_dir()


@pytest.mark.parametrize(
    "pin,current,ok,required",
    [
        ("3.14", (3, 14, 7), True, True),
        ("3.14", (3, 13, 9), False, True),
        ("3.99", (3, 99, 1), True, True),
        ("3.14.6", (3, 14, 7), False, True),
        ("invalid", (3, 14, 7), False, False),
        (b"\xff", (3, 14, 7), False, False),
        (None, (3, 14, 7), False, False),
    ],
)
def test_doctor_python_follows_project_pin(tmp_path, monkeypatch, pin, current, ok, required):
    from types import SimpleNamespace

    from transcript_video.application import diagnostics

    if pin is not None:
        (tmp_path / ".python-version").write_bytes(
            pin if isinstance(pin, bytes) else pin.encode("utf-8")
        )
    monkeypatch.setattr(
        diagnostics,
        "sys",
        SimpleNamespace(version=".".join(map(str, current)), version_info=current),
    )
    check = diagnostics._python_check(tmp_path)
    assert check.ok is ok and check.required is required
    assert ".python-version" in check.detail
    if required:
        assert f"requires {pin}" in check.detail


def test_doctor_checks_report_directory_without_creating_it(tmp_path, monkeypatch):
    from unittest import mock

    from transcript_video.application import diagnostics
    from transcript_video.config import RunSettings

    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    monkeypatch.setattr(
        diagnostics, "get_ffmpeg_exe", mock.Mock(side_effect=FileNotFoundError("test"))
    )
    monkeypatch.setattr(
        diagnostics, "get_ffprobe_exe", mock.Mock(side_effect=FileNotFoundError("test"))
    )
    monkeypatch.setitem(sys.modules, "torch", mock.Mock())
    checks = diagnostics.run_doctor(settings)
    check = next(item for item in checks if item.name == "Writable report_dir")
    assert check.ok
    assert Path(check.detail) == tmp_path / "data/report"
    assert not (tmp_path / "data").exists()

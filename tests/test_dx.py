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

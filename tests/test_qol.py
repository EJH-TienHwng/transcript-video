import json
import sys
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from transcript_video.application.settings import resolve_settings
from transcript_video.cli import app
from transcript_video.course.config import load_course_config, save_course_config


def test_course_roundtrip_preserves_unknown_fields(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    path = tmp_path / "course.json"
    raw = dict(
        title="Course",
        output="course.mp4",
        future={"x": 1},
        sessions=[dict(title="A", number=4, video="a.mp4", note="keep")],
        render={"future": "keep"},
        toc={"future": 9},
    )
    save_course_config(raw, path)
    config = load_course_config(path)
    config.title = "Edited"
    config.sessions[0].video = tmp_path / "replacement.mp4"
    save_course_config(config, path)
    result = json.loads(path.read_text())
    assert result["future"] == raw["future"]
    assert result["render"]["future"] == "keep"
    assert result["toc"]["future"] == 9
    assert result["sessions"][0]["note"] == "keep"
    assert load_course_config(path) == config
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_course_config({"sessions": []}, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("source", ["config", "profile"])
def test_legacy_hardware_settings_are_normalized(tmp_path, source):
    path = tmp_path / "old.toml"
    path.write_text('[transcription]\ndevice="cpu"\ncompute_type="float32"\n')
    resolved = resolve_settings(
        **({"config_path": path} if source == "config" else {"config_path": None, "profile": path})
    )
    for key in ("device", "compute_type"):
        assert resolved.sources[f"hardware.{key}"] == source
        assert f"transcription.{key}" not in resolved.sources


@pytest.mark.parametrize("target", ["transcription", "tts", "all", "translation", "render"])
def test_only_transcription_is_a_force_target(tmp_path, monkeypatch, target):
    from transcript_video import cli

    captured = Mock()
    monkeypatch.setattr(cli, "_run_processing", captured)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "process",
            "--dry-run",
            "--force",
            target,
            "--no-overwrite-srt",
        ],
    )
    if target != "transcription":
        assert result.exit_code == 2
        assert not captured.called
    else:
        assert result.exit_code == 0, result.exception
        settings = captured.call_args.args[1]
        assert settings.transcription.overwrite_srt


@pytest.mark.parametrize("missing", [None, "video", "model"])
def test_dry_run_validates_without_side_effects(tmp_path, monkeypatch, missing):
    from transcript_video.application import processing

    monkeypatch.chdir(tmp_path)
    video = tmp_path / "a.mp4"
    model = tmp_path / "asr"
    if missing != "video":
        video.touch()
    if missing != "model":
        model.mkdir()
        (model / "model.bin").touch()
    execute = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(processing, "execute_process_plan", execute)
    before = set(tmp_path.rglob("*"))
    result = CliRunner().invoke(
        app,
        [
            "process",
            str(video),
            "--model",
            str(model),
            "--dry-run",
            "--events-json",
            "events/e.jsonl",
            "--save-config",
            "save.toml",
        ],
    )
    assert (result.exit_code == 0) == (missing is None), result.exception
    assert not execute.called
    assert set(tmp_path.rglob("*")) == before


def test_inspection_independent_of_models(tmp_path, monkeypatch):
    from transcript_video.application import inspection
    from transcript_video.config import RunSettings

    video = tmp_path / "a.mp4"
    video.touch()
    monkeypatch.setattr(inspection, "get_ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(inspection, "probe_media", lambda *a, **kw: {"format": {"duration": "10"}})
    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    result = inspection.inspect_video(video, settings)
    assert result["metadata"]["format"]["duration"] == "10"
    assert result["artifact_states"]["source_subtitles"] == "unresolved"
    assert result["artifact_states"]["translated_subtitles"] == "missing"
    assert result["workflow"] == "translation_handoff"
    assert result["artifact_states"]["tts_audio"] == "missing"


@pytest.mark.parametrize("module,command", [("cli", "build"), ("tui_cli", "create")])
def test_deprecated_wrappers_are_visible(module, command, monkeypatch, capsys):
    import importlib

    from transcript_video import cli

    monkeypatch.setattr(sys, "argv", ["legacy", "--no-color", "--plain"])
    delegate = Mock()
    monkeypatch.setattr(cli, "main", delegate)
    importlib.import_module(f"transcript_video.course.{module}").main()
    assert "deprecated" in capsys.readouterr().err
    assert sys.argv[1:] == ["--no-color", "--plain", "course", command]
    delegate.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [["--help"], ["process", "--help"], ["course", "--help"], ["config", "--help"], ["--version"]],
)
def test_cli_help_and_version_no_ansi(args):
    result = CliRunner().invoke(app, ["--no-color", *args], env={"NO_COLOR": "1"})
    assert result.exit_code == 0
    assert "\x1b" not in result.output


@pytest.mark.parametrize("invalid", [False, True])
def test_config_validate_json(tmp_path, invalid):
    path = tmp_path / "run.toml"
    path.write_text('[hardware]\ndevice="gpu123"' if invalid else "")
    result = CliRunner().invoke(app, ["--json", "config", "validate", "--config", str(path)])
    assert json.loads(result.stdout)["valid"] is not invalid
    assert result.exit_code == (1 if invalid else 0)


def test_structured_process_error_and_timeout(monkeypatch):
    import subprocess

    from transcript_video import process_runner as runner

    stderr = "first\n" * 500
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess(["tool"], 7, "full stdout", stderr)),
    )
    with pytest.raises(runner.ProcessExecutionError) as caught:
        runner.run_process(["custom-probe", "argument"], tool="ffprobe")
    error = caught.value
    assert error.command == ("custom-probe", "argument")
    assert error.returncode == 7 and error.stderr == stderr and error.stdout == "full stdout"
    assert str(error) == "ffprobe exited with code 7"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        Mock(side_effect=subprocess.TimeoutExpired(["tool"], 2, output=b"out", stderr=b"err")),
    )
    with pytest.raises(runner.ProcessExecutionError) as caught:
        runner.run_process(["tool"], timeout=2)
    assert (
        caught.value.timeout == 2 and caught.value.stdout == "out" and caught.value.stderr == "err"
    )
    assert caught.value.returncode is None


def test_structured_ffmpeg_failure(monkeypatch):
    from transcript_video import process_runner as runner

    process = Mock(stdout=iter(["out_time_us=100\n", "progress=end\n"]), returncode=1)

    def popen(*args, **kwargs):
        kwargs["stderr"].write("complete diagnostic\n" * 100)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(runner.ProcessExecutionError) as caught:
        runner.run_ffmpeg(["custom-ffmpeg", "-i", "video"])
    assert caught.value.tool == "FFmpeg"
    assert str(caught.value) == "FFmpeg exited with code 1"
    assert caught.value.stderr.count("diagnostic") == 100
    assert "progress=end" in caught.value.stdout


def test_reused_event_json_rich_and_plain(tmp_path):
    from io import StringIO

    from rich.console import Console

    from transcript_video.events import EventKind, JsonEventObserver, PipelineEvent, PipelineStage
    from transcript_video.ui.progress import RichProgressObserver
    from transcript_video.ui.theme import THEME

    event = PipelineEvent(PipelineStage.SUBTITLES, "Cached subtitles", kind=EventKind.REUSED)
    stream = StringIO()
    observer = RichProgressObserver(Console(file=stream, theme=THEME, no_color=True), plain=True)
    observer.notify(event)
    assert observer.state.stages["subtitles"].status == "reused"
    assert "Cached subtitles" in stream.getvalue()
    path = tmp_path / "events.jsonl"
    writer = JsonEventObserver(path)
    writer.notify(event)
    writer.close()
    assert json.loads(path.read_text())["kind"] == "reused"


def test_doctor_reports_configured_encoder_separately(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from transcript_video.application import diagnostics
    from transcript_video.config import RunSettings

    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    settings.hardware.device = "cpu"
    settings.hardware.video_encoder = "h264_nvenc"
    monkeypatch.setattr(diagnostics, "get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(diagnostics, "get_ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(
        diagnostics,
        "run_process",
        lambda *a, **kw: SimpleNamespace(stdout="subtitles h264_nvenc libx264"),
    )
    monkeypatch.setattr(
        diagnostics, "ffmpeg_encoder_available", lambda exe, encoder: encoder == "libx264"
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_supported_compute_types=lambda device: {"float32"}),
    )
    checks = {check.name: check for check in diagnostics.run_doctor(settings)}
    assert not checks["Configured encoder"].ok and checks["Configured encoder"].required
    assert checks["Fallback encoder"].ok
    assert checks["NVENC encoder availability"].ok and not checks["NVENC runtime usability"].ok
    assert not checks["CUDA"].required
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("json_mode", [False, True])
def test_inspect_cli_human_and_json_without_models(tmp_path, monkeypatch, json_mode):
    from transcript_video.application import inspection

    monkeypatch.chdir(tmp_path)
    video = tmp_path / "video.mp4"
    video.touch()
    monkeypatch.setattr(inspection, "get_ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(
        inspection,
        "probe_media",
        lambda *a, **kw: {
            "format": {"duration": "90", "size": "1048576"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "codec_name": "h264",
                    "avg_frame_rate": "30/1",
                }
            ],
        },
    )
    result = CliRunner().invoke(app, (["--json"] if json_mode else []) + ["inspect", str(video)])
    assert result.exit_code == 0, result.exception
    if json_mode:
        payload = json.loads(result.stdout)
        assert payload["artifact_states"]["source_subtitles"] == "unresolved"
        assert payload["artifact_states"]["translated_subtitles"] == "missing"
    else:
        assert "01:30" in result.stdout and "Resolution" in result.stdout
        assert '"streams"' not in result.stdout


def test_metadata_cache_invalidates_replaced_file(tmp_path, monkeypatch):
    from transcript_video.application import inspection

    video = tmp_path / "a.mp4"
    video.touch()
    probe = Mock(return_value={"format": {}})
    monkeypatch.setattr(inspection, "probe_media", probe)
    monkeypatch.setattr(inspection, "get_ffprobe_exe", lambda: "ffprobe")
    cache = {}
    inspection.read_media_metadata(video, cache)
    inspection.read_media_metadata(video, cache)
    video.write_bytes(b"changed")
    inspection.read_media_metadata(video, cache)
    assert probe.call_count == 2 and len(cache) == 1


def test_main_unknown_option_exits_two_without_traceback(monkeypatch, capsys):
    from transcript_video import cli

    monkeypatch.setattr(
        sys, "argv", ["transcript-video", "--no-color", "process", "--unknown-option"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    output = capsys.readouterr().err
    assert "No such option" in output and "Traceback" not in output
    assert "\x1b" not in output


def test_tui_no_color_is_applied_during_construction(tmp_path, monkeypatch):
    import os

    from transcript_video.tui import app as tui

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NO_COLOR", raising=False)
    states = []

    class App:
        def __init__(self, **kwargs):
            states.append(os.environ.get("NO_COLOR"))

        def run(self):
            return None

    monkeypatch.setattr(tui, "CourseApp", App)
    result = CliRunner().invoke(app, ["--no-color", "course", "tui"])
    assert result.exit_code == 0 and states == ["1"]
    assert "NO_COLOR" not in os.environ


@pytest.mark.parametrize("code,output", [(1, ""), (0, "[]"), (0, "broken")])
def test_ffprobe_failure_keeps_tool_and_output(monkeypatch, code, output):
    import subprocess
    from pathlib import Path

    from transcript_video import process_runner as runner

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess(["probe"], code, output, "details")),
    )
    with pytest.raises(runner.ProcessExecutionError) as caught:
        runner.probe_media("probe", Path("video"))
    assert caught.value.tool == "ffprobe"
    assert caught.value.stdout == output and caught.value.stderr == "details"


@pytest.mark.parametrize("mode", ["--no-color", "NO_COLOR", "piped"])
def test_main_help_honors_no_color_and_redirection(monkeypatch, capsys, mode):
    from typer import rich_utils

    from transcript_video import cli

    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["transcript-video", *(["--no-color"] if mode == "--no-color" else []), "--help"],
    )
    if mode == "NO_COLOR":
        monkeypatch.setenv("NO_COLOR", "1")
    cli.main()
    assert "\x1b" not in capsys.readouterr().out
    assert rich_utils.FORCE_TERMINAL is True

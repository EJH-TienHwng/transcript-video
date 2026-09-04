from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from transcript_video.cli import parse_args
from transcript_video.config import (
    RunSettings,
    SubtitleSegment,
    load_run_settings,
    save_run_settings,
)
from transcript_video.course.config import CourseConfig, load_course_config
from transcript_video.hardware import get_ffmpeg_exe, video_encoder_args
from transcript_video.processing.models import detect_model_type, read_transformers_model_config
from transcript_video.processing.subtitles import (
    parse_srt_timestamp,
    read_srt,
    remove_repeated_hallucination_segments,
)


def test_detects_sharded_transformers_weights(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    assert detect_model_type(tmp_path) == "huggingface"


def test_reports_invalid_model_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read model config"):
        read_transformers_model_config(tmp_path)


@pytest.mark.parametrize("timestamp", ["00:60:00,000", "00:00:60,000"])
def test_rejects_out_of_range_timestamp(timestamp: str) -> None:
    with pytest.raises(ValueError):
        parse_srt_timestamp(timestamp)


def test_reads_optional_srt_positioning_metadata(tmp_path: Path) -> None:
    srt = tmp_path / "positioned.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,500 position:50%\nHello\n", encoding="utf-8")
    assert read_srt(srt) == [SubtitleSegment(1.0, 2.5, "Hello")]


def test_only_filters_consecutive_repetitions() -> None:
    segments = [
        SubtitleSegment(i, i + 1, text)
        for i, text in enumerate(["Again", "Other", "Again", "Again", "Again", "Again"])
    ]
    cleaned = remove_repeated_hallucination_segments(segments, max_same_text_count=3)
    assert [item.text for item in cleaned] == ["Again", "Other", "Again", "Again", "Again"]


def _course_config(root: Path, overrides: dict[str, object]) -> Path:
    raw: dict[str, object] = {
        "sessions": [{"number": 1, "title": "Session", "video": "data/input.mp4"}]
    }
    raw.update(overrides)
    folder = root / "courses"
    folder.mkdir()
    path = folder / "course.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"add_chapters": "false"}, "add_chapters"),
        ({"output": "data/input.mp4"}, "overwrite"),
        ({"render": {"video_encoder": "unknown"}}, "render.video_encoder"),
    ],
)
def test_rejects_invalid_course_config(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_course_config(_course_config(tmp_path, overrides))


def test_course_cards_use_requested_background_and_text_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transcript_video.course import cards
    from transcript_video.course.config import RenderConfig, SessionConfig
    from transcript_video.course.timeline import SessionTimeline

    project_root = Path.cwd()
    session = SessionConfig("Section title", tmp_path / "session.mp4", 1)
    config = CourseConfig(
        title="Course",
        output=tmp_path / "course.mp4",
        theme_image=project_root / "assets/bosch_theme.png",
        sessions=[session],
        render=RenderConfig(width=320, height=180),
        work_dir=project_root / "data/compilation",
    )
    timeline = [SessionTimeline(session, 10, 5, 10, 20)]
    opened: list[Path] = []
    draws = []
    real_open = cards.Image.open
    real_draw = cards.ImageDraw.Draw

    def tracked_open(path):
        opened.append(Path(path))
        return real_open(path)

    def tracked_draw(image):
        draw = mock.Mock(wraps=real_draw(image))
        draws.append(draw)
        return draw

    monkeypatch.setattr(cards.Image, "open", tracked_open)
    monkeypatch.setattr(cards.ImageDraw, "Draw", tracked_draw)

    cards.render_toc_pages(config, timeline, tmp_path / "toc")
    cards.render_session_card(config, timeline[0], 1, tmp_path / "section.png")

    assert opened[0].resolve() == (project_root / "assets/table_of_content.png").resolve()
    assert {call.kwargs["fill"] for call in draws[0].text.call_args_list} == {"black"}
    assert any(
        call.args[1] in {"Section title", "Section", "title"} and call.kwargs["fill"] == "white"
        for call in draws[1].text.call_args_list
    )


def test_normalization_uses_source_video_duration(tmp_path: Path) -> None:
    from transcript_video.course import media

    source = tmp_path / "input.mp4"
    source.touch()
    config = CourseConfig("Course", tmp_path / "course.mp4", None, [])
    with (
        mock.patch.object(media, "get_media_duration_seconds", return_value=12.5),
        mock.patch.object(media, "media_has_audio", return_value=True),
        mock.patch.object(media, "get_ffmpeg_exe", return_value="ffmpeg"),
        mock.patch.object(media, "video_encoder_args", return_value=["-c:v", "h264_nvenc"]),
        mock.patch.object(media, "run_command") as runner,
    ):
        media.normalize_session_video(source, tmp_path / "output.mp4", config)
    command = runner.call_args.args[0]
    assert command[command.index("-t") + 1] == "12.500"
    assert "h264_nvenc" in command


def test_explicit_ffmpeg_path_has_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.touch()
    monkeypatch.setenv("TRANSCRIPT_VIDEO_FFMPEG", str(ffmpeg))
    assert get_ffmpeg_exe() == str(ffmpeg.resolve())


@pytest.mark.parametrize(("available", "expected"), [(True, "h264_nvenc"), (False, "libx264")])
def test_auto_video_encoder_selection(available: bool, expected: str) -> None:
    with mock.patch("transcript_video.hardware.ffmpeg_encoder_available", return_value=available):
        args = video_encoder_args("ffmpeg", "auto", bitrate="8M" if available else None)
    assert args[1] == expected


def test_saved_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    settings = RunSettings.defaults()
    settings.project.video = "lesson.mp4"
    settings.tts.enabled = True
    save_run_settings(settings, path)
    assert load_run_settings(path) == settings


def test_cli_values_override_saved_config(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    save_run_settings(RunSettings.defaults(), path)
    _, settings = parse_args(["--config", str(path), "--device", "cpu", "--enable-tts"])
    assert settings.hardware.device == "cpu" and settings.tts.enabled


def test_rejects_invalid_toml_value_types(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text('[tts]\nchunk_minutes = "five"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"tts\.chunk_minutes must be an integer"):
        parse_args(["--config", str(path)])


def test_allows_empty_language_for_auto_detection() -> None:
    assert parse_args(["--language", ""])[1].transcription.language == ""


def test_migrates_legacy_hardware_settings(tmp_path: Path) -> None:
    path = tmp_path / "legacy.toml"
    path.write_text('[transcription]\ndevice = "cpu"\ncompute_type = "int8"\n', encoding="utf-8")
    settings = load_run_settings(path)
    assert (settings.hardware.device, settings.hardware.compute_type) == ("cpu", "int8")


def test_rejects_unsupported_compute_type(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text('[hardware]\ncompute_type = "fastest"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"hardware\.compute_type"):
        parse_args(["--config", str(path)])

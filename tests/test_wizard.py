from __future__ import annotations

from dataclasses import asdict
from io import StringIO
from unittest.mock import Mock

import pytest
from rich.console import Console

from transcript_video.course import wizard
from transcript_video.course.config import RenderConfig, TocConfig, load_course_config
from transcript_video.ui.theme import THEME, questionary_style


@pytest.mark.parametrize("count", [1, 3, 8])
def test_multiselect_auto_titles_numbers_and_metadata_cache(tmp_path, monkeypatch, count):
    videos = [tmp_path / f"{i:02d}_test-en-sub.mp4" for i in range(count)]
    for video in videos:
        video.touch()
    probes = Mock(
        return_value={
            "format": {"duration": "90"},
            "streams": [dict(codec_type="video", width=1920, height=1080, codec_name="h264")],
        }
    )
    monkeypatch.setattr(wizard, "probe_media", probes)
    monkeypatch.setattr(wizard, "get_ffprobe_exe", lambda: "ffprobe")
    monkeypatch.setattr(wizard, "_ask", lambda *args, **kwargs: videos)
    cache = {}
    with wizard.wizard_ui(Console(file=StringIO(), theme=THEME, no_color=True)):
        sessions = wizard._collect_sessions(tmp_path, tmp_path, cache)
    assert [s["number"] for s in sessions] == list(range(1, count + 1))
    assert [s["video"] for s in sessions] == [p.name for p in videos]
    assert [s["title"] for s in sessions] == [wizard._default_title_from_video(p) for p in videos]
    for video in videos:
        assert wizard._metadata(video, cache)["duration"] == 90
    assert probes.call_count == count


def test_manual_video_and_probe_failure_cached(tmp_path, monkeypatch):
    video = tmp_path / "manual.mp4"
    video.touch()
    answers = iter([["__manual__"], False])
    monkeypatch.setattr(wizard, "_scan_videos", lambda folder: [])
    monkeypatch.setattr(wizard, "_ask", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(wizard, "_ask_text", lambda *args: str(video))
    monkeypatch.setattr(wizard, "get_ffprobe_exe", lambda: "ffprobe")
    probe = Mock(side_effect=ValueError("bad metadata"))
    monkeypatch.setattr(wizard, "probe_media", probe)
    cache = {}
    sessions = wizard._collect_sessions(tmp_path, tmp_path, cache)
    assert sessions == [dict(number=1, title="manual", video="manual.mp4")]
    assert wizard._metadata(video, cache) == {}
    assert probe.call_count == 1


@pytest.mark.parametrize("width", [80, 120])
def test_review_width_and_no_color_palette(width):
    output = StringIO()
    console = Console(file=output, width=width, theme=THEME, no_color=True, force_terminal=True)
    previous = wizard.console
    with wizard.wizard_ui(console):
        wizard._step(2)
        wizard._review_sessions([dict(number=1, title="[Title]", video="data/output/intro.mp4")])
        assert all(not style for _, style in wizard.PROMPT_STYLE.style_rules)
    assert wizard.console is previous
    text = output.getvalue()
    assert "\x1b" not in text
    assert "[Title]" in text and "STATUS" in text and "intro.mp4" in text
    assert max(map(len, text.splitlines())) <= width
    assert all(not value for _, value in questionary_style(True).style_rules)


def test_remove_defaults_to_keep_and_duplicate_number_reprompts(monkeypatch):
    sessions = [dict(number=i, title=str(i), video=f"{i}.mp4") for i in (1, 2)]
    answers = iter(["Remove", 0, "Keep", "Edit title/number", 0, "Continue"])
    numbers = iter([2, 3])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **kw: next(answers))
    monkeypatch.setattr(wizard, "_ask_int", lambda *a, **kw: next(numbers))
    monkeypatch.setattr(wizard, "_ask_text", lambda *a, **kw: "Edited")
    assert len(wizard._edit_sessions(sessions)) == 2
    assert sessions[0]["number"] == 3


def test_summary_duration_includes_toc_pages_and_cards(tmp_path):
    videos = [tmp_path / f"{i}.mp4" for i in range(3)]
    config = dict(
        title="Test",
        sessions=[
            dict(number=i + 1, title=str(i), video=str(video)) for i, video in enumerate(videos)
        ],
        output="data/compilation/test.mp4",
        toc={**asdict(TocConfig()), "items_per_page": 2},
        render=asdict(RenderConfig()),
        theme_image=None,
        card_duration=5.0,
        add_chapters=True,
    )
    cache = {video: {"duration": 60} for video in videos}
    summary = wizard._summary(config, tmp_path / "test.json", tmp_path, cache)
    assert summary["Source duration"] == "03:00"
    assert summary["Estimated total"] == "03:25"
    cache[videos[0]] = {}
    assert (
        wizard._summary(config, tmp_path / "test.json", tmp_path, cache)["Estimated total"]
        == "Unavailable"
    )


@pytest.mark.parametrize("decision", ["Create configuration", "Cancel"])
def test_final_review_before_any_write_and_config_loads(tmp_path, monkeypatch, decision):
    (tmp_path / "pyproject.toml").touch()
    video = tmp_path / "intro.mp4"
    video.touch()
    monkeypatch.setattr(
        wizard, "_collect_sessions", lambda *a: [dict(number=1, title="Intro", video="intro.mp4")]
    )
    monkeypatch.setattr(wizard, "_metadata", lambda *a: {"duration": 10})
    monkeypatch.setattr(wizard, "_ask_text", lambda message, default=None: default)

    def ask(kind, message, **kwargs):
        if message == "Review session list:":
            return "Continue"
        if message == "Setup mode:":
            return "Recommended"
        if message == "Theme:":
            return "__none__"
        if message == "Final review:":
            assert not (tmp_path / "configs").exists()
            return decision
        pytest.fail(message)

    monkeypatch.setattr(wizard, "_ask", ask)
    result = wizard.create_course_config_interactive(tmp_path, tmp_path)
    if decision == "Cancel":
        assert result is None and not (tmp_path / "configs").exists()
    else:
        config = load_course_config(result)
        assert config.render == RenderConfig() and config.toc.enabled and config.add_chapters
        assert config.sessions[0].video == video


def test_custom_output_and_overwrite_decline_preserves_existing(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "intro.mp4").touch()
    existing = tmp_path / "configs/courses/training_course.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        wizard, "_collect_sessions", lambda *a: [dict(number=1, title="Intro", video="intro.mp4")]
    )
    monkeypatch.setattr(wizard, "_metadata", lambda *a: {})
    monkeypatch.setattr(wizard, "_ask_text", lambda message, default=None: default)
    monkeypatch.setattr(
        wizard,
        "_ask_int",
        lambda message, default, *a: (
            1280 if message == "Width:" else 720 if message == "Height:" else default
        ),
    )
    monkeypatch.setattr(wizard, "_ask_float", lambda message, default, *a: default)
    decisions = iter(["Create configuration", "Cancel"])

    def ask(kind, message, **kwargs):
        if message == "Review session list:":
            return "Continue"
        if message == "Setup mode:":
            return "Custom"
        if message == "Theme:":
            return "__none__"
        if message == "Final review:":
            return next(decisions)
        if message.startswith("Overwrite"):
            return False
        if kind == "confirm":
            return False
        if message == "Video encoder:":
            return "libx264"
        pytest.fail(message)

    monkeypatch.setattr(wizard, "_ask", ask)
    assert wizard.create_course_config_interactive(tmp_path, tmp_path) is None
    assert existing.read_text(encoding="utf-8") == "original"


def test_invalid_theme_reprompts_then_accepts_supported_image(tmp_path, monkeypatch):
    from PIL import Image

    broken = tmp_path / "bad.png"
    broken.write_text("bad", encoding="utf-8")
    valid = tmp_path / "valid.png"
    Image.new("RGB", (20, 10)).save(valid)
    paths = iter([str(broken), str(valid)])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **kw: "__manual__")
    monkeypatch.setattr(wizard, "_ask_text", lambda *a, **kw: next(paths))
    assert wizard._choose_theme(tmp_path) == "valid.png"


@pytest.mark.integration
def test_real_questionary_keyboard_flow(tmp_path, monkeypatch):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "intro.mp4").touch()
    monkeypatch.setattr(wizard, "get_ffprobe_exe", Mock(side_effect=FileNotFoundError("ffprobe")))
    with create_pipe_input() as pipe:
        for kind in ("text", "select", "checkbox", "confirm"):
            original = getattr(wizard.questionary, kind)

            def factory(message, *, _kind=kind, _original=original, **kwargs):
                keys = " \r" if _kind == "checkbox" else "\x1b[B\r" if message == "Theme:" else "\r"
                pipe.send_text(keys)
                return _original(message, input=pipe, output=DummyOutput(), **kwargs)

            monkeypatch.setattr(wizard.questionary, kind, factory)
        with wizard.wizard_ui(Console(file=StringIO(), no_color=True, theme=THEME, width=80)):
            path = wizard.create_course_config_interactive(tmp_path, tmp_path)
    assert load_course_config(path).sessions[0].title == "intro"


def test_add_from_review_rejects_duplicate_and_assigns_unused_number(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.touch()
    second.touch()
    sessions = [dict(number=5, title="First", video="first.mp4")]
    answers = iter(["Add session", [first], [second], "Continue"])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **kw: next(answers))
    monkeypatch.setattr(wizard, "_metadata", lambda *a: {})
    result = wizard._edit_sessions(sessions, tmp_path, tmp_path, {})
    assert [item["number"] for item in result] == [5, 6]
    from transcript_video.course.config import save_course_config

    path = save_course_config(dict(sessions=result), tmp_path / "course.json")
    assert len(load_course_config(path).sessions) == 2

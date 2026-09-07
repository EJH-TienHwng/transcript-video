from copy import deepcopy

import pytest
from textual.widgets import Button, Input, ListView

from transcript_video.tui.app import CourseApp, SessionScreen


@pytest.fixture
def course(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    app = CourseApp(tmp_path / "course.json")
    for index in range(3):
        video = tmp_path / f"{index}.mp4"
        video.touch()
        app.draft.sessions.append(dict(number=index + 1, title=str(index), video=str(video)))
    monkeypatch.setattr(SessionScreen, "read_metadata", lambda *a: None)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["save", "cancel", "invalid", "back", "reorder", "remove"])
async def test_edit_is_transactional(course, action):
    original = deepcopy(course.draft.sessions)
    async with course.run_test() as pilot:
        course.push_screen(SessionScreen())
        await pilot.pause()
        screen = course.screen
        screen.query_one("#sessions", ListView).index = 1
        screen.action_edit()
        assert course.draft.sessions == original
        assert str(screen.query_one("#add", Button).label) == "Save changes"
        screen.query_one("#session-title", Input).value = "Changed"
        if action == "save":
            screen.action_add()
            assert course.draft.sessions[1] == {**original[1], "title": "Changed"}
            assert course.draft.sessions[0] == original[0]
            assert course.draft.sessions[2] == original[2]
        else:
            if action == "cancel":
                screen.action_cancel_edit()
            elif action == "invalid":
                screen.query_one("#session-video", Input).value = "missing.mp4"
                screen.action_add()
            elif action == "back":
                screen.action_back()
            elif action == "reorder":
                screen.action_up()
            else:
                screen.action_remove()
            assert course.draft.sessions == original


def test_corrupted_config_is_reported_and_preserved(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON") as error:
        CourseApp(path)
    assert str(path) in str(error.value)
    assert path.read_text() == "{broken"


@pytest.mark.asyncio
async def test_save_reload_and_invalid_save_preserves_file(course):
    async with course.run_test() as pilot:
        await pilot.pause()
        assert course.save_draft()
        before = course.draft.config_path.read_bytes()
        assert CourseApp(course.draft.config_path).draft.sessions == course.draft.sessions
        course.draft.output = "bad.wav"
        assert not course.save_draft()
        assert course.draft.config_path.read_bytes() == before


def test_missing_config_creates_draft_without_writing(tmp_path):
    path = tmp_path / "new.json"
    assert CourseApp(path).draft.sessions == []
    assert not path.exists()


@pytest.mark.asyncio
async def test_browser_add_duplicate_remove_and_reorder(course, monkeypatch):
    from textual.widgets import DataTable

    root = course.project_root
    folder = root / "data/input"
    folder.mkdir(parents=True)
    video = folder / "new_lesson.mp4"
    video.touch()
    (folder / "hidden.txt").touch()
    async with course.run_test(size=(140, 50)) as pilot:
        course.push_screen(SessionScreen())
        await pilot.pause()
        await course.workers.wait_for_complete()
        screen = course.screen
        table = screen.query_one("#videos", DataTable)
        assert table.row_count == 1
        table.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert screen.query_one("#session-video", Input).value == str(video)
        await pilot.click("#add")
        assert course.draft.sessions[-1]["title"] == "new lesson"
        table.focus()
        await pilot.press("enter")
        await pilot.click("#add")
        assert len(course.draft.sessions) == 4
        screen.query_one("#sessions", ListView).index = 3
        screen.action_up()
        await pilot.pause()
        assert course.draft.sessions[2]["title"] == "new lesson"
        screen.action_down()
        await pilot.pause()
        assert course.draft.sessions[3]["title"] == "new lesson"
        screen.action_remove()
        assert len(course.draft.sessions) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_metadata_worker_result_and_failure(tmp_path, monkeypatch, failure):
    import threading

    from textual.widgets import Static

    from transcript_video.application import inspection

    (tmp_path / "pyproject.toml").touch()
    video = tmp_path / "a.mp4"
    video.touch()
    calls = []

    def probe(*args, **kwargs):
        calls.append(threading.get_ident())
        if failure:
            raise RuntimeError("bad media")
        return {
            "format": {"duration": "90"},
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080, "codec_name": "h264"}
            ],
        }

    monkeypatch.setattr(inspection, "probe_media", probe)
    monkeypatch.setattr(inspection, "get_ffprobe_exe", lambda: "ffprobe")
    app = CourseApp(tmp_path / "course.json")
    async with app.run_test() as pilot:
        app.push_screen(SessionScreen())
        await pilot.pause()
        screen = app.screen
        for _ in range(2):
            screen.read_metadata(str(video))
            await app.workers.wait_for_complete()
            await pilot.pause()
        text = str(screen.query_one("#metadata", Static).render())
        assert ("bad media" if failure else "01:30") in text
        assert calls == [calls[0]] and calls[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_advanced_settings_domain_validation_and_roundtrip(course):
    from textual.widgets import Checkbox

    async with course.run_test() as pilot:
        await pilot.pause()
        screen = course.screen
        screen.query_one("#render-width", Input).value = "1281"
        assert not screen.capture()
        assert "render" not in course.draft.extra
        screen.query_one("#render-width", Input).value = "1280"
        screen.query_one("#render-height", Input).value = "720"
        screen.query_one("#toc-heading", Input).value = "Lessons"
        screen.query_one("#toc-enabled", Checkbox).value = False
        assert screen.capture() and course.save_draft()
        loaded = CourseApp(course.draft.config_path).draft
        assert loaded.extra["render"]["width"] == 1280
        assert loaded.extra["toc"]["heading"] == "Lessons"
        assert not loaded.extra["toc"]["enabled"]


@pytest.mark.asyncio
async def test_review_real_progress_and_failure(course, monkeypatch):
    import asyncio

    from textual.widgets import Log, ProgressBar, Static

    from transcript_video.events import EventKind, PipelineEvent, PipelineStage
    from transcript_video.tui import app as tui

    monkeypatch.setattr(
        tui, "inspect_video", lambda *a: {"metadata": {"format": {"duration": "60"}}}
    )
    async with course.run_test(size=(120, 50)) as pilot:
        course.push_screen(tui.ReviewScreen())
        await pilot.pause()
        await course.workers.wait_for_complete()
        screen = course.screen
        summary = str(screen.query_one("#review-summary", Static).render())
        assert "03:00" in summary and "03:20" in summary
        assert all(str(item["video"]) in summary for item in course.draft.sessions)
        observer = tui.TextualObserver(
            course, screen.query_one("#build-log", Log), screen.apply_progress
        )

        async def send(stage, kind, **kwargs):
            await asyncio.to_thread(
                observer.notify, PipelineEvent(stage, "work", kind=kind, **kwargs)
            )

        await send(PipelineStage.RUN, EventKind.START, details={"stages": ["normalize", "cards"]})
        await send(PipelineStage.NORMALIZE, EventKind.START)
        assert screen.query_one("#stage-progress", ProgressBar).total is None
        await send(PipelineStage.NORMALIZE, EventKind.PROGRESS, current=2, total=3)
        assert screen.query_one("#stage-progress", ProgressBar).progress == 2
        assert "2 / 3" in str(screen.query_one("#session-count", Static).render())
        await send(PipelineStage.NORMALIZE, EventKind.REUSED)
        assert screen.query_one("#overall-progress", ProgressBar).progress == 1
        await send(PipelineStage.CARDS, EventKind.FAILURE)
        assert screen.query_one("#stage-progress", ProgressBar).progress == 0
        assert screen.query_one("#overall-progress", ProgressBar).progress == 1


@pytest.mark.asyncio
async def test_unsaved_quit_confirmation(course):
    from transcript_video.tui.app import ConfirmQuit

    course.draft.dirty = True
    async with course.run_test() as pilot:
        await pilot.pause()
        course.request_quit()
        await pilot.pause()
        assert isinstance(course.screen, ConfirmQuit)
        await pilot.click("#cancel")
        assert not isinstance(course.screen, ConfirmQuit)


@pytest.mark.asyncio
async def test_build_validation_blocks_worker_and_failure_recovers_buttons(course, monkeypatch):
    from unittest.mock import Mock

    from transcript_video.course import builder
    from transcript_video.tui import app as tui

    monkeypatch.setattr(tui, "inspect_video", lambda *a: {"metadata": {}})
    build = Mock(side_effect=RuntimeError("intentional"))
    monkeypatch.setattr(builder, "build_course", build)
    async with course.run_test(size=(120, 50)) as pilot:
        course.push_screen(tui.ReviewScreen())
        await pilot.pause()
        course.draft.output = "bad.wav"
        await pilot.click("#build")
        assert not build.called
        course.draft.output = "course.mp4"
        await pilot.pause(0.25)
        await pilot.click("#build")
        await course.workers.wait_for_complete()
        assert build.call_count == 1
        assert not course.building
        assert not course.screen.query_one("#build", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 120])
async def test_session_layout_fits_terminal_and_preserves_focus(course, width):
    from textual.widgets import DataTable

    async with course.run_test(size=(width, 40)) as pilot:
        course.push_screen(SessionScreen())
        await pilot.pause()
        screen = course.screen
        assert screen.has_class("compact") == (width == 80)
        for selector in ("#videos", "#sessions", "#session-form"):
            region = screen.query_one(selector).region
            assert region.width > 0 and region.x >= 0 and region.right <= width
        browser = screen.query_one("#videos", DataTable)
        browser.focus()
        await pilot.press("tab")
        if screen.focused is screen.query_one("#session-form"):
            await pilot.press("tab")
        assert screen.query_one("#session-title", Input).has_focus
        assert "3" in screen.query_one("#sessions", ListView).border_title

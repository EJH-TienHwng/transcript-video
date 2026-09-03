from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Log, Static

from ..events import PipelineEvent, PipelineStage
from ..process_runner import ffmpeg_progress_handler


@dataclass(slots=True)
class CourseDraft:
    title: str = "Training Course"
    output: str = "data/compilation/training-course.mp4"
    theme_image: str = ""
    sessions: list[dict[str, object]] = field(default_factory=list)
    config_path: Path = Path("configs/courses/training-course.json")
    extra: dict[str, object] = field(default_factory=dict)
    dirty: bool = False


class ConfirmQuit(ModalScreen[bool]):
    def compose(self) -> ComposeResult:
        yield Static("You have unsaved changes. Quit anyway?", id="confirm-message")
        with Horizontal():
            yield Button("Keep editing", id="cancel")
            yield Button("Quit", variant="error", id="quit")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "quit")


class MetadataScreen(Screen):
    BINDINGS: ClassVar = [
        Binding("ctrl+s", "save", "Save"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="form"):
            yield Label("Course metadata", classes="heading")
            yield Label("Title")
            yield Input(self.app.draft.title, id="title")
            yield Label("Output MP4")
            yield Input(self.app.draft.output, id="output")
            yield Label("Theme image (optional)")
            yield Input(self.app.draft.theme_image, id="theme")
            yield Label("Config JSON")
            yield Input(str(self.app.draft.config_path), id="config-path")
            yield Button("Next: sessions", variant="primary", id="next")
        yield Footer()

    def capture(self) -> None:
        draft = self.app.draft
        draft.title = self.query_one("#title", Input).value.strip() or "Training Course"
        draft.output = self.query_one("#output", Input).value.strip()
        draft.theme_image = self.query_one("#theme", Input).value.strip()
        draft.config_path = Path(self.query_one("#config-path", Input).value.strip())
        draft.dirty = True

    @on(Button.Pressed, "#next")
    def next_screen(self) -> None:
        self.capture()
        self.app.push_screen(SessionScreen())

    def action_save(self) -> None:
        self.capture()
        self.app.save_draft()

    def action_quit(self) -> None:
        self.app.request_quit()


class SessionScreen(Screen):
    BINDINGS: ClassVar = [
        Binding("a", "add", "Add"),
        Binding("e", "edit", "Edit"),
        Binding("delete", "remove", "Remove"),
        Binding("u", "up", "Move up"),
        Binding("d", "down", "Move down"),
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with VerticalScroll(id="session-form"):
                yield Label("Session editor", classes="heading")
                yield Input(placeholder="Session title", id="session-title")
                yield Input(placeholder="Video path", id="session-video")
                yield Static("Enter a local video path.", id="metadata")
                yield Button("Add session", variant="primary", id="add")
            yield ListView(id="sessions")
        with Horizontal(id="navigation"):
            yield Button("Back", id="back")
            yield Button("Review / build", variant="success", id="review")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_list()

    def refresh_list(self, selected: int | None = None) -> None:
        view = self.query_one("#sessions", ListView)
        view.clear()
        for index, item in enumerate(self.app.draft.sessions, 1):
            view.append(ListItem(Label(f"{index:02d}. {item['title']}\n{item['video']}")))
        if selected is not None and self.app.draft.sessions:
            view.index = max(0, min(selected, len(self.app.draft.sessions) - 1))

    @on(Input.Changed, "#session-video")
    def inspect_path(self, event: Input.Changed) -> None:
        self.read_metadata(event.value)

    @work(thread=True, exclusive=True, group="metadata")
    def read_metadata(self, value: str) -> None:
        path = Path(value).expanduser()
        text = "Waiting for a valid video path."
        if path.is_file():
            try:
                from ..hardware import get_ffprobe_exe
                from ..process_runner import probe_media

                data = probe_media(get_ffprobe_exe(), path)
                video = next(
                    (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
                )
                duration = float(data.get("format", {}).get("duration", 0))
                text = f"{video.get('width', '?')}x{video.get('height', '?')} • {duration:.1f}s • {video.get('codec_name', '?')}"
            except Exception as exc:
                text = f"Metadata unavailable: {exc}"
        self.app.call_from_thread(self.query_one("#metadata", Static).update, text)

    @on(Button.Pressed, "#add")
    def add_pressed(self) -> None:
        self.action_add()

    def action_add(self) -> None:
        title = self.query_one("#session-title", Input).value.strip()
        video = self.query_one("#session-video", Input).value.strip()
        if not title or not Path(video).expanduser().is_file():
            self.notify("A title and an existing video are required.", severity="error")
            return
        self.app.draft.sessions.append(
            {"number": len(self.app.draft.sessions) + 1, "title": title, "video": video}
        )
        self.app.draft.dirty = True
        self.query_one("#session-title", Input).value = ""
        self.query_one("#session-video", Input).value = ""
        self.refresh_list(len(self.app.draft.sessions) - 1)

    def action_edit(self) -> None:
        view = self.query_one("#sessions", ListView)
        if view.index is None:
            return
        item = self.app.draft.sessions[view.index]
        self.query_one("#session-title", Input).value = str(item["title"])
        self.query_one("#session-video", Input).value = str(item["video"])
        self.app.draft.sessions.pop(view.index)
        self.refresh_list()

    def action_remove(self) -> None:
        view = self.query_one("#sessions", ListView)
        if view.index is not None:
            self.app.draft.sessions.pop(view.index)
            self.app.draft.dirty = True
            self.refresh_list(view.index)

    def _move(self, delta: int) -> None:
        view = self.query_one("#sessions", ListView)
        index = view.index
        if index is None or not 0 <= index + delta < len(self.app.draft.sessions):
            return
        destination = index + delta
        self.app.draft.sessions[index], self.app.draft.sessions[destination] = (
            self.app.draft.sessions[destination],
            self.app.draft.sessions[index],
        )
        self.app.draft.dirty = True
        self.refresh_list(destination)

    def action_up(self) -> None:
        self._move(-1)

    def action_down(self) -> None:
        self._move(1)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_save(self) -> None:
        self.app.save_draft()

    def action_quit(self) -> None:
        self.app.request_quit()

    @on(Button.Pressed, "#back")
    def back_pressed(self) -> None:
        self.action_back()

    @on(Button.Pressed, "#review")
    def review_pressed(self) -> None:
        if not self.app.draft.sessions:
            self.notify("Add at least one session.", severity="error")
            return
        self.app.push_screen(ReviewScreen())


class ReviewScreen(Screen):
    BINDINGS: ClassVar = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="review-summary")
        with Horizontal():
            yield Button("Back", id="back")
            yield Button("Save config", id="save")
            yield Button("Build course", variant="success", id="build")
        yield Log(id="build-log")
        yield Footer()

    def on_mount(self) -> None:
        draft = self.app.draft
        sessions = "\n".join(
            f"{i:02d}. {item['title']} — {item['video']}"
            for i, item in enumerate(draft.sessions, 1)
        )
        self.query_one("#review-summary", Static).update(
            f"[b]{draft.title}[/b]\nOutput: {draft.output}\n\n{sessions}"
        )

    @on(Button.Pressed, "#back")
    def back_pressed(self) -> None:
        self.action_back()

    @on(Button.Pressed, "#save")
    def save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#build")
    def build_pressed(self) -> None:
        self.app.save_draft()
        self.build_in_worker()

    @work(thread=True, exclusive=True, group="build")
    def build_in_worker(self) -> None:
        from ..course.builder import build_course
        from ..course.config import load_course_config

        log = self.query_one("#build-log", Log)
        self.app.call_from_thread(log.write_line, "Building course…")
        try:
            observer = TextualObserver(self.app, log)
            with ffmpeg_progress_handler(
                lambda progress: observer.notify(
                    PipelineEvent(
                        PipelineStage.RENDER,
                        "FFmpeg "
                        f"{progress.elapsed_seconds or 0:.1f}s "
                        f"{progress.speed or ''}".strip(),
                        current=progress.elapsed_seconds,
                    )
                )
            ):
                output = build_course(load_course_config(self.app.draft.config_path), observer)
        except Exception as exc:
            self.app.call_from_thread(log.write_line, f"Build failed: {exc}")
        else:
            self.app.call_from_thread(log.write_line, f"Complete: {output}")

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_save(self) -> None:
        self.app.save_draft()

    def action_quit(self) -> None:
        self.app.request_quit()


class TextualObserver:
    """Forward core pipeline events safely from a worker to Textual."""

    def __init__(self, app: App, log: Log) -> None:
        self.app = app
        self.log = log

    def notify(self, event: PipelineEvent) -> None:
        self.app.call_from_thread(self.log.write_line, f"[{event.stage}] {event.message}")


class CourseApp(App[None]):
    CSS = """
    Screen { background: $surface; }
    #form, #session-form { padding: 1 3; width: 1fr; }
    .heading { text-style: bold; color: $accent; margin-bottom: 1; }
    Input { margin-bottom: 1; }
    #sessions { width: 2fr; border: round $accent; }
    #navigation { height: auto; align-horizontal: right; padding: 1; }
    Button { margin: 0 1; }
    #review-summary { padding: 2; }
    #build-log { height: 1fr; border: round $primary; }
    ConfirmQuit { align: center middle; }
    ConfirmQuit > Static { width: 55; height: 7; padding: 2; background: $panel; border: round $warning; }
    """
    BINDINGS: ClassVar = [Binding("ctrl+c", "request_quit", "Quit", show=False)]

    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__()
        selected = config_path or Path("configs/courses/training-course.json")
        self.draft = self._read_draft(selected)

    @staticmethod
    def _read_draft(config_path: Path) -> CourseDraft:
        if not config_path.is_file():
            return CourseDraft(config_path=config_path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return CourseDraft(config_path=config_path)
        sessions = raw.get("sessions", [])
        return CourseDraft(
            title=str(raw.get("title", "Training Course")),
            output=str(raw.get("output", "data/compilation/training-course.mp4")),
            theme_image=str(raw.get("theme_image") or ""),
            sessions=list(sessions) if isinstance(sessions, list) else [],
            config_path=config_path,
            extra={
                key: value
                for key, value in raw.items()
                if key not in {"title", "output", "theme_image", "sessions"}
            },
        )

    def on_mount(self) -> None:
        self.push_screen(MetadataScreen())

    def save_draft(self) -> None:
        draft = self.draft
        payload = {
            **draft.extra,
            "title": draft.title,
            "output": draft.output,
            "theme_image": draft.theme_image or None,
            "sessions": [{**item, "number": index} for index, item in enumerate(draft.sessions, 1)],
        }
        draft.config_path.parent.mkdir(parents=True, exist_ok=True)
        draft.config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        draft.dirty = False
        self.notify(f"Saved {draft.config_path}")

    def request_quit(self) -> None:
        if not self.draft.dirty:
            self.exit()
            return
        self.push_screen(ConfirmQuit(), self._quit_answered)

    def _quit_answered(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit()

    def action_request_quit(self) -> None:
        self.request_quit()

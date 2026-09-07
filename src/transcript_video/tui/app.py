from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Log,
    ProgressBar,
    Static,
)
from textual.worker import get_current_worker

from ..application.inspection import inspect_video, media_duration, media_fields
from ..application.settings import resolve_settings
from ..artifacts import is_partial_artifact
from ..config import VIDEO_EXTENSIONS, find_project_root
from ..context import current_context, log_context
from ..course.config import (
    RenderConfig,
    TocConfig,
    parse_course_config,
    read_course_document,
    save_course_config,
    validate_session_video,
)
from ..events import EventKind, PipelineEvent, PipelineStage, event_scope
from ..ui.progress import LineThrottle, format_duration, format_pipeline_event
from ..ui.theme import PALETTE


@dataclass(slots=True)
class CourseDraft:
    title: str = "Training Course"
    output: str = "data/compilation/training-course.mp4"
    theme_image: str = ""
    sessions: list[dict[str, object]] = field(default_factory=list)
    config_path: Path = Path("configs/courses/training-course.json")
    extra: dict[str, object] = field(default_factory=dict)
    dirty: bool = False

    def document(self) -> dict:
        return {
            **self.extra,
            "title": self.title,
            "output": self.output,
            "theme_image": self.theme_image or None,
            "sessions": self.sessions,
        }


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
            extra = self.app.draft.extra
            yield Label("Session card duration (seconds)")
            yield Input(str(extra.get("card_duration", 5.0)), id="card-duration")
            yield Checkbox("Add chapters", value=extra.get("add_chapters", True), id="add-chapters")
            toc = {**asdict(TocConfig()), **extra.get("toc", {})}
            yield Checkbox("Table of contents", value=toc["enabled"], id="toc-enabled")
            for name in ("heading", "items_per_page", "page_duration"):
                yield Label("TOC " + name.replace("_", " "))
                yield Input(str(toc[name]), id="toc-" + name.replace("_", "-"))
            with Collapsible(title="Advanced rendering", collapsed=True):
                render = {**asdict(RenderConfig()), **extra.get("render", {})}
                for name in asdict(RenderConfig()):
                    yield Label(name.replace("_", " ").title())
                    yield Input(str(render[name] or ""), id="render-" + name.replace("_", "-"))
            yield Button("Next: sessions", variant="primary", id="next")
        yield Footer()

    @on(Input.Changed)
    @on(Checkbox.Changed)
    def mark_dirty(self, event) -> None:
        if event.control.has_focus:
            self.app.draft.dirty = True

    def capture(self) -> bool:
        draft = self.app.draft
        payload = deepcopy(draft.document())
        payload.update(
            title=self.query_one("#title", Input).value.strip(),
            output=self.query_one("#output", Input).value.strip(),
            theme_image=self.query_one("#theme", Input).value.strip() or None,
        )
        config_path = (
            Path(self.query_one("#config-path", Input).value.strip()).expanduser().resolve()
        )
        try:
            payload["card_duration"] = float(self.query_one("#card-duration", Input).value)
            payload["add_chapters"] = self.query_one("#add-chapters", Checkbox).value
            toc = payload.setdefault("toc", {})
            toc["enabled"] = self.query_one("#toc-enabled", Checkbox).value
            for name, convert in (
                ("heading", str),
                ("items_per_page", int),
                ("page_duration", float),
            ):
                toc[name] = convert(self.query_one("#toc-" + name.replace("_", "-"), Input).value)
            render = payload.setdefault("render", {})
            for name, default in asdict(RenderConfig()).items():
                value = self.query_one("#render-" + name.replace("_", "-"), Input).value.strip()
                render[name] = (
                    int(value)
                    if type(default) is int
                    else value or None
                    if name == "font_path"
                    else value
                )
            parse_course_config(
                payload, find_project_root(config_path.parent), allow_empty_sessions=True
            )
        except (ValueError, OSError) as exc:
            self.notify(f"Invalid settings: {exc}", severity="error")
            return False
        draft.title, draft.output, draft.theme_image = (
            payload["title"],
            payload["output"],
            payload["theme_image"] or "",
        )
        draft.config_path = config_path
        self.app.project_root = find_project_root(config_path.parent)
        self.app.inspection_settings.project.root = str(self.app.project_root)
        draft.extra = {
            key: value
            for key, value in payload.items()
            if key not in {"title", "output", "theme_image", "sessions"}
        }
        draft.dirty = True
        return True

    @on(Button.Pressed, "#next")
    def next_screen(self) -> None:
        if self.capture():
            self.app.push_screen(SessionScreen())

    def action_save(self) -> None:
        if self.capture():
            self.app.save_draft()

    def action_quit(self) -> None:
        self.app.request_quit()


class SessionScreen(Screen):
    editing_index: int | None = None
    BINDINGS: ClassVar = [
        Binding("a", "add", "Add"),
        Binding("e", "edit", "Edit"),
        Binding("delete", "remove", "Remove"),
        Binding("u", "up", "Move up"),
        Binding("d", "down", "Move down"),
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "back", "Back"),
        Binding("ctrl+e", "cancel_edit", "Cancel edit"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="session-workspace"):
            yield DataTable(id="videos", cursor_type="row")
            with VerticalScroll(id="session-form"):
                yield Label("Session editor", classes="heading")
                yield Label("Title")
                yield Input(placeholder="Session title", id="session-title")
                yield Label("Video path")
                yield Input(placeholder="Video path", id="session-video")
                yield Static(
                    "Select a video; Enter fills the form, Add confirms.",
                    id="metadata",
                    markup=False,
                )
                yield Button("Add session", variant="primary", id="add")
                yield Button("Cancel edit", id="cancel-edit", disabled=True)
            yield ListView(id="sessions")
        with Horizontal(id="navigation"):
            yield Button("Back", id="back")
            yield Button("Review / build", variant="success", id="review")
        yield Footer()

    def on_mount(self) -> None:
        self.set_class(self.size.width < 100, "compact")
        self.query_one("#videos", DataTable).border_title = "Available videos"
        self.refresh_list()
        self.query_one("#videos", DataTable).add_column("Available videos (Enter to select)")
        self.scan_videos()

    def on_resize(self, event) -> None:
        self.set_class(event.size.width < 100, "compact")

    @work(thread=True, exclusive=True, group="browser")
    def scan_videos(self) -> None:
        root = self.app.project_root
        try:
            videos = sorted(
                path.resolve()
                for folder in (root / "data/input", root / "data/output")
                if folder.is_dir()
                for path in folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in VIDEO_EXTENSIONS
                and not is_partial_artifact(path)
            )
        except OSError as exc:
            self.app.call_from_thread(
                self.notify, f"Video browser unavailable: {exc}", severity="error"
            )
            return
        self.app.call_from_thread(self.show_videos, videos)

    def show_videos(self, videos: list[Path]) -> None:
        if not self.is_mounted:
            return
        table = self.query_one("#videos", DataTable)
        table.clear()
        for path in videos:
            table.add_row(str(path.relative_to(self.app.project_root)), key=str(path))

    @on(DataTable.RowHighlighted, "#videos")
    def highlight_video(self, event: DataTable.RowHighlighted) -> None:
        self.read_metadata(str(event.row_key.value))

    @on(DataTable.RowSelected, "#videos")
    def select_video(self, event: DataTable.RowSelected) -> None:
        if self.editing_index is not None:
            self.notify(
                "Save or cancel the edit before selecting another video.", severity="warning"
            )
            return
        from ..course.wizard import _default_title_from_video

        path = Path(str(event.row_key.value))
        self.query_one("#session-video", Input).value = str(path)
        self.query_one("#session-title", Input).value = _default_title_from_video(path)

    def refresh_list(self, selected: int | None = None) -> None:
        view = self.query_one("#sessions", ListView)
        view.border_title = f"Course sessions · {len(self.app.draft.sessions)}"
        view.clear()
        for index, item in enumerate(self.app.draft.sessions, 1):
            view.append(
                ListItem(
                    Label(
                        f"{item.get('number') or index:02d}. {item['title']}\n{item['video']}",
                        markup=False,
                    )
                )
            )
        if selected is not None and self.app.draft.sessions:
            view.index = max(0, min(selected, len(self.app.draft.sessions) - 1))

    @on(Input.Changed, "#session-video")
    def inspect_path(self, event: Input.Changed) -> None:
        self.read_metadata(event.value)

    @on(Input.Changed)
    def mark_form_dirty(self, event: Input.Changed) -> None:
        if event.input.has_focus:
            self.app.draft.dirty = True

    @work(thread=True, exclusive=True, group="metadata")
    def read_metadata(self, value: str) -> None:
        worker = get_current_worker()
        path = (self.app.project_root / Path(value).expanduser()).resolve()
        text = "Waiting for a valid video path."
        if path.is_file():
            try:
                with self.app.metadata_lock:
                    if worker.is_cancelled:
                        return
                    result = inspect_video(
                        path, self.app.inspection_settings, self.app.metadata_cache
                    )
                text = "\n".join(
                    f"{key}: {value}" for key, value in media_fields(result["metadata"]).items()
                )
                text += "\n\nArtifacts\n" + "\n".join(
                    f"{key}: {value}" for key, value in result["artifact_states"].items()
                )
            except (OSError, ValueError, RuntimeError) as exc:
                text = f"Metadata unavailable: {exc}"
        if not worker.is_cancelled:
            self.app.call_from_thread(self.show_metadata, text)

    def show_metadata(self, text: str) -> None:
        if self.is_mounted:
            self.query_one("#metadata", Static).update(text)

    @on(Button.Pressed, "#add")
    def add_pressed(self) -> None:
        self.action_add()

    def action_add(self) -> None:
        title = self.query_one("#session-title", Input).value.strip()
        video = self.query_one("#session-video", Input).value.strip()
        draft = self.app.draft
        root = find_project_root(draft.config_path.parent)
        try:
            if not title:
                raise ValueError("A session title is required.")
            validate_session_video(
                video,
                root,
                [
                    (root / str(item["video"])).resolve()
                    for index, item in enumerate(draft.sessions)
                    if index != self.editing_index
                ],
            )
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        selected = self.editing_index
        if selected is None:
            number = (
                max(
                    (item.get("number") or i for i, item in enumerate(draft.sessions, 1)), default=0
                )
                + 1
            )
            draft.sessions.append({"number": number, "title": title, "video": video})
            selected = len(draft.sessions) - 1
        else:
            draft.sessions[selected] = {**draft.sessions[selected], "title": title, "video": video}
        draft.dirty = True
        self.action_cancel_edit()
        self.refresh_list(selected)

    @on(Button.Pressed, "#cancel-edit")
    def action_cancel_edit(self) -> None:
        self.editing_index = None
        self.query_one("#add", Button).label = "Add session"
        self.query_one("#cancel-edit", Button).disabled = True
        self.query_one("#session-title", Input).value = ""
        self.query_one("#session-video", Input).value = ""

    def action_edit(self) -> None:
        view = self.query_one("#sessions", ListView)
        if view.index is None:
            return
        item = self.app.draft.sessions[view.index]
        self.query_one("#session-title", Input).value = str(item["title"])
        self.query_one("#session-video", Input).value = str(item["video"])
        self.editing_index = view.index
        self.query_one("#add", Button).label = "Save changes"
        self.query_one("#cancel-edit", Button).disabled = False

    def action_remove(self) -> None:
        if self.editing_index is not None:
            self.notify("Save or cancel the edit before removing sessions.", severity="warning")
            return
        view = self.query_one("#sessions", ListView)
        if view.index is not None:
            self.app.draft.sessions.pop(view.index)
            self.app.draft.dirty = True
            self.refresh_list(view.index)

    def _move(self, delta: int) -> None:
        if self.editing_index is not None:
            self.notify("Save or cancel the edit before moving sessions.", severity="warning")
            return
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
        self.action_cancel_edit()
        self.app.pop_screen()

    def action_save(self) -> None:
        if self.editing_index is not None:
            self.notify(
                "Use Save changes or Cancel edit before saving the config.", severity="warning"
            )
            return
        self.app.save_draft()

    def action_quit(self) -> None:
        self.app.request_quit()

    @on(Button.Pressed, "#back")
    def back_pressed(self) -> None:
        self.action_back()

    @on(Button.Pressed, "#review")
    def review_pressed(self) -> None:
        if self.editing_index is not None:
            self.notify("Save or cancel the edit before reviewing.", severity="warning")
            return
        if not self.app.draft.sessions:
            self.notify("Add at least one session.", severity="error")
            return
        self.app.push_screen(ReviewScreen())


class ReviewScreen(Screen):
    BINDINGS: ClassVar = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+b", "build", "Build"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="review-details"):
            yield Label("Review & build", classes="heading")
            yield Static(id="review-summary", markup=False)
        with Horizontal(id="review-actions"):
            yield Button("Back", id="back")
            yield Button("Save config", id="save")
            yield Button("Build course", variant="success", id="build")
        yield Label("Overall (completed stages)")
        yield ProgressBar(total=None, id="overall-progress", show_eta=False)
        yield Static("Ready", id="stage-label", markup=False)
        yield ProgressBar(total=None, id="stage-progress")
        yield Static("", id="session-count")
        yield Log(id="build-log")
        yield Footer()

    def on_mount(self) -> None:
        self.completed_stages = set()
        self.planned_stages = []
        self.review_durations()

    @work(thread=True, exclusive=True, group="review-metadata")
    def review_durations(self) -> None:
        draft = self.app.draft
        text = f"{draft.title}\nOutput: {draft.output}\nTheme: {draft.theme_image or 'None'}\n\n"
        durations = []
        for index, item in enumerate(draft.sessions, 1):
            try:
                with self.app.metadata_lock:
                    result = inspect_video(
                        self.app.project_root / str(item["video"]),
                        self.app.inspection_settings,
                        self.app.metadata_cache,
                    )
                duration = media_duration(result["metadata"])
            except (ValueError, OSError, RuntimeError):
                duration = None
            durations.append(duration)
            text += f"{item.get('number') or index:02d}. {item['title']}\n  {item['video']}\n  {format_duration(duration) if duration is not None else 'Duration unavailable'}\n"
        source = (
            sum(durations) if durations and all(value is not None for value in durations) else None
        )
        estimated = None
        if source is not None:
            from ..course.timeline import build_timeline

            try:
                config = parse_course_config(draft.document(), self.app.project_root)
                estimated = build_timeline(config, durations)[-1].content_end
            except (ValueError, OSError):
                pass  # Invalid draft is shown for editing; saving still reports domain validation errors.
        text += f"\nTotal source duration: {format_duration(source) if source is not None else 'Unavailable'}"
        text += f"\nEstimated output duration: {format_duration(estimated) if estimated is not None else 'Unavailable'}"
        if not get_current_worker().is_cancelled:
            self.app.call_from_thread(self.show_review, text)

    def show_review(self, text: str) -> None:
        if self.is_mounted:
            self.query_one("#review-summary", Static).update(text)

    def apply_progress(self, event: PipelineEvent) -> None:
        overall = self.query_one("#overall-progress", ProgressBar)
        progress = self.query_one("#stage-progress", ProgressBar)
        label = self.query_one("#stage-label", Static)
        if event.stage == PipelineStage.RUN:
            if event.kind == EventKind.START:
                self.planned_stages = event.details.get("stages", [])
                self.completed_stages.clear()
                overall.update(total=len(self.planned_stages) or None, progress=0)
            elif event.kind == EventKind.COMPLETE:
                overall.update(progress=len(self.planned_stages))
                label.update("Course build complete")
            elif event.kind == EventKind.FAILURE:
                label.update(f"Build failed: {event.message}")
            return
        if event.kind in {EventKind.ARTIFACT, EventKind.WARNING, EventKind.REVIEW}:
            return
        label.update(
            f"{event.stage.value}: {event.message}"
            + (" (reused)" if event.kind == EventKind.REUSED else "")
        )
        if event.kind == EventKind.START:
            progress.update(total=event.total, progress=0)
        elif event.kind == EventKind.PROGRESS:
            progress.update(
                total=event.total if event.total and event.total > 0 else None,
                progress=event.current or 0,
            )
            if event.stage == PipelineStage.NORMALIZE and event.context.operation is None:
                self.query_one("#session-count", Static).update(
                    f"{event.current or 0:g} / {event.total:g} sessions"
                    if event.total
                    else "Normalizing sessions"
                )
        elif (
            event.kind in {EventKind.COMPLETE, EventKind.REUSED} and event.context.operation is None
        ):
            progress.update(total=1, progress=1)
            if event.stage.value in self.planned_stages:
                self.completed_stages.add(event.stage.value)
                overall.update(progress=len(self.completed_stages))
        elif event.kind == EventKind.FAILURE:
            progress.update(total=1, progress=0)

    @on(Button.Pressed, "#back")
    def back_pressed(self) -> None:
        self.action_back()

    @on(Button.Pressed, "#save")
    def save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#build")
    def build_pressed(self) -> None:
        self.action_build()

    def action_build(self) -> None:
        if not self.app.building and self.app.save_draft():
            self.app.building = True
            for button in self.query("#review-actions Button"):
                button.disabled = True
            self.build_in_worker()

    @work(thread=True, exclusive=True, group="build")
    def build_in_worker(self) -> None:
        from ..course.builder import build_course
        from ..course.config import load_course_config

        log = self.query_one("#build-log", Log)
        try:
            observer = TextualObserver(self.app, log, self.apply_progress)
            with event_scope(observer, **self.app.run_context.fields()):
                build_course(load_course_config(self.app.draft.config_path), observer)
        except Exception as exc:
            with log_context(**self.app.run_context.fields()):
                logging.getLogger(__name__).debug(
                    "Course build failed", exc_info=True, extra={"diagnostic_only": True}
                )
            self.app.call_from_thread(log.write_line, f"Build failed: {exc}")
            self.app.call_from_thread(
                self.apply_progress,
                PipelineEvent(PipelineStage.RUN, str(exc), kind=EventKind.FAILURE),
            )
        finally:
            self.app.call_from_thread(self.build_finished)

    def build_finished(self) -> None:
        self.app.building = False
        for button in self.query("#review-actions Button"):
            button.disabled = False

    def action_back(self) -> None:
        if not self.app.building:
            self.app.pop_screen()

    def action_save(self) -> None:
        if not self.app.building:
            self.app.save_draft()

    def action_quit(self) -> None:
        self.app.request_quit()


class TextualObserver:
    """Forward core pipeline events safely from a worker to Textual."""

    def __init__(self, app: App, log: Log, progress_callback=None) -> None:
        self.app = app
        self.log = log
        self.throttle = LineThrottle()
        self.progress_callback = progress_callback

    def notify(self, event: PipelineEvent) -> None:
        if self.progress_callback is not None:
            self.app.call_from_thread(self.progress_callback, event)
        if event.kind == EventKind.FAILURE or (
            event.kind == EventKind.REVIEW and event.context.operation != "quality"
        ):
            return  # The worker owns its one failure line; review details live in the report.
        if self.throttle.accepts(event):
            text = format_pipeline_event(event)
            if event.context.operation == "quality":
                text += " · " + " · ".join(f"{key}={value}" for key, value in event.details.items())
            if event.artifact:
                text += f" · {event.artifact}"
            self.app.call_from_thread(self.log.write_line, f"{event.timestamp[11:19]} {text}")


class CourseApp(App[None]):
    CSS = """
    Screen { background: $surface; }
    #form, #session-form { padding: 1 3; width: 1fr; }
    #form { max-width: 100; }
    .heading { text-style: bold; color: $accent; margin-bottom: 1; }
    Input { margin-bottom: 1; }
    #sessions, #videos { width: 1fr; border: round $accent; }
    #session-workspace { height: 1fr; }
    SessionScreen.compact #session-workspace { layout: grid; grid-size: 2; grid-columns: 2fr 3fr; grid-rows: 2fr 1fr; }
    SessionScreen.compact #sessions { column-span: 2; width: 1fr; }
    #sessions ListItem { height: auto; padding: 0 1; }
    #sessions Label { width: 1fr; }
    Input:focus, #videos:focus, #sessions:focus { border: heavy $accent; }
    #session-form { padding: 1; width: 1fr; }
    #navigation { height: auto; align-horizontal: right; padding: 1; }
    Button { margin: 0 1; }
    #review-summary { padding: 0 1 1 1; }
    #review-details { padding: 1; }
    #review-details { height: 2fr; }
    #review-actions { height: auto; }
    #build-log { height: 1fr; border: round $primary; }
    ConfirmQuit { align: center middle; }
    ConfirmQuit > Static { width: 55; height: 7; padding: 2; background: $panel; border: round $warning; }
    """
    BINDINGS: ClassVar = [Binding("ctrl+c", "request_quit", "Quit", show=False)]

    def __init__(self, config_path: Path | None = None) -> None:
        super().__init__()
        from textual.theme import Theme

        self.register_theme(
            Theme(
                name="transcript-video",
                primary=PALETTE["accent"],
                secondary=PALETTE["stage"],
                accent=PALETTE["accent"],
                warning=PALETTE["warning"],
                error=PALETTE["error"],
                success=PALETTE["success"],
            )
        )
        self.theme = "transcript-video"
        self.run_context = current_context.get()
        selected = config_path or Path("configs/courses/training-course.json")
        self.draft = self._read_draft(selected)
        self.project_root = find_project_root(selected.parent)
        self.inspection_settings = resolve_settings(
            config_path=self.project_root / "configs/transcription.toml",
            overrides={"project.root": str(self.project_root)},
        ).settings
        self.metadata_cache = {}
        self.metadata_lock = Lock()
        self.building = False

    @staticmethod
    def _read_draft(config_path: Path) -> CourseDraft:
        if not config_path.is_file():
            return CourseDraft(config_path=config_path)
        raw = read_course_document(config_path)
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

    def save_draft(self) -> bool:
        draft = self.draft
        payload = draft.document()
        try:
            save_course_config(payload, draft.config_path)
        except (ValueError, OSError) as exc:
            self.notify(f"Cannot save {draft.config_path}: {exc}", severity="error")
            return False
        draft.dirty = False
        self.notify(f"Saved {draft.config_path}")
        return True

    def request_quit(self) -> None:
        if self.building:
            self.notify("Wait for the course build to finish before quitting.", severity="warning")
            return
        if not self.draft.dirty:
            self.exit()
            return
        self.push_screen(ConfirmQuit(), self._quit_answered)

    def _quit_answered(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit()

    def action_request_quit(self) -> None:
        self.request_quit()

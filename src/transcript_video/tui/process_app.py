from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, DataTable, Footer, Input, Label, Log, Static

from ..application.processing import execute_batch_process_plan
from ..config import find_project_root
from ..interactive import InteractiveBatchController
from .app import TextualObserver


class ProcessApp(App[None]):
    """A keyboard-first workspace over the shared heterogeneous batch application model."""

    CSS = """
    Screen { background: $background; }
    #brand { height: 3; padding: 1 2; text-style: bold; color: $accent; background: $panel; }
    #workspace { height: 1fr; }
    #videos, #preview, #settings { width: 1fr; border: round $border; margin: 1; padding: 1; }
    #preview { overflow-y: auto; }
    #settings Input, #settings Checkbox { margin-bottom: 1; }
    #actions { height: auto; padding: 1; align-horizontal: right; }
    #log { height: 10; border: round $border; margin: 0 1 1 1; }
    ProcessApp.compact #workspace { layout: vertical; overflow-y: auto; }
    Button:focus, Input:focus, Checkbox:focus, DataTable:focus { border: heavy $accent; }
    """
    BINDINGS: ClassVar = [
        Binding("space", "toggle", "Select"),
        Binding("a", "apply_all", "Apply all"),
        Binding("r", "reset", "Reset"),
        Binding("d", "dry_run", "Dry run"),
        Binding("p", "process", "Process"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, root: Path | None = None) -> None:
        super().__init__()
        self.controller = InteractiveBatchController(root or find_project_root())
        self.selected: set[Path] = set()
        self.running = False

    def compose(self) -> ComposeResult:
        yield Static("TRANSCRIPT·VIDEO  /  Process videos", id="brand")
        with Horizontal(id="workspace"):
            yield DataTable(id="videos", cursor_type="row")
            with Vertical(id="settings"):
                yield Label("Batch defaults")
                yield Input(value="", placeholder="Profile (optional)", id="profile")
                yield Checkbox("Enable TTS", id="tts")
                yield Checkbox("Verify final audio", id="verify")
                yield Checkbox("Enable speed-up", id="speedup")
                yield Checkbox("Force transcription", id="force-transcription")
                yield Input(value="auto", placeholder="Encoder", id="encoder")
            yield Static(id="preview", markup=False)
        with Horizontal(id="actions"):
            yield Button("Apply selected", id="apply")
            yield Button("Apply all", id="apply-all")
            yield Button("Reset", id="reset")
            yield Button("Dry run", id="dry-run")
            yield Button("Process", variant="success", id="process")
        yield Log(id="log")
        yield Footer()

    def on_mount(self) -> None:
        self.set_class(self.size.width < 100, "compact")
        table = self.query_one("#videos", DataTable)
        table.add_columns("Selected", "Video")
        self.refresh_videos()
        self.refresh_preview()

    def on_resize(self, event) -> None:
        self.set_class(event.size.width < 100, "compact")

    def refresh_videos(self) -> None:
        table = self.query_one("#videos", DataTable)
        table.clear()
        for video in self.controller.discover_videos():
            table.add_row("x" if video in self.selected else "", video.name, key=str(video))

    def _values(self) -> dict[str, object]:
        return {
            "tts.enabled": self.query_one("#tts", Checkbox).value,
            "tts.verify_final_audio": self.query_one("#verify", Checkbox).value,
            "speedup.enabled": self.query_one("#speedup", Checkbox).value,
            "transcription.overwrite_srt": self.query_one(
                "#force-transcription", Checkbox
            ).value,
            "hardware.video_encoder": self.query_one("#encoder", Input).value.strip() or "auto",
        }

    def _sync_defaults(self) -> None:
        profile = self.query_one("#profile", Input).value.strip() or None
        self.controller.update_defaults(self._values(), profile)

    def refresh_preview(self) -> None:
        self._sync_defaults()
        lines = ["Effective plan"]
        for video, profile, tts, speedup, force in self.controller.preview_rows():
            lines.append(
                f"{video}\n  profile={profile}  tts={tts}  speed-up={speedup}  force={force}"
            )
        if len(lines) == 1:
            lines.append("Select videos with Space or Enter.")
        self.query_one("#preview", Static).update("\n".join(lines))

    def action_toggle(self) -> None:
        table = self.query_one("#videos", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            return
        key = table.coordinate_to_cell_key((table.cursor_row, 0)).row_key
        self.toggle(Path(str(key.value)))

    @on(DataTable.RowSelected, "#videos")
    def selected_row(self, event: DataTable.RowSelected) -> None:
        self.toggle(Path(str(event.row_key.value)))

    def toggle(self, video: Path) -> None:
        if video in self.selected:
            self.selected.remove(video)
        else:
            self.selected.add(video)
        self.controller.select(
            [video for video in self.controller.discover_videos() if video in self.selected]
        )
        self.refresh_videos()
        self.refresh_preview()

    def _apply(self, videos: list[Path]) -> None:
        if not videos:
            self.notify("Select at least one video.", severity="warning")
            return
        self._sync_defaults()
        self.controller.apply(videos, self._values())
        self.refresh_preview()

    def action_apply_all(self) -> None:
        self._apply(self.controller.batch.videos)

    @on(Button.Pressed, "#apply")
    def apply_selected(self) -> None:
        self._apply(list(self.selected))

    @on(Button.Pressed, "#apply-all")
    def apply_all_pressed(self) -> None:
        self.action_apply_all()

    def action_reset(self) -> None:
        self.controller.reset(list(self.selected))
        self.refresh_preview()

    @on(Button.Pressed, "#reset")
    def reset_pressed(self) -> None:
        self.action_reset()

    def _plan(self, dry_run: bool) -> None:
        try:
            self._sync_defaults()
            plan = self.controller.build_plan(defer_video_errors=not dry_run)
        except (OSError, ValueError) as exc:
            self.notify(str(exc), severity="error")
            return
        if dry_run:
            self.query_one("#log", Log).write_line(
                "Dry run\n" + "\n".join(str(video) for video in plan.videos)
            )
            return
        self.running = True
        self.run_plan(plan)

    def action_dry_run(self) -> None:
        self._plan(True)

    @on(Button.Pressed, "#dry-run")
    def dry_run_pressed(self) -> None:
        self.action_dry_run()

    def action_process(self) -> None:
        if not self.running:
            self._plan(False)

    @on(Button.Pressed, "#process")
    def process_pressed(self) -> None:
        self.action_process()

    @work(thread=True, exclusive=True, group="process")
    def run_plan(self, plan) -> None:
        log = self.query_one("#log", Log)
        try:
            summary = execute_batch_process_plan(plan, TextualObserver(self, log))
            self.call_from_thread(
                log.write_line,
                f"Completed: {summary.succeeded}/{summary.total}; waiting={len(summary.waiting_for_translation)}",
            )
        except Exception as exc:
            self.call_from_thread(log.write_line, f"Processing failed: {exc}")
        finally:
            self.call_from_thread(self._finished)

    def _finished(self) -> None:
        self.running = False

    def action_quit(self) -> None:
        if self.running:
            self.notify("Wait for processing to finish before quitting.", severity="warning")
            return
        self.exit()

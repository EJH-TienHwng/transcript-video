from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from ..events import EventKind, PipelineEvent, PipelineStage
from .theme import SYMBOLS, terminal_symbols


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def format_pipeline_event(event: PipelineEvent, symbols: dict | None = None) -> str:
    symbols = symbols or SYMBOLS
    text = f"{symbols[event.kind]} {event.stage.value} · {event.message}"
    if event.current is not None and event.total and event.total > 0:
        if event.details.get("unit") == "seconds":
            text += f" · {min(100, event.current / event.total * 100):.0f}%"
        else:
            text += f" · {event.current:g}/{event.total:g}"
    if speed := event.details.get("speed"):
        text += f" · {speed}"
        try:
            rate = float(speed.rstrip("x"))
            if event.total and event.current is not None and rate > 0:
                text += f" · ETA {format_duration(max(0, event.total - event.current) / rate)}"
        except (ValueError, AttributeError):
            pass
    if "sentences" in event.details and "total_sentences" in event.details:
        text += f" · {event.details['sentences']}/{event.details['total_sentences']} sentences"
    if "elapsed_seconds" in event.details:
        text += f" · {format_duration(event.details['elapsed_seconds'])}"
    return text


@dataclass
class StageState:
    status: str = "pending"
    message: str = ""
    current: float | None = None
    total: float | None = None
    elapsed: float | None = None
    speed: str | None = None


@dataclass
class PipelineProgressState:
    total: int = 0
    completed: int = 0
    video: str = ""
    stages: dict[str, StageState] = field(default_factory=dict)
    videos: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, list[str]] = field(default_factory=dict)
    quality: dict[str, dict] = field(default_factory=dict)
    timings: dict[str, dict[str, float]] = field(default_factory=dict)
    course_details: dict = field(default_factory=dict)

    def apply(self, event: PipelineEvent) -> None:
        video = event.context.video or "course"
        if event.stage == PipelineStage.RUN:
            if event.kind == EventKind.START:
                self.total = int(event.total or 1)
                self.stages = {name: StageState() for name in event.details.get("stages", [])}
            if event.kind == EventKind.COMPLETE:
                self.completed = self.total
            return
        if event.stage == PipelineStage.VIDEO:
            if event.kind == EventKind.START:
                self.video = video
                self.stages = {name: StageState() for name in event.details.get("stages", [])}
            elif event.kind in {EventKind.COMPLETE, EventKind.FAILURE}:
                self.videos[video] = event.kind.value
                self.completed = len(self.videos)
            return
        if event.kind == EventKind.ARTIFACT and event.artifact:
            if event.stage == PipelineStage.COMPLETE:
                self.course_details = event.details
            category = event.details.get("category", "Output")
            paths = self.artifacts.setdefault(category, [])
            if str(event.artifact) not in paths:
                paths.append(str(event.artifact))
            return
        if event.context.operation == "quality":
            self.quality[video] = event.details
            return
        stage = self.stages.setdefault(event.stage.value, StageState())
        if event.context.operation not in {None, "ffmpeg", "chunks", "sentences"}:
            return
        stage.message = event.message
        if event.current is not None:
            stage.current, stage.total = event.current, event.total
        stage.speed = event.details.get("speed")
        if event.kind == EventKind.START:
            stage.status = "active"
            stage.current, stage.total = event.current, event.total
        elif event.kind in {EventKind.COMPLETE, EventKind.REUSED}:
            if event.context.operation == "chunks" and event.kind == EventKind.REUSED:
                return
            if stage.status != "reused":
                stage.status = event.kind.value
            stage.elapsed = event.details.get("elapsed_seconds")
            if stage.elapsed is not None:
                self.timings.setdefault(video, {})[event.stage.value] = stage.elapsed
        elif event.kind == EventKind.FAILURE:
            stage.status = "failed"
        elif event.kind == EventKind.REVIEW:
            stage.status = "review"


class LineThrottle:
    def __init__(self):
        self.last: dict[tuple, tuple[float, float]] = {}

    def accepts(self, event: PipelineEvent) -> bool:
        if event.kind != EventKind.PROGRESS:
            return True
        key = (event.context.video, event.stage, event.context.operation, event.context.chunk)
        now = time.monotonic()
        fraction = event.current / event.total if event.current is not None and event.total else 0
        previous, previous_fraction = self.last.get(key, (-float("inf"), -1))
        if now - previous < 1 and abs(fraction - previous_fraction) < 0.1:
            return False
        self.last[key] = (now, fraction)
        return True


class WorkSpinner(SpinnerColumn):
    def render(self, task):
        return super().render(task) if task.total is None else Text("")


class WorkCount(ProgressColumn):
    def render(self, task):
        return Text(f"{task.completed:g}/{task.total:g}" if task.total is not None else "")


class RichProgressObserver:
    """State reducer plus CLI adapter; non-TTY and --plain never start Live."""

    def __init__(
        self, console: Console, *, verbosity: int = 0, plain: bool = False, root: Path | None = None
    ):
        self.console, self.verbosity = console, verbosity
        self.symbols = terminal_symbols(console.encoding)
        self.root = (root or Path.cwd()).resolve()
        self.state = PipelineProgressState()
        self.dynamic = console.is_terminal and not plain and verbosity >= 0
        self.progress = Progress(
            WorkSpinner(),
            TextColumn("{task.description}", markup=False),
            BarColumn(),
            WorkCount(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            auto_refresh=False,
        )
        self.overall = self.progress.add_task("Overall", total=1)
        self.stage_task = self.progress.add_task("Preparing", total=None, visible=False)
        self.live = (
            Live(console=console, refresh_per_second=8, transient=True) if self.dynamic else None
        )
        self.throttle = LineThrottle()
        self.last_refresh = 0.0
        self.progress_detail = ""

    def __enter__(self):
        if self.live:
            self.live.start()
        return self

    def __exit__(self, *exc):
        if self.live:
            self.live.stop()

    def notify(self, event: PipelineEvent) -> None:
        self.state.apply(event)
        if self.verbosity < 0:
            if event.kind == EventKind.WARNING:
                self.console.print(
                    Text(format_pipeline_event(event, self.symbols), style="warning")
                )
            return
        if event.kind == EventKind.WARNING:
            self.console.print(Text(format_pipeline_event(event, self.symbols), style="warning"))
        detail = event.context.operation
        # File logs own detailed diagnostics. Reviews are summarized once per TTS stage.
        visible = event.kind not in {
            EventKind.ARTIFACT,
            EventKind.REVIEW,
            EventKind.FAILURE,
            EventKind.WARNING,
        }
        if detail == "quality":
            counts = event.details
            self.console.print(
                Text(
                    "TTS quality · "
                    + " · ".join(
                        f"{k}={counts[k]}"
                        for k in (
                            "sentences",
                            "aligned",
                            "regenerated",
                            "speed_adjusted",
                            "shifted",
                            "overflow",
                            "failures",
                            "flagged",
                        )
                        if k in counts
                    ),
                    style="review",
                )
            )
            visible = False
        # Detailed recovery messages are already logged under -v; avoid printing them twice.
        if event.kind == EventKind.REVIEW:
            visible = False
        if self.dynamic:
            self.progress.update(
                self.overall, total=max(1, self.state.total), completed=self.state.completed
            )
            if event.kind == EventKind.PROGRESS:
                self.progress.tasks[self.stage_task].total = event.total
                self.progress_detail = format_pipeline_event(event, self.symbols)
                self.progress.update(
                    self.stage_task,
                    description=event.message,
                    total=event.total,
                    completed=event.current or 0,
                    visible=True,
                )
            elif event.kind == EventKind.START and event.stage not in {
                PipelineStage.RUN,
                PipelineStage.VIDEO,
            }:
                self.progress_detail = ""
                self.progress.tasks[self.stage_task].total = event.total
                self.progress.reset(
                    self.stage_task,
                    description=event.message,
                    total=event.total,
                    completed=0,
                    visible=True,
                )
            elif event.kind in {EventKind.COMPLETE, EventKind.REUSED}:
                if detail is None:
                    self.progress.update(self.stage_task, visible=False)
                else:
                    self.progress.tasks[self.stage_task].total = None
                    self.progress.reset(
                        self.stage_task,
                        description=self.state.stages[event.stage.value].message,
                        total=None,
                        visible=True,
                    )
                    if detail.startswith("load_"):
                        self.console.print(
                            Text(format_pipeline_event(event, self.symbols), style="success")
                        )
            now = time.monotonic()
            if now - self.last_refresh >= 0.125 or event.kind != EventKind.PROGRESS:
                self.live.update(self.render(), refresh=True)
                self.last_refresh = now
            if visible and event.stage == PipelineStage.VIDEO and event.kind == EventKind.COMPLETE:
                self.console.print(
                    Text(format_pipeline_event(event, self.symbols), style="success")
                )
        elif visible and self.throttle.accepts(event):
            self.console.print(
                Text(
                    format_pipeline_event(event, self.symbols),
                    style="success"
                    if event.kind == EventKind.COMPLETE
                    else "warning"
                    if event.kind == EventKind.WARNING
                    else "info",
                )
            )

    def render(self):
        table = Table.grid(padding=(0, 1))
        symbols = {
            "pending": self.symbols["pending"],
            "active": self.symbols["start"],
            "complete": self.symbols["complete"],
            "reused": self.symbols["reused"],
            "review": self.symbols["review"],
            "failed": self.symbols["failure"],
        }
        styles = {
            "pending": "pending",
            "active": "active",
            "complete": "success",
            "reused": "muted",
            "review": "review",
            "failed": "error",
        }
        order = {stage.value: index for index, stage in enumerate(PipelineStage)}
        for name, state in sorted(self.state.stages.items(), key=lambda item: order[item[0]]):
            detail = state.message
            if state.speed:
                detail += f" · {state.speed}"
            if state.elapsed is not None:
                detail += " · " + format_duration(state.elapsed)
            table.add_row(
                Text(symbols[state.status], style=styles[state.status]),
                Text(name, style="stage"),
                Text(detail),
            )
        return Group(
            self.progress,
            Text(self.progress_detail, style="muted"),
            Panel(Group(Text(self.state.video or "Course", style="accent"), table)),
        )

    def display_path(self, value) -> str:
        path = Path(value).resolve()
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def summary(
        self, *, title: str, elapsed: float, failures=(), logs=None, details: dict | None = None
    ):
        text = f"Videos: {self.state.total - len(failures)}/{self.state.total} · Elapsed: {format_duration(elapsed)}"
        if details:
            text = (
                " · ".join(f"{key}: {value}" for key, value in details.items())
                + "\nElapsed: "
                + format_duration(elapsed)
            )
        flagged = sum(item.get("flagged", 0) for item in self.state.quality.values())
        text += f"\nReview: {flagged} flagged sentences · Failures: {len(failures)}"
        for video, status in self.state.videos.items():
            text += f"\n{self.symbols[status]} {video}"
        for failure in failures:
            text += f"\n{failure}"
        self.console.print(
            Panel(Text(text), title=title, border_style="error" if failures else "success")
        )
        for video, stages in self.state.timings.items() if self.verbosity >= 0 else []:
            self.console.print(
                Text(
                    video
                    + " · "
                    + " · ".join(
                        f"{stage} {format_duration(duration)}" for stage, duration in stages.items()
                    ),
                    style="muted",
                )
            )
        for category, paths in self.state.artifacts.items():
            self.console.print(Text(category, style="accent"))
            for path in paths:
                self.console.print(Text(self.display_path(path), style="path"))
        if logs:
            self.console.print(
                Text(
                    f"Logs\nText: {self.display_path(logs.text)}\nJSONL: {self.display_path(logs.jsonl)}",
                    style="path",
                )
            )

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .context import EventContext, current_context, log_context


class PipelineStage(StrEnum):
    RUN = "run"
    VIDEO = "video"
    VALIDATE = "validate"
    NORMALIZE = "normalize"
    TOC = "toc"
    CARDS = "cards"
    CONCAT = "concatenate"
    CHAPTERS = "chapters"
    DISCOVER = "discover"
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    SUBTITLES = "subtitles"
    RENDER = "render"
    TTS = "tts"
    MUX = "mux"
    SPEEDUP = "speedup"
    COMPLETE = "complete"


class EventKind(StrEnum):
    START = "start"
    PROGRESS = "progress"
    COMPLETE = "complete"
    REVIEW = "review"
    WARNING = "warning"
    FAILURE = "failure"
    ARTIFACT = "artifact"
    REUSED = "reused"


class EventSeverity(StrEnum):
    NORMAL = "normal"
    REVIEW = "review"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    stage: PipelineStage
    message: str
    current: float | None = None
    total: float | None = None
    artifact: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)
    kind: EventKind = EventKind.PROGRESS
    context: EventContext = field(default_factory=current_context.get)
    timestamp: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="milliseconds")
    )

    @property
    def severity(self) -> EventSeverity:
        return {
            EventKind.REVIEW: EventSeverity.REVIEW,
            EventKind.WARNING: EventSeverity.WARNING,
            EventKind.FAILURE: EventSeverity.ERROR,
        }.get(self.kind, EventSeverity.NORMAL)

    def to_dict(self) -> dict[str, Any]:
        """Version 1: JSON primitives only; context fields are flattened, absent fields omitted."""
        return {
            "schema_version": 1,
            "timestamp": self.timestamp,
            **self.context.fields(),
            "kind": self.kind.value,
            "severity": self.severity.value,
            "stage": self.stage.value,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "artifact": str(self.artifact) if self.artifact else None,
            "details": self.details,
        }


class PipelineObserver(Protocol):
    def notify(self, event: PipelineEvent) -> None: ...


class NullObserver:
    def notify(self, event: PipelineEvent) -> None:
        del event


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def notify(self, event: PipelineEvent) -> None:
        self.events.append(event)


class CompositeObserver:
    def __init__(self, *observers: PipelineObserver) -> None:
        self.observers = observers

    def notify(self, event: PipelineEvent) -> None:
        for observer in self.observers:
            observer.notify(event)


class JsonEventObserver:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally overwriting a config, media, or earlier run.
        self.stream = path.open("x", encoding="utf-8")

    def notify(self, event: PipelineEvent) -> None:
        self.stream.write(json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


_NULL_OBSERVER = NullObserver()  # Stateless.
_observer: ContextVar[PipelineObserver] = ContextVar("pipeline_observer", default=_NULL_OBSERVER)


@contextmanager
def event_scope(observer: PipelineObserver | None = None, **context):
    token = _observer.set(observer if observer is not None else _observer.get())
    try:
        with log_context(**context):
            yield
    finally:
        _observer.reset(token)


def emit(
    stage: PipelineStage, message: str, *, kind: EventKind = EventKind.PROGRESS, **fields
) -> None:
    event = PipelineEvent(stage, message, kind=kind, **fields)
    if kind != EventKind.PROGRESS:
        with log_context(**{**event.context.fields(), "stage": stage.value}):
            logging.getLogger(__name__).debug(
                "Event %s: %s; %s; artifact=%s",
                kind.value,
                message,
                event.details,
                event.artifact,
                extra={"diagnostic_only": True},
            )
    _observer.get().notify(event)


@contextmanager
def stage_context(
    stage: PipelineStage,
    message: str,
    *,
    operation: str | None = None,
    total: float | None = None,
    details: dict | None = None,
    completed: str | None = None,
    reused: bool = False,
):
    """One lifecycle and monotonic duration; nested operations do not complete their parent stage."""
    with log_context(
        stage=stage.value, operation=operation, chunk=current_context.get().chunk, subtitle=None
    ):
        started = time.perf_counter()
        emit(stage, message, kind=EventKind.START, total=total, details=details or {})
        try:
            yield
        except Exception as exc:
            emit(
                stage,
                str(exc),
                kind=EventKind.FAILURE,
                details={"elapsed_seconds": time.perf_counter() - started},
            )
            raise
        else:
            emit(
                stage,
                completed or message,
                kind=EventKind.REUSED if reused else EventKind.COMPLETE,
                current=total,
                total=total,
                details={"elapsed_seconds": time.perf_counter() - started},
            )


def ffmpeg_events(stage: PipelineStage, message: str, total: float | None = None):
    def notify(progress):
        # Negative timestamps and unknown/nonpositive durations cannot define a percentage.
        current = (
            max(0.0, progress.elapsed_seconds) if progress.elapsed_seconds is not None else None
        )
        emit(
            stage,
            message,
            current=current,
            total=total if total and total > 0 else None,
            context=replace(current_context.get(), operation="ffmpeg"),
            details={"speed": progress.speed, "state": progress.state, "unit": "seconds"},
        )

    return notify


def warn(logger: logging.Logger, message: str, *args) -> None:
    """Persist important warnings and render them once through the active frontend."""
    observed = not isinstance(_observer.get(), NullObserver)
    logger.warning(message, *args, extra={"diagnostic_only": observed})
    if observed:
        stage = PipelineStage(current_context.get().stage or PipelineStage.DISCOVER)
        emit(stage, message % args if args else message, kind=EventKind.WARNING)

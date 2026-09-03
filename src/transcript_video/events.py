from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class PipelineStage(StrEnum):
    DISCOVER = "discover"
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    SUBTITLES = "subtitles"
    RENDER = "render"
    TTS = "tts"
    MUX = "mux"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    stage: PipelineStage
    message: str
    current: float | None = None
    total: float | None = None
    artifact: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)


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

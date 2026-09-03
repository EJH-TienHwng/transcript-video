from __future__ import annotations

from rich.progress import Progress, TaskID

from ..events import PipelineEvent, PipelineStage


class RichProgressObserver:
    """Translate core pipeline events into a caller-owned Rich progress task."""

    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self.progress = progress
        self.task_id = task_id

    def notify(self, event: PipelineEvent) -> None:
        self.progress.update(self.task_id, description=event.message)
        if event.stage is PipelineStage.COMPLETE:
            self.progress.advance(self.task_id)

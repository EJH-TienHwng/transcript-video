"""Invocation context shared by domain events and diagnostic logging."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True, slots=True)
class EventContext:
    run_id: str | None = None
    video: str | None = None
    stage: str | None = None
    chunk: int | None = None
    subtitle: int | None = None
    operation: str | None = None

    def fields(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


_EMPTY_CONTEXT = EventContext()  # Frozen; safe to share across contexts.
current_context: ContextVar[EventContext] = ContextVar("pipeline_context", default=_EMPTY_CONTEXT)


@contextmanager
def log_context(**fields):
    token = current_context.set(replace(current_context.get(), **fields))
    try:
        yield current_context.get()
    finally:
        current_context.reset(token)

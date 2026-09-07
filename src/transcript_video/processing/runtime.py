"""Invocation-owned model reuse with only one active accelerator workload."""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature

from ..context import log_context
from ..events import EventKind, PipelineStage, emit

_runtime: ContextVar[ModelRuntime | None] = ContextVar("model_runtime", default=None)


def _empty_cuda_cache():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move(kind, model, device):
    if kind == "whisper":
        if device == "cpu":
            model.model.unload_model(to_cpu=True)
        else:
            model.model.load_model()
    elif kind == "translation":
        model[1].to(device)
    else:
        model.model.to(device)
        if kind == "qwen":
            # Qwen's codec wrapper is not an nn.Module child; move its weights explicitly.
            model.model.speech_tokenizer.model.to(device)


class ModelRuntime:
    def __init__(self):
        self.models = {}
        self.active = None

    def get(self, kind, key, load):
        cached = self.models.get(kind)
        if kind != "aligner" and self.active != (kind, key):
            self.park()
        if cached is not None and cached[0] != key:
            del self.models[kind]
            cached = None
            gc.collect()
            _empty_cuda_cache()
        if cached is None:
            model = load()
            if model is None:
                return None
            device = str(
                model[1].device
                if kind == "translation"
                else model.model.device
                if kind in {"whisper", "aligner"}
                else model.device
            )
            # faster-whisper exposes device through its CTranslate2 model.
            self.models[kind] = (key, model, device)
        else:
            _, model, device = cached
            if kind != "aligner" and self.active != (kind, key) and device.startswith("cuda"):
                _move(kind, model, device)
            with log_context(operation="load_" + kind):
                emit(
                    PipelineStage.TTS
                    if kind in {"qwen", "aligner"}
                    else PipelineStage.TRANSLATE
                    if kind == "translation"
                    else PipelineStage.TRANSCRIBE,
                    "Reusing loaded " + kind,
                    kind=EventKind.REUSED,
                )
        if kind != "aligner":
            self.active = (kind, key)
        return model

    def park(self):
        if self.active is not None:
            kind, _ = self.active
            _, model, device = self.models[kind]
            if device.startswith("cuda"):
                _move(kind, model, "cpu")
                _empty_cuda_cache()
            self.active = None

    def close(self):
        self.active = None
        self.models.clear()
        gc.collect()
        _empty_cuda_cache()


@contextmanager
def model_runtime():
    # ponytail: one CPU-resident instance per backend; use stage batches if host RAM becomes limiting.
    runtime = ModelRuntime()
    token = _runtime.set(runtime)
    try:
        yield runtime
    finally:
        _runtime.reset(token)
        runtime.close()


def reuse_model(kind):
    def decorate(load):
        parameters = signature(load)

        @wraps(load)
        def wrapped(*args, **kwargs):
            runtime = _runtime.get()
            if runtime is None:
                return load(*args, **kwargs)
            bound = parameters.bind(*args, **kwargs)
            bound.apply_defaults()
            key = tuple((name, str(value)) for name, value in bound.arguments.items())
            return runtime.get(kind, key, lambda: load(*args, **kwargs))

        return wrapped

    return decorate

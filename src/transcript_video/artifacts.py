"""Publish complete artifacts with a same-directory replace on Windows and Linux."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def is_partial_artifact(path: Path) -> bool:
    return path.name.startswith(".") and path.stem.endswith(".partial")


@contextmanager
def atomic_output(destination: Path):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=f".partial{destination.suffix}",
        dir=destination.parent,
    )
    os.close(descriptor)  # FFmpeg/SoundFile must reopen it; required on Windows.
    temporary = Path(name)
    temporary.unlink()
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_audio(destination: Path, audio, sample_rate: int) -> None:
    import soundfile as sf

    with atomic_output(destination) as temporary:
        sf.write(str(temporary), audio, sample_rate)


def write_text(destination: Path, text: str) -> None:
    with atomic_output(destination) as temporary:
        temporary.write_text(text, encoding="utf-8")

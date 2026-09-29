from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG_PATH, RunSettings
from .settings import resolve_settings


@dataclass(slots=True)
class VideoOverride:
    """A sparse patch over batch settings for one selected video."""

    profile: str | Path | None = None
    values: dict[str, Any] = field(default_factory=dict)
    translated_srt: Path | None = None


@dataclass(slots=True)
class BatchSettings:
    """User intent for a batch; effective settings are resolved only when needed."""

    videos: list[Path]
    config_path: Path | None = DEFAULT_CONFIG_PATH
    profile: str | Path | None = None
    values: dict[str, Any] = field(default_factory=dict)
    overrides: dict[Path, VideoOverride] = field(default_factory=dict)
    require_config: bool = False

    def resolve_items(self) -> list[ProcessItem]:
        items: list[ProcessItem] = []
        for video in self.videos:
            override = self.overrides.get(video, VideoOverride())
            resolved = resolve_settings(
                config_path=self.config_path,
                profile=override.profile if override.profile is not None else self.profile,
                overrides={**self.values, **override.values},
                require_config=self.require_config,
            )
            items.append(ProcessItem(video, resolved.settings, override.translated_srt))
        return items


@dataclass(frozen=True, slots=True)
class ProcessItem:
    """One video and its independently resolved effective settings."""

    video: Path
    settings: RunSettings
    translated_srt: Path | None = None


def selected_overrides(videos: Sequence[Path], values: dict[str, Any]) -> dict[Path, VideoOverride]:
    """Build sparse patches without replacing unrelated per-video choices."""
    return {video: VideoOverride(values=dict(values)) for video in videos}

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import questionary
from questionary import Choice
from rich.console import Console
from rich.table import Table

from .application.batch import BatchSettings, VideoOverride
from .application.processing import (
    BatchProcessPlan,
    build_batch_process_plan,
    execute_batch_process_plan,
)
from .config import DEFAULT_CONFIG_PATH, ProjectPaths, find_project_root
from .processing.media import find_videos
from .ui.progress import RichProgressObserver
from .ui.theme import questionary_style

PROMPT_STYLE = questionary_style(False)


@contextmanager
def launcher_ui(console: Console):
    global PROMPT_STYLE
    previous = console._color_system
    previous_style = PROMPT_STYLE
    if console.no_color:
        console._color_system = None
    PROMPT_STYLE = questionary_style(bool(console.no_color))
    try:
        yield
    finally:
        console._color_system = previous
        PROMPT_STYLE = previous_style


def interaction_available() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


class InteractiveBatchController:
    """Stateful editor for sparse batch and per-video overrides."""

    def __init__(
        self,
        root: Path | None = None,
        config_path: Path | None = DEFAULT_CONFIG_PATH,
        profile: str | Path | None = None,
    ) -> None:
        self.root = (root or find_project_root()).expanduser().resolve()
        self.batch = BatchSettings([], config_path=config_path, profile=profile)

    def discover_videos(self) -> list[Path]:
        input_dir = ProjectPaths.from_root(self.root).input_dir
        return find_videos(input_dir) if input_dir.is_dir() else []

    def select(self, videos: list[Path]) -> None:
        self.batch.videos = list(dict.fromkeys(videos))

    def update_defaults(self, values: dict[str, Any], profile: str | Path | None = None) -> None:
        self.batch.values.update(
            {name: value for name, value in values.items() if value is not None}
        )
        if profile is not None:
            self.batch.profile = profile

    def apply(
        self,
        videos: list[Path],
        values: dict[str, Any],
        profile: str | Path | None = None,
    ) -> None:
        for video in videos:
            existing = self.batch.overrides.get(video, VideoOverride())
            existing.values.update(
                {name: value for name, value in values.items() if value is not None}
            )
            if profile is not None:
                existing.profile = profile
            self.batch.overrides[video] = existing

    def reset(self, videos: list[Path]) -> None:
        for video in videos:
            self.batch.overrides.pop(video, None)

    def resolved_items(self):
        self.batch.values["project.root"] = str(self.root)
        return self.batch.resolve_items()

    def build_plan(self, *, defer_video_errors: bool = False) -> BatchProcessPlan:
        return build_batch_process_plan(
            self.resolved_items(), defer_video_errors=defer_video_errors
        )

    def preview_rows(self) -> list[tuple[str, str, str, str, str]]:
        rows = []
        for item in self.resolved_items():
            settings = item.settings
            override = self.batch.overrides.get(item.video)
            rows.append(
                (
                    item.video.name,
                    str(
                        override.profile
                        if override and override.profile
                        else self.batch.profile or "default"
                    ),
                    "yes" if settings.tts.enabled else "no",
                    "yes" if settings.speedup.enabled else "no",
                    "ASR" if settings.transcription.overwrite_srt else "-",
                )
            )
        return rows


def _ask(kind: str, message: str, **kwargs):
    answer = getattr(questionary, kind)(message, style=PROMPT_STYLE, **kwargs).unsafe_ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def _profiles(root: Path) -> list[str]:
    return sorted(path.stem for path in (root / "configs/profiles").glob("*.toml"))


def _basic_values(controller: InteractiveBatchController) -> dict[str, Any]:
    current = controller.resolved_items()[0].settings if controller.batch.videos else None
    return {
        "hardware.device": _ask(
            "select",
            "Device:",
            choices=["cuda", "cpu"],
            default=current.hardware.device if current else "cuda",
        ),
        "hardware.video_encoder": _ask(
            "select",
            "Video encoder:",
            choices=["auto", "h264_nvenc", "libx264"],
            default=current.hardware.video_encoder if current else "auto",
        ),
        "tts.enabled": _ask(
            "confirm", "Enable TTS?", default=current.tts.enabled if current else False
        ),
        "tts.verify_final_audio": _ask(
            "confirm",
            "Verify final audio?",
            default=current.tts.verify_final_audio if current else False,
        ),
        "speedup.enabled": _ask(
            "confirm", "Enable speed-up?", default=current.speedup.enabled if current else False
        ),
        "transcription.overwrite_srt": _ask(
            "confirm",
            "Force transcription?",
            default=current.transcription.overwrite_srt if current else False,
        ),
    }


def _customize(controller: InteractiveBatchController, console: Console) -> None:
    while True:
        action = _ask(
            "select",
            "Per-video settings:",
            choices=[
                "Edit one video",
                "Apply basic settings to selected videos",
                "Apply basic settings to all videos",
                "Reset selected overrides",
                "Done",
            ],
        )
        if action == "Done":
            return
        selected = controller.batch.videos
        if action in {
            "Edit one video",
            "Apply basic settings to selected videos",
            "Reset selected overrides",
        }:
            if action == "Edit one video":
                selected = [_ask("select", "Video:", choices=controller.batch.videos)]
            else:
                selected = _ask(
                    "checkbox",
                    "Videos:",
                    choices=[Choice(video.name, value=video) for video in selected],
                )
        if not selected:
            continue
        if action == "Reset selected overrides":
            controller.reset(selected)
            continue
        values = _basic_values(controller)
        profile = _ask(
            "select",
            "Profile:",
            choices=[Choice("Batch default", value=None), *_profiles(controller.root)],
        )
        controller.apply(selected, values, profile)
        console.print(f"Updated {len(selected)} video(s).")


def _review(controller: InteractiveBatchController, console: Console) -> str:
    table = Table(title="Process plan")
    for name in ("Video", "Profile", "TTS", "Speed-up", "Force"):
        table.add_column(name)
    for row in controller.preview_rows():
        table.add_row(*row)
    console.print(table)
    return _ask("select", "Next:", choices=["Back", "Dry run", "Run", "Cancel"])


def _run_batch(controller: InteractiveBatchController, console: Console, *, dry_run: bool) -> None:
    plan = controller.build_plan(defer_video_errors=not dry_run)
    if dry_run:
        for row in controller.preview_rows():
            console.print(" · ".join(row))
        return
    with RichProgressObserver(console, root=plan.plans[0].root) as progress:
        summary = execute_batch_process_plan(plan, progress)
    console.print(
        f"Completed: {summary.succeeded}/{summary.total} · "
        f"Waiting for translation: {len(summary.waiting_for_translation)}"
    )
    for failure in summary.failures:
        console.print(failure)


def launch(console: Console, root: Path | None = None) -> None:
    """Run the guided launcher. It only invokes application services directly."""
    with launcher_ui(console):
        while True:
            action = _ask(
                "select",
                "TRANSCRIPT·VIDEO\n\nWhat do you want to do?",
                choices=[
                    "Process videos",
                    "Speed up videos",
                    "Course builder",
                    "Inspect media",
                    "Diagnostics",
                    "Configuration",
                    "Open full TUI",
                    "Exit",
                ],
            )
            if action == "Exit":
                return
            if action != "Process videos":
                console.print(f"Use the direct command for now: transcript-video {action.lower()}")
                continue
            controller = InteractiveBatchController(root)
            choices = [Choice(video.name, value=video) for video in controller.discover_videos()]
            selected = _ask("checkbox", "Select videos:", choices=choices)
            if not selected:
                continue
            controller.select(selected)
            profile = _ask(
                "select",
                "Batch profile:",
                choices=[Choice("Default", value=None), *_profiles(controller.root)],
            )
            controller.update_defaults(_basic_values(controller), profile)
            _customize(controller, console)
            while True:
                next_action = _review(controller, console)
                if next_action == "Back":
                    _customize(controller, console)
                elif next_action == "Cancel":
                    break
                else:
                    _run_batch(controller, console, dry_run=next_action == "Dry run")
                    if next_action == "Run":
                        break

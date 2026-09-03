from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import click
import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .application.settings import ResolvedSettings, resolve_settings
from .config import (
    DEFAULT_CONFIG_PATH,
    RunSettings,
    save_run_settings,
)
from .ui.console import ConsolePair, make_consoles
from .ui.logging import configure_logging
from .ui.progress import RichProgressObserver

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="Local video transcription, translation, TTS, and course building.",
    no_args_is_help=True,
    invoke_without_command=True,
)
course_app = typer.Typer(help="Create and build multi-session courses.")
config_app = typer.Typer(help="Inspect and validate reusable run configuration.")
app.add_typer(course_app, name="course")
app.add_typer(config_app, name="config")


class ForceTarget(StrEnum):
    transcription = "transcription"
    translation = "translation"
    tts = "tts"
    render = "render"


class TaskChoice(StrEnum):
    translate = "translate"
    transcribe = "transcribe"


class DeviceChoice(StrEnum):
    cuda = "cuda"
    cpu = "cpu"


class EncoderChoice(StrEnum):
    auto = "auto"
    h264_nvenc = "h264_nvenc"
    libx264 = "libx264"


class TTSModeChoice(StrEnum):
    timed = "timed"
    simple = "simple"


class GenerationChoice(StrEnum):
    chunked = "chunked"
    full = "full"


class AudioChoice(StrEnum):
    replace = "replace"
    mix = "mix"


@dataclass(slots=True)
class CLIState:
    consoles: ConsolePair
    verbosity: int
    log_file: Path
    json_output: bool

    def logging(self, *, write_file: bool = True) -> None:
        configure_logging(self.consoles.err, self.verbosity, self.log_file if write_file else None)


@app.callback()
def global_options(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, help="Show the application version."),
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable ANSI colors.")] = False,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Only show warnings and errors.")
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            "-v", "--verbose", count=True, help="Increase detail; use -vv for tracebacks."
        ),
    ] = 0,
    log_file: Annotated[
        Path, typer.Option("--log-file", help="Detailed rotating log file.")
    ] = Path("logs/transcript-video.log"),
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit JSON for read-only commands.")
    ] = False,
) -> None:
    if version:
        typer.echo("transcript-video 0.3.0")
        raise typer.Exit()
    ctx.obj = CLIState(
        make_consoles(no_color=no_color), -1 if quiet else min(verbose, 2), log_file, json_output
    )


def _state(ctx: typer.Context) -> CLIState:
    return ctx.ensure_object(CLIState)


def _resolved(
    config: Path | None, profile: str | None, overrides: dict[str, Any], explicit: bool = False
) -> ResolvedSettings:
    return resolve_settings(
        config_path=config, profile=profile, overrides=overrides, require_config=explicit
    )


@app.command("process")
def process_command(
    ctx: typer.Context,
    video: Annotated[
        Path | None, typer.Argument(help="Video to process; omit to scan data/input.")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Base TOML run config.")
    ] = DEFAULT_CONFIG_PATH,
    profile: Annotated[
        str | None, typer.Option("--profile", help="Profile name or TOML path.")
    ] = None,
    save_config: Annotated[
        Path | None, typer.Option("--save-config", help="Save effective settings.")
    ] = None,
    root: Annotated[Path | None, typer.Option("--root")] = None,
    legacy_video: Annotated[
        str | None, typer.Option("--video", help="Deprecated alias for VIDEO.", hidden=True)
    ] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    translation_model: Annotated[str | None, typer.Option("--translation-model")] = None,
    task: Annotated[TaskChoice | None, typer.Option("--task")] = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    device: Annotated[DeviceChoice | None, typer.Option("--device")] = None,
    compute_type: Annotated[str | None, typer.Option("--compute-type")] = None,
    video_encoder: Annotated[EncoderChoice | None, typer.Option("--video-encoder")] = None,
    translation_batch_size: Annotated[
        int | None, typer.Option("--translation-batch-size", min=1)
    ] = None,
    overwrite_srt: Annotated[
        bool | None, typer.Option("--overwrite-srt/--no-overwrite-srt")
    ] = None,
    skip_burn: Annotated[bool | None, typer.Option("--skip-burn/--no-skip-burn")] = None,
    enable_tts: Annotated[bool | None, typer.Option("--enable-tts/--no-enable-tts")] = None,
    overwrite_tts: Annotated[
        bool | None, typer.Option("--overwrite-tts/--no-overwrite-tts")
    ] = None,
    tts_mode: Annotated[TTSModeChoice | None, typer.Option("--tts-mode")] = None,
    tts_generation_mode: Annotated[
        GenerationChoice | None, typer.Option("--tts-generation-mode")
    ] = None,
    rerun_tts_chunk: Annotated[int | None, typer.Option("--rerun-tts-chunk", min=0)] = None,
    tts_model: Annotated[str | None, typer.Option("--tts-model")] = None,
    tts_language: Annotated[str | None, typer.Option("--tts-language")] = None,
    tts_speaker: Annotated[str | None, typer.Option("--tts-speaker")] = None,
    tts_instruct: Annotated[str | None, typer.Option("--tts-instruct")] = None,
    tts_attn_implementation: Annotated[
        str | None, typer.Option("--tts-attn-implementation")
    ] = None,
    audio_mode: Annotated[AudioChoice | None, typer.Option("--audio-mode")] = None,
    split_tts_audio: Annotated[
        bool | None, typer.Option("--split-tts-audio/--no-split-tts-audio")
    ] = None,
    tts_chunk_minutes: Annotated[int | None, typer.Option("--tts-chunk-minutes", min=1)] = None,
    tts_max_speedup: Annotated[float | None, typer.Option("--tts-max-speedup", min=1.0)] = None,
    tts_chunk_tail_seconds: Annotated[
        float | None, typer.Option("--tts-chunk-tail-seconds", min=0)
    ] = None,
    force: Annotated[
        list[ForceTarget] | None,
        typer.Option("--force", help="Rebuild a repeatable pipeline target."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and print the execution plan without writes."),
    ] = False,
) -> None:
    state = _state(ctx)
    state.logging(write_file=not dry_run)
    selected_video = legacy_video or (str(video) if video else None)
    force_set = set(force or [])
    overrides = {
        "project.root": str(root) if root else None,
        "project.video": selected_video,
        "project.model": model,
        "project.translation_model": translation_model,
        "hardware.device": device,
        "hardware.compute_type": compute_type,
        "hardware.video_encoder": video_encoder,
        "transcription.task": task,
        "transcription.language": language,
        "transcription.translation_batch_size": translation_batch_size,
        "transcription.overwrite_srt": True
        if force_set & {ForceTarget.transcription, ForceTarget.translation}
        else overwrite_srt,
        "transcription.skip_burn": skip_burn,
        "tts.enabled": enable_tts,
        "tts.overwrite": True if ForceTarget.tts in force_set else overwrite_tts,
        "tts.mode": tts_mode,
        "tts.generation_mode": tts_generation_mode,
        "tts.rerun_chunk": rerun_tts_chunk,
        "tts.model": tts_model,
        "tts.language": tts_language,
        "tts.speaker": tts_speaker,
        "tts.instruct": tts_instruct,
        "tts.attn_implementation": tts_attn_implementation,
        "tts.audio_mode": audio_mode,
        "tts.split_audio": split_tts_audio,
        "tts.chunk_minutes": tts_chunk_minutes,
        "tts.max_speedup": tts_max_speedup,
        "tts.chunk_tail_seconds": tts_chunk_tail_seconds,
    }
    resolved = _resolved(config, profile, overrides, explicit=_option_present("--config"))
    if save_config:
        if dry_run:
            state.consoles.out.print(f"[warning]Would save config:[/] {save_config.resolve()}")
        else:
            save_run_settings(resolved.settings, save_config)
    _run_processing(state, resolved.settings, dry_run=dry_run)


def _run_processing(state: CLIState, settings: RunSettings, *, dry_run: bool) -> None:
    from .application.processing import build_process_plan, execute_process_plan

    plan = build_process_plan(settings)
    if dry_run:
        table = Table(title="Dry-run execution plan")
        table.add_column("Item", style="info")
        table.add_column("Resolved value")
        table.add_row("Videos", "\n".join(str(item) for item in plan.videos))
        table.add_row("Model", str(plan.model))
        table.add_row("Translation model", str(plan.translation_model or "disabled"))
        table.add_row("Artifacts", "\n".join(str(path) for path in plan.artifacts))
        table.add_row(
            "Device / encoder",
            f"{settings.hardware.device} / {settings.hardware.video_encoder} (resolved at runtime)",
        )
        table.add_row("TTS", "enabled" if settings.tts.enabled else "disabled")
        state.consoles.out.print(table)
        return
    with Progress(
        SpinnerColumn(),
        TextColumn("[stage]{task.description}"),
        TimeElapsedColumn(),
        console=state.consoles.out,
    ) as progress:
        task_id = progress.add_task("Processing videos", total=len(plan.videos))
        summary = execute_process_plan(plan, RichProgressObserver(progress, task_id))
    artifact_text = "\n".join(str(path) for path in summary.artifacts) or "None"
    state.consoles.out.print(
        Panel.fit(
            f"Videos: {summary.succeeded}/{summary.total}\n"
            f"Elapsed: {summary.elapsed_seconds:.1f}s\nArtifacts:\n{artifact_text}",
            title="[success]Run complete[/]",
        )
    )
    if summary.failures:
        for failure in summary.failures:
            logger.error("%s", failure)
        raise RuntimeError("Failed videos: " + ", ".join(summary.failures))


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    sources: Annotated[bool, typer.Option("--sources")] = False,
) -> None:
    state = _state(ctx)
    state.logging()
    resolved = _resolved(config, profile, {}, explicit=True)
    payload = asdict(resolved.settings)
    if state.json_output:
        state.consoles.out.print_json(
            data={"settings": payload, "sources": resolved.sources if sources else None}
        )
        return
    table = Table(title="Effective configuration")
    table.add_column("Setting", style="info")
    table.add_column("Value")
    if sources:
        table.add_column("Source", style="dim")
    for section, values in payload.items():
        for key, value in values.items():
            row = [f"{section}.{key}", str(value)]
            if sources:
                row.append(resolved.sources[f"{section}.{key}"])
            table.add_row(*row)
    state.consoles.out.print(table)


@config_app.command("validate")
def config_validate(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    state = _state(ctx)
    state.logging()
    resolved = _resolved(config, profile, {}, explicit=True)
    state.consoles.out.print(f"[success]Valid configuration:[/] {resolved.config_path}")


@app.command("inspect")
def inspect_command(
    ctx: typer.Context,
    video: Annotated[Path, typer.Argument()],
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    from .application.inspection import inspect_video

    state = _state(ctx)
    state.logging()
    result = inspect_video(
        video, _resolved(config, profile, {}, explicit=_option_present("--config")).settings
    )
    if state.json_output:
        state.consoles.out.print_json(data=result)
    else:
        state.consoles.out.print(
            Panel.fit(json.dumps(result, indent=2, ensure_ascii=False), title="Media inspection")
        )


@app.command("doctor")
def doctor_command(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    from .application.diagnostics import run_doctor

    state = _state(ctx)
    state.logging()
    checks = run_doctor(
        _resolved(config, profile, {}, explicit=_option_present("--config")).settings
    )
    if state.json_output:
        state.consoles.out.print_json(data=[asdict(check) for check in checks])
    else:
        table = Table(title="Environment doctor")
        table.add_column("Status")
        table.add_column("Check")
        table.add_column("Detail")
        for check in checks:
            status = (
                "[success]PASS[/]"
                if check.ok
                else "[error]FAIL[/]"
                if check.required
                else "[warning]WARN[/]"
            )
            table.add_row(status, check.name, check.detail)
        state.consoles.out.print(table)
    if any(not item.ok and item.required for item in checks):
        raise typer.Exit(1)


@course_app.command("build")
def course_build(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Course JSON config.")],
) -> None:
    from .course.builder import build_course
    from .course.config import load_course_config

    state = _state(ctx)
    state.logging()
    result = build_course(load_course_config(config))
    state.consoles.out.print(f"[success]Course built:[/] [path]{result}[/]")


@course_app.command("create")
def course_create(
    ctx: typer.Context,
    root: Annotated[Path | None, typer.Option("--root")] = None,
    video_dir: Annotated[Path | None, typer.Option("--video-dir")] = None,
) -> None:
    from .course.wizard import create_course_config_interactive

    _state(ctx).logging()
    create_course_config_interactive(
        root=root.resolve() if root else None, output_dir=video_dir.resolve() if video_dir else None
    )


@course_app.command("tui")
def course_tui(
    ctx: typer.Context, config: Annotated[Path | None, typer.Option("--config")] = None
) -> None:
    _state(ctx).logging()
    from .tui.app import CourseApp

    CourseApp(config_path=config).run()


def _option_present(option: str) -> bool:
    return option in sys.argv or any(item.startswith(option + "=") for item in sys.argv)


def _normalize_legacy_argv(argv: list[str]) -> list[str]:
    commands = {"process", "course", "config", "inspect", "doctor"}
    if (
        not argv
        or any(item in commands for item in argv)
        or argv[0] in {"--help", "-h", "--version"}
        or argv[0].startswith(("--show-completion", "--install-completion"))
    ):
        return argv
    prefix: list[str] = []
    remaining = list(argv)
    while remaining and (
        remaining[0] in {"--no-color", "--json", "-q", "--quiet", "-v", "-vv", "--verbose"}
        or remaining[0].startswith("-vv")
    ):
        prefix.append(remaining.pop(0))
    if remaining and remaining[0] == "--log-file":
        prefix.extend(remaining[:2])
        remaining = remaining[2:]
    return [*prefix, "process", *remaining]


def parse_args(argv: list[str] | None = None) -> tuple[object, RunSettings]:
    """Compatibility parser for integrations using the pre-Typer Python API."""
    import argparse

    raw = list(argv or [])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device")
    parser.add_argument("--language")
    parser.add_argument("--enable-tts", action="store_true", default=None)
    known, _ = parser.parse_known_args(raw)
    overrides = {
        "hardware.device": known.device,
        "transcription.language": known.language,
        "tts.enabled": known.enable_tts,
    }
    resolved = resolve_settings(
        config_path=known.config,
        overrides=overrides,
        require_config=_option_in(raw, "--config"),
    )
    return known, resolved.settings


def _option_in(argv: list[str], option: str) -> bool:
    return option in argv or any(item.startswith(option + "=") for item in argv)


def main() -> None:
    argv = _normalize_legacy_argv(sys.argv[1:])
    try:
        result = app(args=argv, prog_name="transcript-video", standalone_mode=False)
        if isinstance(result, int) and result:
            raise SystemExit(result)
    except (click.ClickException, ValueError, FileNotFoundError, RuntimeError) as exc:
        consoles = make_consoles(no_color="--no-color" in argv)
        consoles.err.print(f"[error]Error:[/] {exc}")
        if "-vv" in argv:
            consoles.err.print_exception()
        raise SystemExit(2 if isinstance(exc, click.UsageError) else 1) from None
    except KeyboardInterrupt:
        make_consoles(no_color="--no-color" in argv).err.print("[warning]Cancelled by user.[/]")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

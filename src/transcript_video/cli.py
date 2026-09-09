from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .application.settings import ResolvedSettings, resolve_settings
from .config import (
    DEFAULT_CONFIG_PATH,
    ProjectPaths,
    RunSettings,
    configure_binary_path,
    save_run_settings,
)
from .context import log_context
from .events import CompositeObserver, JsonEventObserver
from .ui.console import ConsolePair, help_output, make_consoles
from .ui.logging import RunLogs, close_logging, configure_logging
from .ui.progress import RichProgressObserver, format_duration

try:
    from typer import TyperException
except ImportError:  # Typer before it vendored Click.
    from click import ClickException as TyperException

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="Vietnamese transcription, external English subtitle handoff, TTS, and course building.",
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
)
course_app = typer.Typer(help="Create and build multi-session courses.")
config_app = typer.Typer(help="Inspect and validate reusable run configuration.")
app.add_typer(course_app, name="course")
app.add_typer(config_app, name="config")


class ForceTarget(StrEnum):
    transcription = "transcription"


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
    plain: bool = False
    run_id: str = ""
    logs: RunLogs | None = None

    def logging(
        self, *, write_file: bool = True, command: str | None = None, console_enabled: bool = True
    ) -> None:
        self.logs = configure_logging(
            self.consoles.err,
            self.verbosity,
            self.log_file,
            run_id=self.run_id if command else None,
            command=command or "command",
            write_files=write_file,
            console_enabled=console_enabled,
        )
        logger.debug("Invocation started: %s", command or "read-only")


@app.callback()
def global_options(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", is_eager=True, help="Show the application version."),
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable ANSI colors.")] = False,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Warnings, errors, and final result; no progress.")
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            "-v", "--verbose", count=True, help="Increase detail; use -vv for tracebacks."
        ),
    ] = 0,
    log_file: Annotated[
        Path,
        typer.Option(
            "--log-file",
            help="Global rotating text log; per-run DEBUG text/JSONL go in its sibling runs/ directory.",
        ),
    ] = Path("logs/transcript-video.log"),
    plain: Annotated[
        bool, typer.Option("--plain", help="Line-based status without live redraw.")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit JSON for read-only commands.")
    ] = False,
) -> None:
    if version:
        typer.echo("transcript-video 0.3.0")
        raise typer.Exit()
    ctx.obj = CLIState(
        make_consoles(no_color=no_color),
        -1 if quiet else min(verbose, 2),
        log_file,
        json_output,
        plain,
        datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:10],
    )
    ctx.with_resource(log_context(run_id=ctx.obj.run_id))
    ctx.call_on_close(close_logging)


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
    videos: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Videos in processing order (names in data/input or absolute paths); omit to use project.video or scan data/input."
        ),
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
    translated_srt: Annotated[
        Path | None,
        typer.Option(
            "--translated-srt",
            help="Use an externally translated English SRT for subtitle rendering and English TTS.",
        ),
    ] = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    device: Annotated[DeviceChoice | None, typer.Option("--device")] = None,
    compute_type: Annotated[str | None, typer.Option("--compute-type")] = None,
    video_encoder: Annotated[EncoderChoice | None, typer.Option("--video-encoder")] = None,
    overwrite_srt: Annotated[
        bool | None, typer.Option("--overwrite-srt/--no-overwrite-srt")
    ] = None,
    skip_burn: Annotated[bool | None, typer.Option("--skip-burn/--no-skip-burn")] = None,
    enable_tts: Annotated[bool | None, typer.Option("--enable-tts/--no-enable-tts")] = None,
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
    verify_final_audio: Annotated[
        bool | None,
        typer.Option(
            "--verify-final-audio/--no-verify-final-audio",
            help="ASR text review of final timed sentences (extra CPU inference).",
        ),
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
    enable_speedup: Annotated[
        bool | None,
        typer.Option(
            "--speedup/--no-speedup",
            help="Create an additional speed-up video from the final rendered video.",
        ),
    ] = None,
    speedup_spec: Annotated[
        Path | None,
        typer.Option(
            "--speedup-spec",
            help="Speed-up TOML for one video; passing it enables speed-up.",
        ),
    ] = None,
    force: Annotated[
        list[ForceTarget] | None,
        typer.Option("--force", help="Regenerate the Vietnamese source transcription."),
    ] = None,
    events_json: Annotated[
        Path | None,
        typer.Option(
            "--events-json", help="Write semantic JSONL events to a new file (exclusive creation)."
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and print the execution plan without writes."),
    ] = False,
) -> None:
    state = _state(ctx)
    state.logging(write_file=not dry_run, command="process")
    if videos and legacy_video:
        raise typer.BadParameter("Use positional VIDEOS or --video, not both.")
    selected_video = legacy_video or (str(videos[0]) if videos and len(videos) == 1 else None)
    force_set = set(force or [])
    overrides = {
        "project.root": str(root) if root else None,
        "project.video": selected_video,
        "project.model": model,
        "hardware.device": device,
        "hardware.compute_type": compute_type,
        "hardware.video_encoder": video_encoder,
        "transcription.language": language,
        "transcription.overwrite_srt": True
        if ForceTarget.transcription in force_set
        else overwrite_srt,
        "transcription.skip_burn": skip_burn,
        "tts.enabled": enable_tts,
        "tts.mode": tts_mode,
        "tts.generation_mode": tts_generation_mode,
        "tts.rerun_chunk": rerun_tts_chunk,
        "tts.model": tts_model,
        "tts.language": tts_language,
        "tts.speaker": tts_speaker,
        "tts.instruct": tts_instruct,
        "tts.attn_implementation": tts_attn_implementation,
        "tts.verify_final_audio": verify_final_audio,
        "tts.audio_mode": audio_mode,
        "tts.split_audio": split_tts_audio,
        "tts.chunk_minutes": tts_chunk_minutes,
        "tts.max_speedup": tts_max_speedup,
        "tts.chunk_tail_seconds": tts_chunk_tail_seconds,
        "speedup.enabled": True if speedup_spec is not None else enable_speedup,
        "speedup.spec": str(speedup_spec.expanduser().resolve()) if speedup_spec else None,
    }
    resolved = _resolved(
        config,
        profile,
        overrides,
        explicit=ctx.get_parameter_source("config").name == "COMMANDLINE",
    )
    if save_config:
        if dry_run:
            state.consoles.out.print(f"[warning]Would save config:[/] {save_config.resolve()}")
        else:
            save_run_settings(resolved.settings, save_config)
    _run_processing(
        state,
        resolved.settings,
        dry_run=dry_run,
        videos=videos,
        translated_srt=translated_srt,
        events_json=events_json,
    )


def _run_processing(
    state: CLIState,
    settings: RunSettings,
    *,
    dry_run: bool,
    videos: list[Path] | None = None,
    translated_srt: Path | None = None,
    events_json: Path | None = None,
) -> None:
    from .application.processing import build_process_plan, execute_process_plan

    try:
        plan = build_process_plan(settings, videos, translated_srt)
    except (ValueError, OSError) as exc:
        if dry_run:
            raise
        if events_json:
            from .events import EventKind, PipelineEvent, PipelineStage

            writer = JsonEventObserver(events_json)
            try:
                writer.notify(
                    PipelineEvent(
                        PipelineStage.RUN, "Validating execution plan", kind=EventKind.START
                    )
                )
                writer.notify(PipelineEvent(PipelineStage.RUN, str(exc), kind=EventKind.FAILURE))
            finally:
                writer.close()
        _show_error(state, exc)
        raise typer.Exit(1) from None
    if dry_run:
        table = Table(title="Dry-run execution plan")
        table.add_column("Item", style="info")
        table.add_column("Resolved value")
        table.add_row("Videos", "\n".join(str(item) for item in plan.videos))
        table.add_row("Model", str(plan.model))
        table.add_row(
            "Source subtitles",
            "\n".join(
                f"{'exists' if path.is_file() else 'missing'} · {path}"
                for path in plan.source_srt_paths
            ),
        )
        table.add_row(
            "Translated subtitles",
            "\n".join(
                f"{'exists' if path.is_file() else 'missing'} · {path}"
                for path in plan.translated_srt_paths
            ),
        )
        table.add_row(
            "Workflow",
            "\n".join(
                f"{video.name}: "
                + ("render/TTS can proceed" if translated.is_file() else "translation handoff")
                for video, translated in zip(plan.videos, plan.translated_srt_paths, strict=True)
            ),
        )
        table.add_row("Artifacts", "\n".join(str(path) for path in plan.artifacts))
        table.add_row(
            "Device / encoder",
            f"{settings.hardware.device} / {settings.hardware.video_encoder} (resolved at runtime)",
        )
        table.add_row("TTS", "enabled" if settings.tts.enabled else "disabled")
        if settings.speedup.enabled:
            from .processing.speedup import parse_speedup_spec

            spec_rows = []
            for path in plan.speedup_spec_paths:
                detail = "missing"
                if path.is_file():
                    segments = parse_speedup_spec(path)
                    factors = (
                        ", ".join(f"\N{MULTIPLICATION SIGN}{item.speed}" for item in segments)
                        or "0 segments"
                    )
                    detail = f"exists · {len(segments)} segments · {factors}"
                spec_rows.append(f"{detail} · {path}")
            table.add_row("Speed-up", "enabled")
            table.add_row("Speed-up spec", "\n".join(spec_rows))
            table.add_row(
                "Normal output", "\n".join(str(path) for path in plan.normal_output_paths)
            )
            table.add_row(
                "Speed-up output", "\n".join(str(path) for path in plan.speedup_output_paths)
            )
        state.consoles.out.print(table)
        return
    if state.verbosity >= 0:
        state.consoles.out.print(
            Panel(
                Text(
                    f"Processing {len(plan.videos)} videos · {settings.hardware.device} · encoder {settings.hardware.video_encoder}\n"
                    f"ASR: {plan.model.name} · TTS: {settings.tts.model if settings.tts.enabled else 'disabled'}"
                ),
                title="transcript-video",
                border_style="accent",
            )
        )
    with ExitStack() as stack:
        progress = stack.enter_context(
            RichProgressObserver(
                state.consoles.out, verbosity=state.verbosity, plain=state.plain, root=plan.root
            )
        )
        observer = progress
        if events_json:
            json_events = JsonEventObserver(events_json)
            stack.callback(json_events.close)
            observer = CompositeObserver(progress, json_events)
        try:
            summary = execute_process_plan(plan, observer)
        except Exception as exc:
            logger.debug("Processing failed", exc_info=True, extra={"diagnostic_only": True})
            if progress.live:
                progress.live.stop()
            _show_error(state, exc)
            raise typer.Exit(1) from None
    progress.summary(
        title="Run completed with errors" if summary.failures else "Run complete",
        elapsed=summary.elapsed_seconds,
        failures=summary.failures,
        logs=state.logs,
    )
    if summary.failures:
        if state.verbosity >= 2:
            from rich.traceback import Traceback

            for error in summary.errors:
                state.consoles.err.print(
                    Traceback.from_exception(
                        type(error), error, error.__traceback__, show_locals=False
                    )
                )
        raise typer.Exit(1)


def _show_error(state: CLIState, exc: Exception) -> None:
    from .application.errors import describe_error

    message = describe_error(exc)
    if state.logs:
        message += f"\n\nLog: {state.logs.text}"
    state.consoles.err.print(Panel(Text(message), title="Processing failed", border_style="error"))
    if state.verbosity >= 2:
        state.consoles.err.print_exception(show_locals=False)


@app.command("speedup")
def speedup_command(
    ctx: typer.Context,
    video: Annotated[
        Path,
        typer.Argument(help="Original video name used to resolve the final rendered output."),
    ],
    spec: Annotated[Path | None, typer.Option("--spec", help="Custom speed-up TOML path.")] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Base TOML run config.")
    ] = DEFAULT_CONFIG_PATH,
    profile: Annotated[
        str | None, typer.Option("--profile", help="Profile name or TOML path.")
    ] = None,
    root: Annotated[Path | None, typer.Option("--root")] = None,
    video_encoder: Annotated[EncoderChoice | None, typer.Option("--video-encoder")] = None,
) -> None:
    """Regenerate only the speed-up artifact from an existing final video."""
    from .events import event_scope
    from .processing.speedup import process_speedup_video

    state = _state(ctx)
    state.logging(command="speedup")
    resolved = _resolved(
        config,
        profile,
        {
            "project.root": str(root) if root else None,
            "hardware.video_encoder": video_encoder,
        },
        explicit=ctx.get_parameter_source("config").name == "COMMANDLINE",
    )
    settings = resolved.settings
    project_root = Path(settings.project.root).expanduser().resolve()
    paths = ProjectPaths.from_root(project_root)
    normal = paths.normal_video_path(video, tts_enabled=settings.tts.enabled)
    spec_path = paths.speedup_spec_path(video, spec or settings.speedup.spec)
    output = paths.speedup_output_path(normal)
    configure_binary_path(project_root)
    started = time.perf_counter()
    with RichProgressObserver(
        state.consoles.out,
        verbosity=state.verbosity,
        plain=state.plain,
        root=project_root,
    ) as progress:
        try:
            with event_scope(progress, video=video.name):
                result = process_speedup_video(
                    normal,
                    spec_path,
                    output,
                    video_encoder=settings.hardware.video_encoder,
                    video_stem=video.stem,
                )
        except (ValueError, OSError, RuntimeError) as exc:
            if progress.live:
                progress.live.stop()
            _show_error(state, exc)
            raise typer.Exit(1) from None
    progress.summary(
        title="Speed-up complete" if result else "Speed-up skipped",
        elapsed=time.perf_counter() - started,
        failures=(),
        logs=state.logs,
        details=(
            {
                "Original": format_duration(result.original_duration),
                "Segments": len(result.segments),
                "Output": format_duration(result.actual_duration),
                "Saved": format_duration(result.time_saved),
            }
            if result
            else {"Result": "No speed-up video created"}
        ),
    )


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
    try:
        resolved = _resolved(config, profile, {}, explicit=True)
    except (ValueError, OSError) as exc:
        if state.json_output:
            state.consoles.out.print_json(
                data={"valid": False, "config": str(config.resolve()), "error": str(exc)}
            )
        else:
            state.consoles.err.print(
                Panel(Text(str(exc)), title="Configuration invalid", border_style="error")
            )
        raise typer.Exit(1) from None
    checks = [
        "TOML syntax",
        "Known sections",
        "Known fields",
        "Enum values",
        "Hardware settings",
        "Translation settings",
        "TTS settings",
        "Profile resolution",
    ]
    if state.json_output:
        state.consoles.out.print_json(
            data={
                "valid": True,
                "config": str(resolved.config_path),
                "profile": str(resolved.profile_path) if resolved.profile_path else None,
                "checks": checks,
            }
        )
    else:
        table = Table(title="Configuration validation")
        table.add_column("Status")
        table.add_column("Check")
        for check in checks:
            table.add_row("PASS", check)
        state.consoles.out.print(table)
        state.consoles.out.print(
            Text(f"Valid configuration: {resolved.config_path}", style="success")
        )


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
        video,
        _resolved(
            config, profile, {}, explicit=ctx.get_parameter_source("config").name == "COMMANDLINE"
        ).settings,
    )
    if state.json_output:
        state.consoles.out.print_json(data=result)
    else:
        from .ui.inspection import inspection_table

        state.consoles.out.print(inspection_table(result))


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
        _resolved(
            config, profile, {}, explicit=ctx.get_parameter_source("config").name == "COMMANDLINE"
        ).settings
    )
    if state.json_output:
        state.consoles.out.print_json(data=[asdict(check) for check in checks])
    else:
        groups = {"Environment": [], "Media": [], "Project": [], "Hardware": []}
        for check in checks:
            group = (
                "Environment"
                if check.name in {"Python", "Free storage"}
                else "Media"
                if check.name
                in {
                    "FFmpeg",
                    "ffprobe",
                    "Subtitle filter",
                    "Configured encoder",
                    "NVENC encoder availability",
                    "NVENC runtime usability",
                    "Fallback encoder",
                }
                else "Hardware"
                if check.name
                in {
                    "CUDA",
                    "PyTorch",
                    "PyTorch version",
                    "PyTorch CUDA build",
                    "ASR compute support",
                }
                else "Project"
            )
            groups[group].append(check)
        for name, items in groups.items():
            table = Table(title=name, expand=True, border_style="muted")
            table.add_column("Status", width=6)
            table.add_column("Check", style="info")
            table.add_column("Detail", overflow="fold")
            for check in items:
                label, style = (
                    ("PASS", "success")
                    if check.ok
                    else ("FAIL", "error")
                    if check.required
                    else ("WARN", "warning")
                )
                table.add_row(Text(label, style=style), Text(check.name), Text(check.detail))
            state.consoles.out.print(table)
        required = sum(not item.ok and item.required for item in checks)
        optional = sum(not item.ok and not item.required for item in checks)
        state.consoles.out.print(
            Text(
                f"{required} required checks failed · {optional} optional checks failed"
                if required or optional
                else "Environment ready",
                style="error" if required else "warning" if optional else "success",
            )
        )
    if any(not item.ok and item.required for item in checks):
        raise typer.Exit(1)


@course_app.command("build")
def course_build(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", help="Course JSON config.")],
    events_json: Annotated[
        Path | None,
        typer.Option("--events-json", help="Write semantic JSONL events to a new file."),
    ] = None,
) -> None:
    from .course.builder import build_course
    from .course.config import load_course_config

    state = _state(ctx)
    state.logging(command="course-build")
    started = time.perf_counter()
    with ExitStack() as stack:
        progress = stack.enter_context(
            RichProgressObserver(state.consoles.out, verbosity=state.verbosity, plain=state.plain)
        )
        observers = [progress]
        if events_json:
            json_events = JsonEventObserver(events_json)
            stack.callback(json_events.close)
            observers.append(json_events)
        try:
            build_course(load_course_config(config), CompositeObserver(*observers))
        except Exception as exc:
            logger.debug("Course build failed", exc_info=True, extra={"diagnostic_only": True})
            if progress.live:
                progress.live.stop()
            _show_error(state, exc)
            raise typer.Exit(1) from None
    details = progress.state.course_details
    progress.summary(
        title="Course built",
        elapsed=time.perf_counter() - started,
        logs=state.logs,
        details={
            "Sessions": details.get("sessions", "?"),
            "Duration": format_duration(details.get("duration", 0)),
            "Chapters": details.get("chapters", 0),
        },
    )


@course_app.command("create")
def course_create(
    ctx: typer.Context,
    root: Annotated[Path | None, typer.Option("--root")] = None,
    video_dir: Annotated[Path | None, typer.Option("--video-dir")] = None,
) -> None:
    from .course.wizard import create_course_config_interactive, wizard_ui

    state = _state(ctx)
    state.logging(command="course-create", console_enabled=False)
    with wizard_ui(state.consoles.out):
        create_course_config_interactive(
            root=root.resolve() if root else None,
            output_dir=video_dir.resolve() if video_dir else None,
        )


@course_app.command("tui")
def course_tui(
    ctx: typer.Context, config: Annotated[Path | None, typer.Option("--config")] = None
) -> None:
    _state(ctx).logging(command="course-tui", console_enabled=False)
    from .tui.app import CourseApp

    # Textual installs its monochrome filter during construction, not on attribute assignment.
    previous_no_color = os.environ.get("NO_COLOR")
    if _state(ctx).consoles.out.no_color:
        os.environ["NO_COLOR"] = "1"
    try:
        tui = CourseApp(config_path=config)
    finally:
        if previous_no_color is None:
            os.environ.pop("NO_COLOR", None)
        else:
            os.environ["NO_COLOR"] = previous_no_color
    tui.run()


def _normalize_legacy_argv(argv: list[str]) -> list[str]:
    commands = {"process", "speedup", "course", "config", "inspect", "doctor"}
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
        remaining[0]
        in {"--plain", "--no-color", "--json", "-q", "--quiet", "-v", "-vv", "--verbose"}
        or remaining[0].startswith("-vv")
    ):
        prefix.append(remaining.pop(0))
    if remaining and remaining[0] == "--log-file":
        prefix.extend(remaining[:2])
        remaining = remaining[2:]
    if remaining in (["--help"], ["-h"], ["--version"]):
        return [*prefix, *remaining]
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


def legacy_course_main(executable: str, command: str) -> None:
    """Keep global options before the delegated subcommand in compatibility executables."""
    make_consoles(no_color="--no-color" in sys.argv).err.print(
        f"Warning: {executable} is deprecated; use 'transcript-video course {command}'.",
        style="warning",
    )
    global_args, remaining = [], []
    arguments = iter(sys.argv[1:])
    for argument in arguments:
        if argument in {
            "--no-color",
            "--plain",
            "--json",
            "-q",
            "--quiet",
            "-v",
            "-vv",
            "--verbose",
        } or argument.startswith("--log-file="):
            global_args.append(argument)
        elif argument == "--log-file":
            global_args.append(argument)
            value = next(arguments, None)
            if value is None:
                make_consoles(no_color="--no-color" in sys.argv).err.print(
                    "Error: --log-file requires a path.", style="error"
                )
                raise SystemExit(2)
            global_args.append(value)
        else:
            remaining.append(argument)
    sys.argv[1:] = [*global_args, "course", command, *remaining]
    main()


def main() -> None:
    argv = _normalize_legacy_argv(sys.argv[1:])
    try:
        with help_output("--no-color" in argv):
            result = app(args=argv, prog_name="transcript-video", standalone_mode=False)
        if isinstance(result, int) and result:
            raise SystemExit(result)
    except (TyperException, ValueError, OSError, RuntimeError) as exc:
        consoles = make_consoles(no_color="--no-color" in argv)
        consoles.err.print(f"[error]Error:[/] {exc}")
        if "-vv" in argv:
            consoles.err.print_exception(show_locals=False)
        raise SystemExit(exc.exit_code if isinstance(exc, TyperException) else 1) from None
    except (KeyboardInterrupt, EOFError):
        make_consoles(no_color="--no-color" in argv).err.print("[warning]Cancelled by user.[/]")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

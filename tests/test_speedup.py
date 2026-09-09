from __future__ import annotations

import json
import math
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from transcript_video.cli import app
from transcript_video.config import ProjectPaths, RunSettings, SubtitleSegment
from transcript_video.hardware import get_ffmpeg_exe, get_ffprobe_exe
from transcript_video.process_runner import probe_media, run_ffmpeg, run_process
from transcript_video.processing.speedup import (
    SpeedupSegment,
    TimelineSegment,
    build_atempo_chain,
    build_overlay_text,
    build_speedup_filter_complex,
    build_speedup_timeline,
    calculate_speedup_duration,
    parse_speedup_spec,
    parse_speedup_timestamp,
    process_speedup_outputs,
    process_speedup_video,
    validate_speedup_segments,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:01:25", 85),
        ("00:01:25.500", 85.5),
        ("00:01:25,500", 85.5),
        ("01:02:03.250", 3723.25),
        ("01:25", 85),
        ("01:25.500", 85.5),
    ],
)
def test_parse_speedup_timestamp(value, expected):
    assert parse_speedup_timestamp(value) == expected


@pytest.mark.parametrize("value", ["abc", "00:61:00", "00:00:60", "-1", "1:02.1234"])
def test_rejects_invalid_speedup_timestamp(value):
    with pytest.raises(ValueError, match="timestamp"):
        parse_speedup_timestamp(value)


def test_parse_speedup_spec_with_optional_label(tmp_path):
    spec = tmp_path / "video.speedup.toml"
    spec.write_text(
        '[[segment]]\nstart = "00:01"\nend = "00:02.500"\nspeed = 5\nlabel = "Running build..."\n',
        encoding="utf-8",
    )
    assert parse_speedup_spec(spec) == (SpeedupSegment(1, 2.5, 5, "Running build..."),)


def test_malformed_speedup_spec_fails_explicitly(tmp_path):
    spec = tmp_path / "video.speedup.toml"
    spec.write_text("abc =", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid TOML"):
        parse_speedup_spec(spec)


@pytest.mark.parametrize("content", ["", "# Reviewed: no speed-up required.\n"])
def test_empty_and_comment_only_specs_mean_no_speedup(tmp_path, content):
    spec = tmp_path / "video.speedup.toml"
    spec.write_text(content, encoding="utf-8")
    assert parse_speedup_spec(spec) == ()


@pytest.mark.parametrize("speed", [1, 3, 20])
def test_rejects_unsupported_speed(speed):
    with pytest.raises(ValueError, match="2, 5, 10"):
        validate_speedup_segments([SpeedupSegment(1, 2, speed)], 10)


@pytest.mark.parametrize(
    "segment",
    [
        SpeedupSegment(-1, 2, 2),
        SpeedupSegment(1, 1, 2),
        SpeedupSegment(2, 1, 2),
        SpeedupSegment(1, 11, 2),
    ],
)
def test_rejects_invalid_speedup_bounds(segment):
    with pytest.raises(ValueError):
        validate_speedup_segments([segment], 10)


def test_speedup_validation_sorts_accepts_adjacent_and_rejects_overlap():
    later = SpeedupSegment(20, 30, 10)
    earlier = SpeedupSegment(10, 20, 5)
    assert validate_speedup_segments([later, earlier], 40) == (earlier, later)
    with pytest.raises(ValueError, match="Speed-up intervals overlap"):
        validate_speedup_segments([earlier, SpeedupSegment(19, 25, 2)], 40)


def test_timeline_and_duration_include_normal_intervals():
    segments = (SpeedupSegment(100, 200, 10), SpeedupSegment(300, 400, 5))
    timeline = build_speedup_timeline(segments, 600)
    assert timeline == (
        TimelineSegment(0, 100),
        TimelineSegment(100, 200, 10),
        TimelineSegment(200, 300),
        TimelineSegment(300, 400, 5),
        TimelineSegment(400, 600),
    )
    assert calculate_speedup_duration(600, segments) == pytest.approx((430, 170))
    assert calculate_speedup_duration(600, [segments[0]]) == pytest.approx((510, 90))


@pytest.mark.parametrize("speed", [2, 5, 10])
def test_atempo_chain_has_requested_product(speed):
    factors = [float(item.split("=", 1)[1]) for item in build_atempo_chain(speed).split(",")]
    assert math.prod(factors) == speed
    assert all(0.5 <= factor <= 2 for factor in factors)


def test_custom_label_is_kept_out_of_filter_syntax(tmp_path):
    label = "Build: 50% 'quoted' \"double\" \\ path, comma"
    segment = TimelineSegment(1, 2, 5, label)
    textfile = tmp_path / "overlay.txt"
    graph = build_speedup_filter_complex((segment,), {0: textfile})
    assert build_overlay_text(segment) == f"{label}\nSpeed up \N{MULTIPLICATION SIGN}5"
    assert label not in graph
    assert "textfile=" in graph and "expansion=none" in graph


def test_missing_spec_creates_template_without_touching_normal_video(tmp_path):
    source = tmp_path / "normal.mp4"
    source.write_bytes(b"normal")
    spec = tmp_path / "specs" / "lesson.speedup.toml"
    output = tmp_path / "normal_speedup.mp4"
    assert (
        process_speedup_video(source, spec, output, video_encoder="libx264", video_stem="lesson")
        is None
    )
    assert source.read_bytes() == b"normal"
    assert "Allowed speed values: 2, 5, 10" in spec.read_text(encoding="utf-8")
    assert not output.exists()


@pytest.mark.parametrize("content", ["", "# Reviewed: no speed-up required.\n"])
def test_empty_spec_skips_without_probing_or_rendering(tmp_path, monkeypatch, content):
    from transcript_video.processing import speedup

    source = tmp_path / "normal.mp4"
    source.write_bytes(b"normal")
    spec = tmp_path / "lesson.speedup.toml"
    spec.write_text(content, encoding="utf-8")
    output = tmp_path / "normal_speedup.mp4"
    probe = Mock(side_effect=AssertionError("empty specs must not probe or render"))
    monkeypatch.setattr(speedup, "get_media_duration_seconds", probe)
    assert (
        process_speedup_video(source, spec, output, video_encoder="libx264", video_stem="lesson")
        is None
    )
    assert source.read_bytes() == b"normal"
    assert not output.exists() and not probe.called


def test_speedup_outputs_processes_both_variants_with_one_spec(tmp_path, monkeypatch):
    from transcript_video.processing import speedup

    sources = (tmp_path / "lesson_vi-dub_en-sub.mp4", tmp_path / "lesson_en-dub_en-sub.mp4")
    for source in sources:
        source.touch()
    spec = tmp_path / "lesson.speedup.toml"
    spec.write_text('[[segment]]\nstart="00:00"\nend="00:01"\nspeed=2\n')
    process = Mock(return_value=None)
    monkeypatch.setattr(speedup, "process_speedup_video", process)
    assert (
        process_speedup_outputs(sources, spec, video_encoder="libx264", video_stem="lesson") == ()
    )
    assert [call.args[:2] for call in process.call_args_list] == [
        (sources[0], spec),
        (sources[1], spec),
    ]
    assert [call.args[2] for call in process.call_args_list] == [
        tmp_path / "lesson_vi-dub_en-sub_speedup.mp4",
        tmp_path / "lesson_en-dub_en-sub_speedup.mp4",
    ]


def test_speedup_outputs_missing_spec_creates_only_one_template(tmp_path):
    sources = (tmp_path / "lesson_vi-dub_en-sub.mp4", tmp_path / "lesson_en-dub_en-sub.mp4")
    for source in sources:
        source.write_bytes(b"normal")
    spec = tmp_path / "specs/lesson.speedup.toml"
    assert (
        process_speedup_outputs(sources, spec, video_encoder="libx264", video_stem="lesson") == ()
    )
    assert spec.is_file()
    assert all(source.read_bytes() == b"normal" for source in sources)
    assert not any(tmp_path.glob("*_speedup.mp4"))


def test_invalid_spec_preserves_existing_normal_and_speedup_output(tmp_path, monkeypatch):
    from transcript_video.processing import speedup

    source = tmp_path / "normal.mp4"
    output = tmp_path / "normal_speedup.mp4"
    spec = tmp_path / "lesson.speedup.toml"
    source.write_bytes(b"normal")
    output.write_bytes(b"previous")
    spec.write_text(
        '[[segment]]\nstart="00:01"\nend="00:03"\nspeed=5\n'
        '[[segment]]\nstart="00:02"\nend="00:04"\nspeed=10\n'
    )
    monkeypatch.setattr(speedup, "get_media_duration_seconds", lambda _path: 10.0)
    with pytest.raises(ValueError, match="overlap"):
        process_speedup_video(source, spec, output, video_encoder="libx264", video_stem="lesson")
    assert source.read_bytes() == b"normal"
    assert output.read_bytes() == b"previous"


def test_pipeline_runs_speedup_after_normal_render(tmp_path, monkeypatch):
    from transcript_video.processing import pipeline

    paths = ProjectPaths.from_root(tmp_path)
    paths.create_dirs()
    video = paths.input_dir / "lesson.mp4"
    source = paths.source_subtitle_dir / "lesson_vi_faster.srt"
    translated = paths.translated_subtitle_dir / "lesson_en.srt"
    for path in (video, source, translated):
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nText\n", encoding="utf-8")
    settings = RunSettings.defaults()
    settings.speedup.enabled = True
    spec = paths.speedup_spec_path(video)
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text('[[segment]]\nstart="00:00"\nend="00:01"\nspeed=2\n')

    def fake_burn(_video, _srt, output, **_kwargs):
        output.write_bytes(b"normal")

    speedup = Mock()
    monkeypatch.setattr(pipeline, "burn_subtitles", fake_burn)
    monkeypatch.setattr(pipeline, "get_media_duration_seconds", lambda _path: 1.0)
    monkeypatch.setattr(pipeline, "process_speedup_outputs", speedup)
    pipeline.process_video(video, tmp_path / "model", source, translated, paths, settings)
    assert speedup.call_args.args[0] == (paths.normal_video_path(video, tts_enabled=False),)
    assert speedup.call_args.args[0][0].read_bytes() == b"normal"


def test_pipeline_with_tts_speeds_up_both_normal_outputs(tmp_path, monkeypatch):
    from transcript_video.processing import pipeline

    paths = ProjectPaths.from_root(tmp_path)
    paths.create_dirs()
    video = paths.input_dir / "lesson.mp4"
    source = paths.source_subtitle_dir / "lesson_vi_faster.srt"
    translated = paths.translated_subtitle_dir / "lesson_en.srt"
    video.touch()
    for path in (source, translated):
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nText\n", encoding="utf-8")
    settings = RunSettings.defaults()
    settings.tts.enabled = True
    settings.speedup.enabled = True

    def write_output(*args, **_kwargs):
        Path(args[-1]).write_bytes(b"normal")

    monkeypatch.setattr(pipeline, "burn_subtitles", write_output)
    monkeypatch.setattr(pipeline, "mux_audio_into_video_replace", write_output)
    monkeypatch.setattr(
        pipeline,
        "synthesize_tts_audio_by_time_chunks",
        Mock(return_value=[SubtitleSegment(0, 1, "Text")]),
    )
    monkeypatch.setattr(pipeline, "get_media_duration_seconds", lambda _path: 1.0)
    speedup = Mock()
    monkeypatch.setattr(pipeline, "process_speedup_outputs", speedup)
    pipeline.process_video(video, tmp_path / "model", source, translated, paths, settings)
    assert speedup.call_args.args[:2] == (
        paths.normal_video_paths(video, tts_enabled=True),
        paths.speedup_spec_path(video),
    )


def test_skip_burn_only_uses_existing_normal_outputs(tmp_path, monkeypatch):
    from transcript_video.processing import pipeline

    paths = ProjectPaths.from_root(tmp_path)
    paths.create_dirs()
    video = paths.input_dir / "lesson.mp4"
    source = paths.source_subtitle_dir / "lesson_vi_faster.srt"
    translated = paths.translated_subtitle_dir / "lesson_en.srt"
    for path in (video, source, translated):
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nText\n", encoding="utf-8")
    normal = paths.normal_video_path(video, tts_enabled=False)
    normal.touch()
    settings = RunSettings.defaults()
    settings.transcription.skip_burn = True
    settings.speedup.enabled = True
    burn = Mock(side_effect=AssertionError("skip-burn must not render"))
    speedup = Mock()
    monkeypatch.setattr(pipeline, "burn_subtitles", burn)
    monkeypatch.setattr(pipeline, "process_speedup_outputs", speedup)
    pipeline.process_video(video, tmp_path / "model", source, translated, paths, settings)
    burn.assert_not_called()
    assert speedup.call_args.args[0] == (normal,)


def test_pipeline_missing_spec_keeps_render_and_creates_template(tmp_path, monkeypatch):
    from transcript_video.processing import pipeline

    paths = ProjectPaths.from_root(tmp_path)
    paths.create_dirs()
    video = paths.input_dir / "lesson.mp4"
    source = paths.source_subtitle_dir / "lesson_vi_faster.srt"
    translated = paths.translated_subtitle_dir / "lesson_en.srt"
    video.touch()
    for path in (source, translated):
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nText\n", encoding="utf-8")
    settings = RunSettings.defaults()
    settings.speedup.enabled = True

    def fake_burn(_video, _srt, output, **_kwargs):
        output.write_bytes(b"normal")

    monkeypatch.setattr(pipeline, "burn_subtitles", fake_burn)
    monkeypatch.setattr(pipeline, "get_media_duration_seconds", lambda _path: 1.0)
    pipeline.process_video(video, tmp_path / "model", source, translated, paths, settings)
    assert paths.normal_video_path(video, tts_enabled=False).read_bytes() == b"normal"
    assert paths.speedup_spec_path(video).is_file()
    assert not paths.speedup_output_path(paths.normal_video_path(video, tts_enabled=False)).exists()


def test_speedup_dry_run_does_not_create_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "lesson.mp4"
    model = tmp_path / "model"
    video.touch()
    model.mkdir()
    (model / "model.bin").touch()
    result = CliRunner().invoke(
        app,
        ["process", str(video), "--model", str(model), "--speedup", "--dry-run"],
    )
    assert result.exit_code == 0, result.exception
    assert "Speed-up" in result.output and "missing" in result.output
    assert not (tmp_path / "data/speedup/lesson.speedup.toml").exists()


def test_explicit_speedup_spec_rejects_batch(tmp_path):
    from transcript_video.application.processing import build_process_plan

    paths = ProjectPaths.from_root(tmp_path)
    paths.input_dir.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").touch()
    videos = [paths.input_dir / "a.mp4", paths.input_dir / "b.mp4"]
    for video in videos:
        video.touch()
    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    settings.project.model = str(model)
    settings.speedup.enabled = True
    plan = build_process_plan(settings, videos)
    assert plan.speedup_spec_paths == tuple(paths.speedup_spec_path(video) for video in videos)
    settings.speedup.spec = "custom.toml"
    with pytest.raises(ValueError, match="one video"):
        build_process_plan(settings, videos)


def test_process_plan_reports_dual_outputs_only_for_configured_segments(tmp_path):
    from transcript_video.application.processing import build_process_plan

    paths = ProjectPaths.from_root(tmp_path)
    paths.input_dir.mkdir(parents=True)
    paths.speedup_dir.mkdir(parents=True)
    video = paths.input_dir / "lesson.mp4"
    video.touch()
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").touch()
    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    settings.project.model = str(model)
    settings.tts.enabled = True
    settings.speedup.enabled = True
    spec = paths.speedup_spec_path(video)
    spec.write_text('[[segment]]\nstart="00:00"\nend="00:01"\nspeed=2\n')
    plan = build_process_plan(settings, [video])
    assert plan.normal_output_paths == paths.normal_video_paths(video, tts_enabled=True)
    assert plan.speedup_output_paths == tuple(
        paths.speedup_output_path(path) for path in plan.normal_output_paths
    )
    spec.write_text("# Reviewed: no speed-up required.\n")
    plan = build_process_plan(settings, [video])
    assert plan.speedup_output_paths == ()
    assert not any(path.name.endswith("_speedup.mp4") for path in plan.artifacts)


def test_process_speedup_spec_option_implies_enabled(tmp_path, monkeypatch):
    from transcript_video import cli

    run = Mock()
    monkeypatch.setattr(cli, "_run_processing", run)
    monkeypatch.chdir(tmp_path)
    spec = tmp_path / "custom.toml"
    result = CliRunner().invoke(
        app, ["process", "lesson.mp4", "--speedup-spec", str(spec), "--dry-run"]
    )
    assert result.exit_code == 0, result.exception
    settings = run.call_args.args[1]
    assert settings.speedup.enabled
    assert settings.speedup.spec == str(spec.resolve())


@pytest.mark.parametrize("tts_enabled", [False, True])
def test_standalone_speedup_uses_existing_variant_without_core_pipeline(
    tmp_path, monkeypatch, tts_enabled
):
    from transcript_video.processing import speedup

    paths = ProjectPaths.from_root(tmp_path)
    paths.output_dir.mkdir(parents=True)
    normal = paths.normal_video_path("Analysis.mp4", tts_enabled=tts_enabled)
    normal.touch()
    called = Mock(return_value=None)
    monkeypatch.setattr(speedup, "process_speedup_video", called)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["speedup", "Analysis.mp4", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.exception
    assert called.call_args.args[:3] == (
        normal,
        paths.speedup_spec_path("Analysis.mp4"),
        paths.speedup_output_path(normal),
    )


def test_standalone_speedup_discovers_both_existing_variants(tmp_path, monkeypatch):
    from transcript_video.processing import speedup

    paths = ProjectPaths.from_root(tmp_path)
    paths.output_dir.mkdir(parents=True)
    normals = paths.normal_video_paths("Analysis.mp4", tts_enabled=True)
    for normal in normals:
        normal.touch()
    paths.speedup_dir.mkdir(parents=True)
    paths.speedup_spec_path("Analysis.mp4").write_text(
        '[[segment]]\nstart="00:00"\nend="00:01"\nspeed=2\n'
    )
    called = Mock(return_value=None)
    monkeypatch.setattr(speedup, "process_speedup_video", called)
    result = CliRunner().invoke(app, ["speedup", "Analysis.mp4", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.exception
    assert [call.args[0] for call in called.call_args_list] == list(normals)


def test_standalone_speedup_fails_when_no_rendered_variant_exists(tmp_path):
    result = CliRunner().invoke(app, ["speedup", "Analysis.mp4", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "No rendered normal video was found" in result.output


@pytest.mark.integration
def test_ffmpeg_speedup_preserves_audio_video_and_expected_duration(tmp_path, monkeypatch):
    from transcript_video.processing import speedup as speedup_module

    try:
        ffprobe = get_ffprobe_exe()
    except FileNotFoundError:
        ffprobe = None
        monkeypatch.setattr(
            speedup_module, "get_media_duration_seconds", Mock(side_effect=[6.0, 3.8])
        )
        monkeypatch.setattr(speedup_module, "media_has_audio", lambda _path: True)
    source = tmp_path / "source.mp4"
    output = tmp_path / "source_speedup.mp4"
    spec = tmp_path / "source.speedup.toml"
    label = "Build: 50% 'quoted' \"double\" \\ path, comma"
    run_ffmpeg(
        [
            get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=25:d=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=6",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    spec.write_text(
        '[[segment]]\nstart="00:01"\nend="00:02"\nspeed=2\n'
        f"label={json.dumps(label)}\n"
        '[[segment]]\nstart="00:03"\nend="00:04"\nspeed=5\n'
        '[[segment]]\nstart="00:04"\nend="00:05"\nspeed=10\n',
        encoding="utf-8",
    )
    result = process_speedup_video(
        source, spec, output, video_encoder="libx264", video_stem="source"
    )
    assert result is not None
    assert result.expected_duration == pytest.approx(3.8, abs=0.05)
    assert result.actual_duration == pytest.approx(3.8, abs=0.25)
    if ffprobe:
        metadata = probe_media(ffprobe, output)
        assert {stream["codec_type"] for stream in metadata["streams"]} >= {"audio", "video"}
    else:
        run_process(
            [
                get_ffmpeg_exe(),
                "-v",
                "error",
                "-i",
                output,
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-f",
                "null",
                os.devnull,
            ],
            tool="FFmpeg",
        )

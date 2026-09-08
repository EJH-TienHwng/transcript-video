import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from transcript_video.cli import app
from transcript_video.config import ProjectPaths, RunSettings, SubtitleSegment
from transcript_video.events import EventKind, RecordingObserver
from transcript_video.processing.subtitles import (
    plan_legacy_translated_subtitle_migration,
    read_srt,
    write_srt,
)


def _workflow(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)
    paths.create_dirs()
    video = tmp_path / "lesson.mp4"
    video.touch()
    model = tmp_path / "asr"
    model.mkdir()
    (model / "model.bin").touch()
    source = paths.source_subtitle_dir / "lesson_vi_faster.srt"
    translated = paths.translated_subtitle_dir / "lesson_en.srt"
    return paths, video, model, source, translated


def _patch_downstream(monkeypatch, pipeline):
    burn, tts, mux = Mock(), Mock(), Mock()
    monkeypatch.setattr(pipeline, "burn_subtitles", burn)
    monkeypatch.setattr(pipeline, "synthesize_tts_audio_by_time_chunks", tts)
    monkeypatch.setattr(pipeline, "mux_audio_into_video_replace", mux)
    monkeypatch.setattr(pipeline, "get_media_duration_seconds", lambda *args: 10)
    return burn, tts, mux


def test_missing_source_generates_then_stops_at_translation_handoff(tmp_path, monkeypatch):
    from transcript_video.processing import pipeline
    from transcript_video.processing.tts import core

    paths, video, model, source, translated = _workflow(tmp_path)
    asr = Mock(return_value=[SubtitleSegment(0, 1, "Xin chào")])
    monkeypatch.setattr(pipeline, "transcribe_video", asr)
    burn, tts, mux = _patch_downstream(monkeypatch, pipeline)
    qwen = Mock()
    monkeypatch.setattr(core, "load_qwen_tts_model", qwen)
    observer = RecordingObserver()

    pipeline.process_video(
        video, model, source, translated, paths, RunSettings.defaults(), observer
    )

    asr.assert_called_once()
    assert read_srt(source)[0].text == "Xin chào"
    assert not translated.exists()
    assert not burn.called and not tts.called and not mux.called
    assert not qwen.called
    handoff = next(
        event for event in observer.events if event.details.get("status") == "translation_handoff"
    )
    assert handoff.kind == EventKind.WARNING
    assert str(translated) in handoff.message
    assert "docs\\prompts\\optimal_prompt.md" in handoff.message


def test_existing_source_is_enough_without_model_or_config_invalidation(tmp_path, monkeypatch):
    from transcript_video.processing import pipeline

    paths, video, model, source, translated = _workflow(tmp_path)
    write_srt([SubtitleSegment(0, 1, "Nguồn")], source)
    asr = Mock(side_effect=AssertionError("ASR must not run"))
    monkeypatch.setattr(pipeline, "transcribe_video", asr)
    settings = RunSettings.defaults()
    settings.transcription.language = "en"

    pipeline.process_video(video, model, source, translated, paths, settings)

    assert not asr.called
    assert read_srt(source)[0].text == "Nguồn"


def test_cli_prints_translation_handoff(tmp_path, monkeypatch):
    _paths, video, model, source, translated = _workflow(tmp_path)
    write_srt([SubtitleSegment(0, 1, "Nguồn")], source)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "--plain",
            "process",
            str(video),
            "--root",
            str(tmp_path),
            "--model",
            str(model),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert "Waiting for translated English subtitles" in result.stdout
    assert translated.name in result.stdout
    assert "optimal_prompt.md" in result.stdout


def test_force_transcription_only_rewrites_source_and_english_drives_burn_and_tts(
    tmp_path, monkeypatch
):
    from transcript_video.processing import pipeline

    paths, video, model, source, translated = _workflow(tmp_path)
    write_srt([SubtitleSegment(0, 1, "Cũ")], source)
    translated_bytes = b"1\n00:00:00,000 --> 00:00:01,000\nManual English edit.\n"
    translated.write_bytes(translated_bytes)
    asr = Mock(return_value=[SubtitleSegment(0, 1, "Mới")])
    monkeypatch.setattr(pipeline, "transcribe_video", asr)
    burn, tts, mux = _patch_downstream(monkeypatch, pipeline)
    settings = RunSettings.defaults()
    settings.transcription.overwrite_srt = True
    settings.tts.enabled = True

    pipeline.process_video(video, model, source, translated, paths, settings)

    asr.assert_called_once()
    assert read_srt(source)[0].text == "Mới"
    assert translated.read_bytes() == translated_bytes
    assert burn.call_args.args[1] == translated
    assert tts.call_args.kwargs["segments"][0].text == "Manual English edit."
    assert tts.call_args.kwargs["regenerate_all_chunks"] is True
    assert mux.called
    assert not list(tmp_path.rglob("*.provenance.json"))


def test_explicit_translated_srt_override_and_canonical_plan(tmp_path):
    from transcript_video.application.processing import build_process_plan

    _paths, video, model, source, translated = _workflow(tmp_path)
    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    settings.project.model = str(model)
    plan = build_process_plan(settings, [video])
    assert plan.source_srt_paths == (source,)
    assert plan.translated_srt_paths == (translated,)

    override = tmp_path / "manual.srt"
    plan = build_process_plan(settings, [video], override)
    assert plan.translated_srt_paths == (override.resolve(),)


def test_asr_cannot_write_into_translated_namespace(tmp_path, monkeypatch):
    from transcript_video.processing import pipeline

    paths, video, model, _source, translated = _workflow(tmp_path)
    asr = Mock()
    monkeypatch.setattr(pipeline, "transcribe_video", asr)
    with pytest.raises(ValueError, match="must be written under"):
        pipeline.process_video(
            video,
            model,
            paths.translated_subtitle_dir / "wrong.srt",
            translated,
            paths,
            RunSettings.defaults(),
        )
    assert not asr.called


def test_migration_planning_is_non_recursive_and_collision_safe(tmp_path):
    root = tmp_path / "subtitles"
    root.mkdir()
    (root / "old_subtitle").mkdir()
    (root / "old_subtitle/archive_faster.srt").touch()
    (root / "lesson.srt").touch()
    (root / "lesson_faster.srt").touch()
    with pytest.raises(ValueError, match="same translated destination"):
        plan_legacy_translated_subtitle_migration(root)
    assert len(list(root.glob("*.srt"))) == 2
    assert (root / "old_subtitle/archive_faster.srt").exists()
    (root / "lesson.srt").unlink()
    (root / "translated").mkdir()
    (root / "translated/lesson_en.srt").touch()
    with pytest.raises(FileExistsError, match="already exists"):
        plan_legacy_translated_subtitle_migration(root)
    assert (root / "lesson_faster.srt").exists()


@pytest.mark.parametrize(
    "name,digest",
    [
        ("Analysis_en.srt", "9362ABF70DB59BFDF195CC3B7AEC40E7A2D28881B4ED8DE99824A65A3CDA26B9"),
        ("ASW_Demo_en.srt", "F7937601D1BD7689D2806F269FFB1A71CC2CB9D1235663B3D2A445DB63B6C3EC"),
        ("Find_Delta_en.srt", "A856981288B2D243C8CDE601686388CA46D4678EA01037315A381FD69AFE936F"),
        ("Report_en.srt", "6484DE35A0875D253D906CE2255B86BC9A1219F626D4C885386A628BAB9011B7"),
        (
            "Test_Case_Generation_en.srt",
            "C00197F27A3583F3954E3B8C582C1CC9CBBE640E93D5966F745313E67F2BA225",
        ),
        (
            "UT_Cantata_Demo_Version_1_en.srt",
            "314AEBE5539F7B67063131F1E5B53E5F357310F1E57FA217B5022212A995284D",
        ),
        (
            "UT_Gtest_Overview_en.srt",
            "739C1778216B30E5E8223EA7CEA5DEFCC81C1738E352BDDC7703AFB0CD803A2F",
        ),
    ],
)
def test_repository_english_subtitles_were_moved_without_content_changes(name, digest):
    path = Path(__file__).parents[1] / "data/subtitles/translated" / name
    content = path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest().upper() == digest
    assert not list(path.parents[1].glob("*.srt"))

from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest

from transcript_video.config import RunSettings, SubtitleSegment
from transcript_video.events import PipelineStage, RecordingObserver, event_scope
from transcript_video.processing.provenance import (
    cache_matches,
    subtitle_provenance,
    tts_provenance,
    write_provenance,
)
from transcript_video.processing.tts import chunks


@pytest.mark.parametrize(
    "field,value",
    [("speaker", "Ryan"), ("instruct", "New instruction"), ("verify_final_audio", True)],
)
def test_tts_fingerprint_changes_for_generation_settings(tmp_path, field, value):
    settings = RunSettings.defaults()
    audio = tmp_path / "a.wav"
    audio.touch()
    segments = [SubtitleSegment(0, 1, "one")]
    original = tts_provenance(segments, asdict(settings.tts))
    write_provenance(audio, original)
    assert cache_matches(audio, original, PipelineStage.TTS)
    setattr(settings.tts, field, value)
    observer = RecordingObserver()
    with event_scope(observer):
        assert not cache_matches(
            audio, tts_provenance(segments, asdict(settings.tts)), PipelineStage.TTS
        )
    assert field + " changed" in observer.events[0].message


def test_subtitle_provenance_tracks_config_and_input_identity(tmp_path):
    video, model, srt = [tmp_path / name for name in ("a.mp4", "model", "a.srt")]
    video.touch()
    model.mkdir()
    (model / "model.bin").touch()
    srt.touch()
    settings = RunSettings.defaults()
    provenance = subtitle_provenance(video, model, None, settings)
    write_provenance(srt, provenance)
    assert cache_matches(srt, provenance, PipelineStage.SUBTITLES)
    settings.transcription.language = "en"
    assert not cache_matches(
        srt, subtitle_provenance(video, model, None, settings), PipelineStage.SUBTITLES
    )
    settings.transcription.language = "vi"
    video.write_bytes(b"replacement")
    assert not cache_matches(
        srt, subtitle_provenance(video, model, None, settings), PipelineStage.SUBTITLES
    )


@pytest.mark.parametrize(
    "field,value", [("tts_speaker", "Ryan"), ("tts_instruct", "New instruction")]
)
def test_chunk_pipeline_reuses_then_regenerates_changed_voice(tmp_path, monkeypatch, field, value):
    model = Mock()
    model.generate_custom_voice.return_value = ([np.full(100, 0.2)], 100)
    loader = Mock(return_value=model)
    monkeypatch.setattr(chunks, "load_qwen_tts_model", loader)
    monkeypatch.setattr(chunks, "get_media_duration_seconds", lambda *args: 2)
    kwargs = dict(
        segments=[SubtitleSegment(0, 1, "one")],
        audio_out=tmp_path / "a.wav",
        chunks_dir=tmp_path / "chunks",
        video_path=tmp_path / "a.mp4",
        tts_model_name="local",
        tts_language="English",
        tts_speaker="Aiden",
        tts_instruct="steady",
        device="cpu",
        attn_implementation="sdpa",
        review_log_path=tmp_path / "reports/review.jsonl",
    )
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    assert loader.call_count == 1
    kwargs[field] = value
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    assert loader.call_count == 2
    assert (
        model.generate_custom_voice.call_args.kwargs[
            {"tts_speaker": "speaker", "tts_instruct": "instruct"}[field]
        ]
        == value
    )


def test_full_pipeline_cache_reuse_and_voice_changes(tmp_path, monkeypatch):
    from transcript_video.config import ProjectPaths
    from transcript_video.processing import pipeline
    from transcript_video.processing.tts import core

    paths = ProjectPaths.from_root(tmp_path)
    paths.create_dirs()
    video = tmp_path / "a.mp4"
    video.touch()
    asr_path = tmp_path / "asr"
    asr_path.mkdir()
    (asr_path / "model.bin").touch()
    settings = RunSettings.defaults()
    settings.tts.enabled = True
    settings.tts.generation_mode = "full"
    settings.tts.split_audio = False
    model = Mock()
    model.generate_custom_voice.return_value = ([np.full(100, 0.2)], 100)
    loader = Mock(return_value=model)
    asr = Mock(return_value=[SubtitleSegment(0.12345, 1.12345, "one")])
    monkeypatch.setattr(core, "load_qwen_tts_model", loader)
    monkeypatch.setattr(core, "load_faster_whisper_aligner", lambda *a: None)
    monkeypatch.setattr(pipeline, "transcribe_video", asr)
    for module in (core, pipeline):
        monkeypatch.setattr(module, "get_media_duration_seconds", lambda *a: 2)
    monkeypatch.setattr(pipeline, "burn_subtitles", Mock())
    monkeypatch.setattr(pipeline, "mux_audio_into_video_replace", Mock())
    for _ in range(2):
        pipeline.process_video(video, asr_path, None, paths, settings)
    assert loader.call_count == asr.call_count == 1
    settings.tts.speaker = "Ryan"
    pipeline.process_video(video, asr_path, None, paths, settings)
    settings.tts.instruct = "Changed instruction"
    pipeline.process_video(video, asr_path, None, paths, settings)
    assert loader.call_count == 3 and asr.call_count == 1
    assert model.generate_custom_voice.call_args.kwargs["speaker"] == "Ryan"
    assert model.generate_custom_voice.call_args.kwargs["instruct"] == "Changed instruction"
    settings.transcription.language = "en"
    observer = RecordingObserver()
    pipeline.process_video(video, asr_path, None, paths, settings, observer)
    assert asr.call_count == 2
    durations = next(e.details for e in observer.events if e.context.operation == "duration_qa")
    assert durations["source_video_duration"] == durations["tts_audio_duration"] == 2
    assert durations["duration_delta"] == 0

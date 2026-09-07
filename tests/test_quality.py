from unittest.mock import Mock

import numpy as np
import pytest

from transcript_video.events import RecordingObserver, event_scope
from transcript_video.processing.tts import core, qa


def test_final_word_and_added_missing_words():
    result = qa.compare_text("Then we run these testcases.", "Then we run these test")
    assert result["text_review_reasons"] == ["missing_final_word", "missing_words", "added_words"]
    assert result["missing_words"] == ["testcases"] and result["added_words"] == ["test"]
    assert result["word_coverage"] == 0.8 and result["word_error_ratio"] == 0.2
    assert result["first_word_match"] and not result["last_word_match"]
    assert qa.compare_text("ONE, two!", "one two")["word_error_ratio"] == 0
    assert qa.compare_text("one one two", "one two")["missing_words"] == ["one"]
    assert (
        "missing_final_word"
        not in qa.compare_text("one two", "one two extra")["text_review_reasons"]
    )


def test_final_verifier_uses_final_range_without_expected_prompt(monkeypatch):
    transcribe = Mock(return_value=[core.WordTiming("Then", 0, 0.1)])
    monkeypatch.setattr(core, "transcribe_word_timings", transcribe)
    entry = dict(text="Then run", cache_start_sample=10, cache_end_sample=30, review_reason="")
    audio = np.arange(100)
    qa.verify_sentences(audio, 1000, [entry], object(), "English")
    np.testing.assert_array_equal(transcribe.call_args.args[1], audio[10:30])
    assert transcribe.call_args.args[3] == ""
    assert "missing_final_word" in entry["review_reason"]
    assert entry["final_verification_status"] == "completed"
    qa.verify_sentences(audio, 1000, [entry], None, "English")
    assert entry["final_verification_status"] == "unavailable"


def test_acoustic_tail_is_only_a_review_hint():
    assert qa.check_tail(np.full(1000, 0.1), 1000)["possible_truncated_tail"]
    assert not qa.check_tail(np.r_[np.full(980, 0.1), np.zeros(20)], 1000)[
        "possible_truncated_tail"
    ]
    assert not qa.check_tail(np.full(1000, 0.001), 1000)["possible_truncated_tail"]


def test_every_speedup_has_semantic_review_and_summary(tmp_path):
    entries = [
        dict(subtitle_index=i, action="context_aligned", applied_speedup=speed, review_reason="")
        for i, speed in enumerate([1.0, 1.01, 1.15])
    ]
    observer = RecordingObserver()
    with event_scope(observer):
        core.log_tts_summary(3, entries, tmp_path / "review.jsonl")
    reviews = [e for e in observer.events if e.context.operation == "review"]
    assert [e.context.subtitle for e in reviews] == [1, 2]
    assert all(e.message == "speed_adjusted" for e in reviews)
    summary = next(e.details for e in observer.events if e.context.operation == "quality")
    assert summary["speed_adjusted"] == summary["flagged"] == 2
    assert summary["overflow"] == 0


def test_duration_delta_and_unknown_measurements():
    assert qa.duration_qa(10, 12, 12.1) == dict(
        source_video_duration=10,
        tts_audio_duration=12,
        final_output_duration=12.1,
        tts_duration_delta=2,
        duration_delta=pytest.approx(2.1),
    )
    assert qa.duration_qa(None, 12, None)["duration_delta"] is None


@pytest.mark.parametrize("generation", ["timed", "chunked"])
def test_verification_reads_published_pcm_sentence(tmp_path, monkeypatch, generation):
    import json

    import soundfile as sf

    from transcript_video.config import SubtitleSegment
    from transcript_video.processing.tts import chunks

    model = Mock()
    model.generate_custom_voice.return_value = ([np.full(1000, 0.123456)], 1000)
    inspected = []

    def transcribe(aligner, wav, sr, prompt, language):
        if not prompt:
            inspected.append(wav.copy())
            return [core.WordTiming("one", 0.1, 0.3)]
        return [core.WordTiming("one", 0.1, 0.3), core.WordTiming("two", 0.5, 0.7)]

    monkeypatch.setattr(core, "transcribe_word_timings", transcribe)
    for module in (core, chunks):
        monkeypatch.setattr(module, "load_qwen_tts_model", lambda *a: model)
        monkeypatch.setattr(module, "load_faster_whisper_aligner", lambda *a: object())
        monkeypatch.setattr(module, "get_media_duration_seconds", lambda *a: 2)
    output, report = tmp_path / "speech.wav", tmp_path / "review.jsonl"
    kwargs = dict(
        segments=[SubtitleSegment(0.2, 1.2, "one two")],
        audio_out=output,
        video_path=tmp_path / "video.mp4",
        tts_model_name="local",
        tts_language="English",
        tts_speaker="Aiden",
        tts_instruct="steady",
        device="cpu",
        attn_implementation="sdpa",
        alignment_model_name="aligner",
        verify_final_audio=True,
        review_log_path=report,
    )
    if generation == "timed":
        core.synthesize_timed_tts_audio(**kwargs)
    else:
        chunks.synthesize_tts_audio_by_time_chunks(**kwargs, chunks_dir=tmp_path / "chunks")
    row = json.loads(report.read_text())
    persisted, _ = sf.read(output, dtype="float32")
    np.testing.assert_array_equal(
        inspected[0], persisted[row["cache_start_sample"] : row["cache_end_sample"]]
    )
    assert "missing_final_word" in row["review_reason"]
    assert len(inspected) == 1

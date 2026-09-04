from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np

from transcript_video.config import SubtitleSegment
from transcript_video.processing.tts import chunks, core
from transcript_video.processing.tts.chunks import _dependent_owner_chunks
from transcript_video.processing.tts.core import (
    TTSContextGroup,
    WordTiming,
    align_context_group,
    build_tts_context_groups,
    fit_wav_to_available_duration,
    generate_context_group_items,
    overlay_tts_items,
    tts_review_counts,
    write_tts_review_log,
)


def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment(10.0, 11.4, "Open the configuration window."),
        SubtitleSegment(12.4, 14.0, "Then select the network interface."),
        SubtitleSegment(14.5, 16.2, "Now update the address."),
    ]


def test_context_group_keeps_one_second_pause_and_complete_sentences() -> None:
    groups = build_tts_context_groups(_segments())
    assert len(groups) == 1
    assert groups[0].text == (
        "Open the configuration window. Then select the network interface. Now update the address."
    )
    assert [segment.start for _, segment in groups[0].segments] == [10.0, 12.4, 14.5]


def test_context_group_limits_and_large_gap_split() -> None:
    segments = [
        SubtitleSegment(0.0, 1.0, "One."),
        SubtitleSegment(2.0, 3.0, "Two."),
        SubtitleSegment(7.0, 8.0, "Three."),
    ]
    assert [len(group.segments) for group in build_tts_context_groups(segments)] == [2, 1]
    assert [len(group.segments) for group in build_tts_context_groups(segments, 1)] == [1, 1, 1]
    assert [len(group.segments) for group in build_tts_context_groups(segments[:2], 4, 5)] == [
        1,
        1,
    ]


def test_alignment_normalizes_case_and_punctuation() -> None:
    group = build_tts_context_groups(_segments()[:2])[0]
    words = [
        WordTiming(text, index * 0.2, index * 0.2 + 0.15)
        for index, text in enumerate(
            [
                "OPEN",
                "the",
                "configuration",
                "window",
                "then",
                "select",
                "the",
                "network",
                "interface",
            ]
        )
    ]
    aligned, failures = align_context_group(group, words, 2.0)
    assert failures == {}
    assert [item.subtitle_index for item in aligned] == [1, 2]
    assert aligned[0].source_end <= aligned[1].source_start


def test_low_confidence_alignment_is_rejected() -> None:
    group = TTSContextGroup(0, [(1, SubtitleSegment(0, 1, "one two three four"))])
    aligned, failures = align_context_group(group, [WordTiming("one", 0.1, 0.3)], 1.0)
    assert aligned == []
    assert failures == {1: "alignment_low_confidence"}


def test_group_generation_places_extracted_sentences_on_original_timeline(
    monkeypatch,
) -> None:
    segments = _segments()
    groups = build_tts_context_groups(segments)
    model = mock.Mock()
    model.generate_custom_voice.return_value = ([np.ones(5000, dtype=np.float32)], 1000)
    timings = [
        WordTiming(text, index * 0.3, index * 0.3 + 0.2)
        for index, text in enumerate(
            [
                "Open",
                "the",
                "configuration",
                "window",
                "Then",
                "select",
                "the",
                "network",
                "interface",
                "Now",
                "update",
                "the",
                "address",
            ]
        )
    ]
    monkeypatch.setattr(core, "transcribe_word_timings", lambda *args: timings)

    items, sample_rate, reviews = generate_context_group_items(
        model=model,
        aligner=object(),
        groups=groups,
        all_segments=segments,
        language="English",
        speaker="Aiden",
        instruct="steady",
        max_speedup=1.15,
        video_duration=20.0,
    )
    audio = overlay_tts_items(items, sample_rate, 20.0)

    assert model.generate_custom_voice.call_count == 1
    assert reviews == []
    assert audio[10_000] != 0 and audio[12_400] != 0 and audio[14_500] != 0
    assert np.all(audio[11_400:12_400] == 0)


def test_duration_fit_is_bounded_pitch_preserving_and_never_truncates(monkeypatch) -> None:
    wav = np.ones(2000, dtype=np.float32)
    called = {}

    def stretch(audio, sample_rate, speed):
        called["speed"] = speed
        return audio[: round(len(audio) / speed)]

    monkeypatch.setattr(core, "_pitch_preserving_speedup", stretch)
    fitted = fit_wav_to_available_duration(wav, 1000, 1.0, max_speedup=1.15)
    assert called == {"speed": 1.15}
    assert len(fitted) > 1000


def test_alignment_failure_and_overflow_create_complete_review_entries(monkeypatch) -> None:
    segments = [
        SubtitleSegment(0.0, 0.5, "A long first sentence."),
        SubtitleSegment(1.0, 1.5, "Next."),
    ]
    group = build_tts_context_groups(segments)[0]
    model = mock.Mock()
    model.generate_custom_voice.return_value = ([np.ones(4000, dtype=np.float32)], 1000)
    monkeypatch.setattr(core, "transcribe_word_timings", lambda *args: [])
    monkeypatch.setattr(
        core,
        "_pitch_preserving_speedup",
        lambda wav, sample_rate, speed: wav[: round(len(wav) / speed)],
    )

    _, _, reviews = generate_context_group_items(
        model=model,
        aligner=object(),
        groups=[group],
        all_segments=segments,
        language="English",
        speaker="Aiden",
        instruct="steady",
        max_speedup=1.15,
        video_duration=2.0,
    )
    reasons = {entry["review_reason"] for entry in reviews}
    assert {"alignment_failed", "exceeds_max_speedup"} <= reasons
    required = {
        "subtitle_index",
        "text",
        "start",
        "end",
        "next_start",
        "available_duration",
        "generated_speech_duration",
        "required_speedup",
        "max_speedup",
        "context_group_index",
        "alignment_source_start",
        "alignment_source_end",
        "action",
        "review_reason",
    }
    assert required <= reviews[0].keys()


def test_review_log_contains_only_given_problems(tmp_path: Path) -> None:
    path = tmp_path / "lesson_tts_review.jsonl"
    write_tts_review_log(path, [])
    assert path.read_text(encoding="utf-8") == ""
    entry = {"subtitle_index": 2, "review_reason": "timing_overflow"}
    write_tts_review_log(path, [entry])
    assert json.loads(path.read_text(encoding="utf-8")) == entry
    assert tts_review_counts(
        2,
        [
            {
                "subtitle_index": 1,
                "review_reason": "exceeds_max_speedup",
                "action": "kept_overflow_after_pitch_preserving_speedup",
            }
        ],
    ) == (2, 1)


def test_cache_boundary_group_owner_is_regenerated_with_target_chunk() -> None:
    groups = build_tts_context_groups(
        [
            SubtitleSegment(299.0, 299.8, "Before boundary."),
            SubtitleSegment(300.5, 301.5, "After boundary."),
        ]
    )
    assert len(groups) == 1
    assert _dependent_owner_chunks(groups, 300, 1) == {0, 1}


def test_cache_boundary_group_is_written_once_at_original_starts(
    tmp_path: Path, monkeypatch
) -> None:
    import soundfile as sf

    segments = [
        SubtitleSegment(299.0, 299.8, "Before boundary."),
        SubtitleSegment(300.5, 301.5, "After boundary."),
    ]
    groups = build_tts_context_groups(segments)

    def generated(**kwargs):
        items = [
            (index, segment, np.ones(5, dtype=np.float32))
            for group in kwargs["groups"]
            for index, segment in group.segments
        ]
        return items, 10 if items else None, []

    monkeypatch.setattr(chunks, "generate_context_group_items", generated)
    first = tmp_path / "chunk_000.wav"
    second = tmp_path / "chunk_001.wav"
    common = {
        "model": object(),
        "tts_language": "English",
        "tts_speaker": "Aiden",
        "tts_instruct": "steady",
        "sample_rate": 10,
        "all_segments": segments,
        "context_groups": groups,
    }
    chunks.synthesize_one_fixed_time_chunk(
        chunk_index=0,
        chunk_start=0,
        chunk_end=300,
        chunk_segments=[segments[0]],
        chunk_audio_out=first,
        **common,
    )
    chunks.synthesize_one_fixed_time_chunk(
        chunk_index=1,
        chunk_start=300,
        chunk_end=302.5,
        chunk_segments=[segments[1]],
        chunk_audio_out=second,
        **common,
    )
    first_audio, _ = sf.read(first, dtype="float32")
    second_audio, _ = sf.read(second, dtype="float32")
    assert first_audio[2990] > 0.9 and first_audio[3005] > 0.9
    assert np.all(second_audio == 0)

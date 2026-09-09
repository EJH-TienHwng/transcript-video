from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from transcript_video.config import ProjectPaths, SubtitleSegment, TTSSettings
from transcript_video.processing.tts import chunks, core
from transcript_video.processing.tts.chunks import _dependent_owner_chunks
from transcript_video.processing.tts.core import (
    TTSContextGroup,
    WordTiming,
    align_context_group,
    build_retimed_subtitle_segments,
    build_tts_context_groups,
    find_safe_inter_sentence_boundary,
    fit_wav_to_available_duration,
    generate_context_group_items,
    overlay_tts_items,
    subtitle_retiming_summary,
    tts_review_counts,
    write_tts_review_log,
)


@pytest.fixture(autouse=True)
def isolate_tts_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(chunks, "find_project_root", lambda: tmp_path)


def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment(10.0, 11.4, "Open the configuration window."),
        SubtitleSegment(12.4, 14.0, "Then select the network interface."),
        SubtitleSegment(14.5, 16.2, "Now update the address."),
    ]


def _qwen_model(pad=None, eos=2150):
    return SimpleNamespace(
        model=SimpleNamespace(
            talker=SimpleNamespace(
                generation_config=SimpleNamespace(pad_token_id=pad, eos_token_id=999)
            ),
            config=SimpleNamespace(
                talker_config=SimpleNamespace(codec_eos_token_id=eos, codec_pad_id=777),
                tts_pad_token_id=888,
            ),
            generation_config=SimpleNamespace(pad_token_id=666),
        ),
        generate_custom_voice=mock.Mock(return_value=([np.ones((2, 3))], np.int64(24000))),
    )


@pytest.mark.parametrize(
    "pad,eos,expected",
    [
        (123, 456, 123),
        (0, 456, 0),
        (None, 2150, 2150),
        (None, [2150, 2151], 2150),
        (None, (2150, 2151), 2150),
        (None, np.int64(2150), 2150),
        (None, None, None),
        (None, [], None),
        (None, (), None),
        (None, True, None),
        (None, -1, None),
        (None, "2150", None),
        (None, object(), None),
        (None, [2150, False], None),
        (None, [2150, -1], None),
        (None, [[2150]], None),
        (False, 2150, None),
        (-1, 2150, None),
        ([123], 2150, None),
        (object(), 2150, None),
    ],
)
def test_qwen_pad_resolution_and_explicit_kwargs(pad, eos, expected):
    model = _qwen_model(pad, eos)
    assert core.resolve_generation_pad_token_id(model) == expected
    wav, sr = core.generate_qwen_custom_voice(model, "Hello", "English", "Aiden", "steady")
    kwargs = dict(text="Hello", language="English", speaker="Aiden", instruct="steady")
    if expected is not None:
        kwargs["pad_token_id"] = expected
        assert type(core.resolve_generation_pad_token_id(model)) is int
    model.generate_custom_voice.assert_called_once_with(**kwargs)
    assert model.model.talker.generation_config.pad_token_id is pad
    np.testing.assert_array_equal(wav, np.ones(6, dtype=np.float32))
    assert wav.dtype == np.float32 and type(sr) is int and sr == 24000


@pytest.mark.parametrize("model", [object(), SimpleNamespace(model=object())])
def test_qwen_pad_missing_generation_config(model):
    assert core.resolve_generation_pad_token_id(model) is None


@pytest.mark.parametrize(
    "pad,eos,expected", [(None, 2150, 2150), (123, 456, 123), (None, None, None)]
)
def test_qwen_loader_initializes_only_missing_talker_pad(monkeypatch, pad, eos, expected):
    import sys

    model = _qwen_model(pad, eos)
    factory = mock.Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=factory)),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16", float32="float32"))
    monkeypatch.setattr(core, "resolve_torch_device", lambda *args: "cpu")
    assert core.load_qwen_tts_model("local-model", "cpu", "sdpa") is model
    factory.assert_called_once_with(
        "local-model", device_map="cpu", dtype="float32", attn_implementation="sdpa"
    )
    assert vars(model.model.talker.generation_config) == dict(
        pad_token_id=expected, eos_token_id=999
    )
    assert model.model.config.talker_config.codec_eos_token_id == eos
    assert model.model.generation_config.pad_token_id == 666


@pytest.mark.parametrize(
    "result,match",
    [
        ((None, 24000), "no waveform"),
        (([], 24000), "no waveform"),
        (([np.ones(2)], True), "sample rate"),
        (([np.ones(2)], -1), "sample rate"),
        (([np.ones(2)], 0), "sample rate"),
        (([np.ones(2)], 24000.0), "sample rate"),
        (([np.array([])], 24000), "empty, silent, or non-finite"),
        (([np.zeros(2)], 24000), "empty, silent, or non-finite"),
        (([np.array([np.nan])], 24000), "empty, silent, or non-finite"),
        (([np.array([np.inf])], 24000), "empty, silent, or non-finite"),
    ],
)
def test_qwen_explicit_pad_preserves_audio_validation(result, match):
    model = _qwen_model()
    model.generate_custom_voice.return_value = result
    with pytest.raises(ValueError, match=match):
        core.generate_qwen_custom_voice(model, "Hello", "English", "Aiden", "steady")
    assert model.generate_custom_voice.call_args.kwargs["pad_token_id"] == 2150


@pytest.mark.integration
def test_installed_qwen_talker_pad_prevents_transformers_fallback(monkeypatch, caplog):
    """Exercise installed Qwen prompt/kwargs flow and HF token preparation without weights."""
    from copy import deepcopy
    from functools import partial

    torch = pytest.importorskip("torch")
    qwen_tts = pytest.importorskip("qwen_tts")
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    from transformers.generation.utils import logger as generation_logger

    # Transformers does not propagate to pytest's root capture handler by default.
    monkeypatch.setattr(
        generation_logger, "handlers", [*generation_logger.handlers, caplog.handler]
    )

    model = _qwen_model()
    qwen = model.model
    talker = qwen.talker
    talker.generation_config = GenerationConfig(eos_token_id=999)
    talker.device = torch.device("cpu")
    talker.config = SimpleNamespace(is_encoder_decoder=False)
    talker.text_projection = torch.nn.Identity()
    embedding = torch.nn.Embedding(16, 2)
    talker.get_text_embeddings = lambda: embedding
    talker.get_input_embeddings = lambda: embedding
    for field in (
        "codec_nothink_id",
        "codec_think_bos_id",
        "codec_think_eos_id",
        "codec_pad_id",
        "codec_bos_id",
    ):
        setattr(qwen.config.talker_config, field, 1)
    qwen.config.talker_config.vocab_size = 4096
    qwen.config.tts_bos_token_id = 2
    qwen.config.tts_eos_token_id = 3
    qwen.config.tts_pad_token_id = 4
    qwen.tts_model_type = "custom_voice"
    qwen.tts_model_size = "1b7"
    qwen.generate = mock.Mock(wraps=partial(Qwen3TTSForConditionalGeneration.generate, qwen))

    wrapper = object.__new__(qwen_tts.Qwen3TTSModel)
    wrapper.model = qwen
    wrapper.generate_defaults = {}
    wrapper._validate_languages = mock.Mock()
    wrapper._validate_speakers = mock.Mock()
    wrapper._tokenize_texts = lambda texts: [torch.zeros((1, 12), dtype=torch.long)]

    class ReachedTalker(Exception):
        pass

    prepared_pads = []

    def prepare_tokens(**kwargs):
        assert "pad_token_id" not in kwargs  # Qwen 0.1.1 drops it at this boundary.
        assert kwargs["eos_token_id"] == 2150  # Overrides talker's generation_config EOS.
        config = deepcopy(talker.generation_config)
        config.update(**kwargs)
        GenerationMixin._prepare_special_tokens(talker, config, kwargs_has_attention_mask=True)
        prepared_pads.append(config._pad_token_tensor.item())
        raise ReachedTalker

    talker.generate = prepare_tokens
    warning = "Setting `pad_token_id` to `eos_token_id`:2150 for open-end generation."
    with pytest.raises(ReachedTalker):
        core.generate_qwen_custom_voice(wrapper, "Hello", "Auto", "", "")
    assert qwen.generate.call_args.kwargs["pad_token_id"] == 2150
    assert warning in caplog.text

    monkeypatch.setattr(qwen_tts.Qwen3TTSModel, "from_pretrained", lambda *args, **kwargs: wrapper)
    monkeypatch.setattr(core, "resolve_torch_device", lambda *args: "cpu")
    loaded = core.load_qwen_tts_model("fixture-without-weights", "cpu", "auto")
    caplog.clear()
    for _ in range(3):
        with pytest.raises(ReachedTalker):
            core.generate_qwen_custom_voice(loaded, "Hello", "Auto", "", "")
    assert warning not in caplog.text
    assert prepared_pads == [2150] * 4
    assert talker.generation_config.pad_token_id == 2150


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
        WordTiming(text, index * 0.4, index * 0.4 + 0.15)
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
    aligned, failures = align_context_group(group, words, 4.0)
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
    model = _qwen_model()
    waveform = np.zeros(5000, dtype=np.float32)
    timings = [
        WordTiming(
            text,
            index * 0.3 + 0.2 * (index >= 4) + 0.2 * (index >= 9),
            index * 0.3 + 0.2 * (index >= 4) + 0.2 * (index >= 9) + 0.2,
        )
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
    for timing in timings:
        waveform[round(timing.start * 1000) : round(timing.end * 1000)] = 0.2
    model.generate_custom_voice.return_value = ([waveform], 1000)
    monkeypatch.setattr(core, "transcribe_word_timings", lambda *args: timings)

    items, sample_rate, reviews = generate_context_group_items(
        model=model,
        aligner=object(),
        groups=groups,
        all_segments=segments,
        language="English",
        speaker="Aiden",
        instruct="steady",
        max_speedup=1.25,
        video_duration=20.0,
    )
    audio = overlay_tts_items(items, sample_rate, 20.0)

    assert model.generate_custom_voice.call_count == 1
    assert model.generate_custom_voice.call_args.kwargs["pad_token_id"] == 2150
    assert all(entry["action"] == "context_aligned" for entry in reviews)
    assert audio[10_000] != 0 and audio[12_400] != 0 and audio[14_500] != 0
    assert np.all(audio[11_400:12_400] == 0)


def test_duration_fit_is_bounded_pitch_preserving_and_never_truncates(monkeypatch) -> None:
    wav = np.ones(2000, dtype=np.float32)
    called = {}

    def stretch(audio, sample_rate, speed):
        called["speed"] = speed
        return audio[: round(len(audio) / speed)]

    monkeypatch.setattr(core, "_pitch_preserving_speedup", stretch)
    assert TTSSettings().max_speedup == 1.25
    fitted = fit_wav_to_available_duration(wav, 1000, 1.0)
    assert called == {"speed": 1.25}
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
        max_speedup=1.25,
        video_duration=2.0,
    )
    reasons = {reason for entry in reviews for reason in entry["review_reason"].split(";")}
    assert {"alignment_failed", "exceeds_max_speedup"} <= reasons
    assert reviews[0]["required_speedup"] > 1.25
    assert reviews[0]["applied_speedup"] == 1.25
    assert reviews[0]["overflow_duration"] > 0
    required = {
        "subtitle_index",
        "text",
        "start",
        "end",
        "next_start",
        "available_duration",
        "generated_speech_duration",
        "raw_generated_duration",
        "applied_speedup",
        "final_audio_duration",
        "overflow_duration",
        "required_speedup",
        "max_speedup",
        "context_group_index",
        "alignment_source_start",
        "alignment_source_end",
        "asr_last_word_end",
        "detected_acoustic_end",
        "tail_extension_seconds",
        "next_asr_word_start",
        "detected_next_onset",
        "boundary_safe",
        "boundary_reason",
        "original_start",
        "original_end",
        "actual_start",
        "actual_end",
        "timing_shift",
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


def test_chunk_boundary_group_owner_is_regenerated_with_target_chunk() -> None:
    groups = build_tts_context_groups(
        [
            SubtitleSegment(299.0, 299.8, "Before boundary."),
            SubtitleSegment(300.5, 301.5, "After boundary."),
        ]
    )
    assert len(groups) == 1
    assert _dependent_owner_chunks(groups, 300, 1) == {0, 1}


def test_chunk_boundary_group_is_written_once_at_original_starts(
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
        return (
            items,
            10 if items else None,
            [_entry(index, segment, len(wav) / 10) for index, segment, wav in items],
        )

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


def _entry(index: int, segment: SubtitleSegment, duration: float) -> dict[str, object]:
    return core._review_entry(
        subtitle_index=index,
        segment=segment,
        next_start=None,
        available_duration=max(0.001, segment.end - segment.start),
        generated_duration=duration,
        max_speedup=1.25,
        group_index=0,
        source_start=None,
        source_end=None,
        action="context_aligned",
        reason="",
    )


def _placed_entry(
    index: int, segment: SubtitleSegment, actual_start: float, actual_end: float
) -> dict[str, object]:
    entry = _entry(index, segment, actual_end - actual_start)
    entry.update(actual_start=actual_start, actual_end=actual_end)
    return entry


@pytest.mark.parametrize(
    ("segment", "actual", "expected", "changed", "reasons"),
    [
        (
            SubtitleSegment(10, 13, "shorter"),
            (10.000041, 11.8),
            SubtitleSegment(10, 13, "shorter"),
            False,
            {"translated_window_preserved", "unchanged"},
        ),
        (
            SubtitleSegment(10, 13, "longer"),
            (10, 13.4),
            SubtitleSegment(10, 13.4, "longer"),
            True,
            {"tts_extended"},
        ),
        (
            SubtitleSegment(12.2, 14, "shifted"),
            (12.52, 14.3),
            SubtitleSegment(12.52, 14.3, "shifted"),
            True,
            {"tts_shifted", "tts_extended"},
        ),
    ],
)
def test_retiming_preserves_translated_windows_and_uses_millisecond_precision(
    segment, actual, expected, changed, reasons
) -> None:
    reviews = [_placed_entry(1, segment, *actual)]
    assert build_retimed_subtitle_segments(reviews) == [expected]
    assert reviews[0]["subtitle_timing_changed"] is changed
    assert set(reviews[0]["subtitle_timing_change_reason"]) == reasons


def test_retiming_does_not_unnecessarily_shrink_separated_cues() -> None:
    reviews = [
        _placed_entry(1, SubtitleSegment(10, 13, "A"), 10, 12.1),
        _placed_entry(2, SubtitleSegment(13.2, 16, "B"), 13.2, 15.1),
    ]
    assert build_retimed_subtitle_segments(reviews) == [
        SubtitleSegment(10, 13, "A"),
        SubtitleSegment(13.2, 16, "B"),
    ]
    assert all(not entry["subtitle_timing_changed"] for entry in reviews)


def test_retiming_resolves_visual_conflict_without_hiding_narration() -> None:
    reviews = [
        _placed_entry(1, SubtitleSegment(10, 13, "A"), 10, 12.5),
        _placed_entry(2, SubtitleSegment(12.2, 14, "B"), 12.62, 14.3),
    ]
    assert build_retimed_subtitle_segments(reviews) == [
        SubtitleSegment(10, 12.62, "A"),
        SubtitleSegment(12.62, 14.3, "B"),
    ]
    assert reviews[0]["retimed_end"] >= reviews[0]["actual_tts_end"]
    assert "next_cue_conflict" in reviews[0]["subtitle_timing_change_reason"]
    assert reviews[0]["subtitle_end_shift"] == pytest.approx(-0.38)


def test_retiming_mixed_sequence_updates_final_review_and_summary() -> None:
    reviews = [
        _placed_entry(1, SubtitleSegment(0, 2, "unchanged"), 0, 1),
        _placed_entry(2, SubtitleSegment(3, 5, "shifted"), 3.2, 4.5),
        _placed_entry(3, SubtitleSegment(6, 8, "extended"), 6, 8.4),
        _placed_entry(4, SubtitleSegment(9, 12, "conflict"), 9, 10.5),
        _placed_entry(5, SubtitleSegment(11, 13, "unchanged"), 11.4, 12.5),
    ]
    segments = build_retimed_subtitle_segments(reviews)
    assert segments == [
        SubtitleSegment(0, 2, "unchanged"),
        SubtitleSegment(3.2, 5, "shifted"),
        SubtitleSegment(6, 8.4, "extended"),
        SubtitleSegment(9, 11.4, "conflict"),
        SubtitleSegment(11.4, 13, "unchanged"),
    ]
    assert [(entry["retimed_start"], entry["retimed_end"]) for entry in reviews] == [
        (segment.start, segment.end) for segment in segments
    ]
    assert subtitle_retiming_summary(reviews) == {
        "total_cues": 5,
        "unchanged": 1,
        "start_shifted": 2,
        "end_extended": 1,
        "conflict_adjusted": 1,
        "max_start_shift": pytest.approx(0.4),
        "max_end_extension": pytest.approx(0.4),
    }


def _generate(model, segments, aligner=None, max_speedup=1.0):
    return generate_context_group_items(
        model=model,
        aligner=aligner,
        groups=build_tts_context_groups(segments),
        all_segments=segments,
        language="English",
        speaker="Aiden",
        instruct="steady",
        max_speedup=max_speedup,
        video_duration=20.0,
    )


def test_shorter_audio_never_slows_down_and_logs_actual_speed(monkeypatch) -> None:
    speedup = mock.Mock(side_effect=AssertionError("must not stretch shorter audio"))
    monkeypatch.setattr(core, "_pitch_preserving_speedup", speedup)
    wav = np.full(2000, 0.1, dtype=np.float32)
    np.testing.assert_array_equal(fit_wav_to_available_duration(wav, 1000, 3.0), wav)
    segment = SubtitleSegment(0, 2, "Short sentence.")
    model = mock.Mock()
    model.generate_custom_voice.return_value = ([wav], 1000)
    monkeypatch.setattr(
        core,
        "transcribe_word_timings",
        lambda *args: [WordTiming("Short", 0, 0.5), WordTiming("sentence", 0.6, 1.0)],
    )
    _, _, reviews = generate_context_group_items(
        model=model,
        aligner=object(),
        groups=build_tts_context_groups([segment]),
        all_segments=[segment],
        language="English",
        speaker="Aiden",
        instruct="steady",
        max_speedup=1.25,
        video_duration=3.0,
    )
    assert reviews[0]["required_speedup"] == pytest.approx(2 / 3)
    assert reviews[0]["applied_speedup"] == 1.0
    assert reviews[0]["final_audio_duration"] == 2.0
    speedup.assert_not_called()


def test_bounded_speedup_preserves_tail_marker(monkeypatch) -> None:
    wav = np.full(4000, 0.1, dtype=np.float32)
    wav[-100:] = 0.9

    def stretch(audio, sample_rate, speed):
        assert speed == 1.25
        # Resample the entire synthetic marker, rather than mock a destructive crop.
        return np.interp(
            np.linspace(0, len(audio) - 1, round(len(audio) / speed)), np.arange(len(audio)), audio
        )

    monkeypatch.setattr(core, "_pitch_preserving_speedup", stretch)
    fitted = fit_wav_to_available_duration(wav, 1000, 3.0, 1.25)
    assert len(fitted) == round(4000 / 1.25)
    assert fitted[-1] == pytest.approx(0.9)
    assert np.count_nonzero(fitted == wav[-1]) > 50


@pytest.mark.parametrize("failure", ["exception", "low_confidence", "invalid_range"])
def test_unsafe_alignment_regenerates_only_failed_sentence(monkeypatch, failure) -> None:
    segment = SubtitleSegment(0, 1, "one two three four")
    context = np.full(2000, 0.1, dtype=np.float32)
    individual = np.full(4000, 0.3, dtype=np.float32)
    individual[-20:] = 0.9
    model = mock.Mock()
    model.generate_custom_voice.side_effect = [([context], 1000), ([individual], 1000)]
    valid = [
        WordTiming(word, i * 0.3, i * 0.3 + 0.2) for i, word in enumerate(segment.text.split())
    ]
    bad = RuntimeError("ASR failed") if failure == "exception" else [WordTiming("one", 0, 0.2)]
    if failure == "invalid_range":
        bad = [WordTiming(word, 0.1, 0.3) for word in segment.text.split()]
    monkeypatch.setattr(core, "transcribe_word_timings", mock.Mock(side_effect=[bad, valid]))
    # This sentinel must never be invoked, even if someone reintroduces the old helper.
    monkeypatch.setattr(
        core,
        "proportional_alignment_fallback",
        mock.Mock(side_effect=AssertionError("unsafe slicing")),
        raising=False,
    )
    items, sr, reviews = _generate(model, [segment], aligner=object())
    assert model.generate_custom_voice.call_count == 2
    for call in model.generate_custom_voice.call_args_list:
        assert call.kwargs == dict(
            text=segment.text, language="English", speaker="Aiden", instruct="steady"
        )
    np.testing.assert_array_equal(items[0][2], individual)
    audio = overlay_tts_items(items, sr, 1, reviews=reviews)
    assert audio[-1] == pytest.approx(0.9)
    assert reviews[0]["action"] == "regenerated_individual_sentence"
    assert reviews[0]["individual_attempts"] == 1


def test_individual_retry_keeps_best_coverage_and_reports_failure(monkeypatch) -> None:
    segment = SubtitleSegment(0, 1, "one two three four")
    first, second = np.full(1000, 0.2), np.full(2000, 0.4)
    model = mock.Mock()
    model.generate_custom_voice.side_effect = [([first], 1000), ([second], 1000)]
    monkeypatch.setattr(
        core,
        "transcribe_word_timings",
        mock.Mock(
            side_effect=[
                [WordTiming("one", 0, 0.2), WordTiming("two", 0.3, 0.5)],
                [WordTiming("one", 0, 0.2)],
            ]
        ),
    )
    wav, sr, metadata = core.generate_individual_sentence_fallback(
        model, object(), 1, segment, "English", "Aiden", "steady", 1000
    )
    assert model.generate_custom_voice.call_count == 2
    np.testing.assert_array_equal(wav, first.astype(np.float32))
    assert sr == 1000
    assert metadata["verification_confidence"] == 0.5
    assert metadata["verification_reason"] == "individual_coverage_low"


def test_individual_retry_recovers_missing_final_words(monkeypatch) -> None:
    segment = SubtitleSegment(0, 1, "one two")
    model = _qwen_model()
    model.generate_custom_voice.side_effect = [
        ([np.ones(1000) * 0.1], 1000),
        ([np.ones(2000) * 0.3], 1000),
    ]
    monkeypatch.setattr(
        core,
        "transcribe_word_timings",
        mock.Mock(
            side_effect=[
                [WordTiming("one", 0, 0.2)],
                [WordTiming("one", 0, 0.2), WordTiming("two", 0.3, 0.5)],
            ]
        ),
    )
    wav, _, metadata = core.generate_individual_sentence_fallback(
        model, object(), 1, segment, "English", "Aiden", "steady", 1000
    )
    assert len(wav) == 2000
    assert metadata["individual_attempts"] == 2
    assert metadata["verification_reason"] == ""
    assert len(model.generate_custom_voice.call_args_list) == 2
    assert all(
        call.kwargs["pad_token_id"] == 2150 for call in model.generate_custom_voice.call_args_list
    )


@pytest.mark.parametrize(
    "result",
    [
        ([], 1000),
        ([np.ones(10)], 0),
        ([np.ones(10)], 2000),
        ([np.array([np.nan])], 1000),
        ([np.zeros(10)], 1000),
    ],
)
def test_invalid_individual_audio_is_explicit_failure(result, caplog) -> None:
    caplog.set_level("INFO", logger="transcript_video.processing.tts.core")
    model = mock.Mock()
    model.generate_custom_voice.return_value = result
    wav, _, metadata = core.generate_individual_sentence_fallback(
        model, None, 42, SubtitleSegment(0, 1, "Text"), "English", "Aiden", "steady", 1000
    )
    assert wav is None
    assert metadata["generation_failures"] == 2
    assert "#42" in caplog.text and "failed" in caplog.text


def test_timed_generation_failure_writes_review_and_does_not_publish(tmp_path, monkeypatch) -> None:
    model = mock.Mock()
    model.generate_custom_voice.side_effect = RuntimeError("Qwen failed")
    monkeypatch.setattr(core, "load_qwen_tts_model", lambda *args: model)
    monkeypatch.setattr(core, "get_media_duration_seconds", lambda *args: 2.0)
    output = tmp_path / "test_tts.wav"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError):
        core.synthesize_timed_tts_audio(
            [SubtitleSegment(0, 1, "Text")],
            output,
            tmp_path / "video.mp4",
            "local-model",
            "English",
            "Aiden",
            "steady",
            "cuda",
            "sdpa",
        )
    assert output.read_bytes() == b"previous output"
    entry = json.loads((tmp_path / "data/report/tts/test_tts_review.jsonl").read_text())
    assert entry["action"] == "generation_failed"
    assert entry["generation_failures"] == 2


@pytest.mark.parametrize("mode", ["simple", "timed", "chunked"])
def test_qwen_pad_reaches_generation_in_all_output_modes(tmp_path, monkeypatch, mode):
    import soundfile as sf

    model = _qwen_model()
    model.generate_custom_voice.return_value = ([np.full(1000, 0.2)], 1000)
    for module in (core, chunks):
        monkeypatch.setattr(module, "load_qwen_tts_model", lambda *args: model)
        monkeypatch.setattr(module, "get_media_duration_seconds", lambda *args: 2.0)
    output = tmp_path / "speech.wav"
    kwargs = dict(
        segments=[SubtitleSegment(0, 1, "Hello")],
        audio_out=output,
        tts_model_name="local-model",
        tts_language="English",
        tts_speaker="Aiden",
        tts_instruct="steady",
        device="cpu",
        attn_implementation="auto",
    )
    if mode == "simple":
        core.synthesize_simple_tts_audio(**kwargs)
    elif mode == "timed":
        core.synthesize_timed_tts_audio(**kwargs, video_path=tmp_path / "video.mp4")
    else:
        chunks.synthesize_tts_audio_by_time_chunks(
            **kwargs, video_path=tmp_path / "video.mp4", chunks_dir=tmp_path / "chunks"
        )
        chunks.synthesize_tts_audio_by_time_chunks(
            **kwargs,
            video_path=tmp_path / "video.mp4",
            chunks_dir=tmp_path / "chunks",
            rerun_chunk=0,
        )
    # Without an aligner, timed modes generate context then fall back to individual audio.
    assert model.generate_custom_voice.call_count == {"simple": 1, "timed": 2, "chunked": 4}[mode]
    assert all(
        call.kwargs["pad_token_id"] == 2150 for call in model.generate_custom_voice.call_args_list
    )
    wav, sr = sf.read(output)
    assert sr == 1000
    np.testing.assert_allclose(wav[:1000], 0.2, atol=1 / 32768)


def test_alignment_padding_preserves_late_tail_without_next_sentence(monkeypatch) -> None:
    segments = [SubtitleSegment(0, 1, "first"), SubtitleSegment(3, 4, "second")]
    wav = np.zeros(2000, dtype=np.float32)
    wav[100:500] = 0.1
    wav[500:600] = 0.9  # ASR ends 100 ms before the acoustic tail.
    wav[1000:1500] = 0.4
    model = mock.Mock()
    model.generate_custom_voice.return_value = ([wav], 1000)
    monkeypatch.setattr(
        core,
        "transcribe_word_timings",
        lambda *args: [WordTiming("first", 0.1, 0.5), WordTiming("second", 1.0, 1.5)],
    )
    items, _, reviews = _generate(model, segments, aligner=object())
    assert reviews[0]["alignment_source_end"] == pytest.approx(0.63)
    assert reviews[0]["asr_last_word_end"] == 0.5
    assert reviews[0]["detected_acoustic_end"] == 0.6
    assert reviews[0]["boundary_safe"] is True
    assert np.count_nonzero(items[0][2] == np.float32(0.9)) == 100
    assert not np.any(items[0][2] == np.float32(0.4))
    assert len(items[1][2]) == 530  # Speech plus release guard; trailing silence is excluded.


def test_alignment_rejects_missing_middle_words_and_acoustic_check_rejects_no_gap() -> None:
    group = build_tts_context_groups([SubtitleSegment(0, 1, "one two three four")])[0]
    _, failures = align_context_group(
        group,
        [WordTiming("one", 0, 0.1), WordTiming("three", 0.2, 0.3), WordTiming("four", 0.4, 0.5)],
        1.0,
    )
    assert failures == {1: "alignment_low_confidence"}
    group = build_tts_context_groups([SubtitleSegment(0, 1, "one"), SubtitleSegment(1, 2, "two")])[
        0
    ]
    aligned, failures = align_context_group(
        group, [WordTiming("one", 0, 0.5), WordTiming("two", 0.55, 0.9)], 1.0
    )
    assert not failures and len(aligned) == 2
    boundary = find_safe_inter_sentence_boundary(np.ones(1000), 1000, 0.5, 0.55)
    assert boundary.safe is False
    assert boundary.reason == "no_safe_acoustic_gap"


def test_collision_placement_preserves_samples_logs_shift_and_recovers_gap(caplog) -> None:
    caplog.set_level("INFO", logger="transcript_video.processing.tts.core")
    segments = [
        SubtitleSegment(10, 11, "A"),
        SubtitleSegment(13, 14, "B"),
        SubtitleSegment(20, 21, "C"),
    ]
    waves = [
        np.full(4000, 0.3, dtype=np.float32),
        np.full(1000, 0.4, dtype=np.float32),
        np.full(1000, 0.5, dtype=np.float32),
    ]
    waves[0][-10:] = 0.9
    reviews = [
        _entry(i, segment, len(wav) / 1000)
        for i, (segment, wav) in enumerate(zip(segments, waves, strict=True), 1)
    ]
    audio = overlay_tts_items(
        [(i, seg, wav) for i, (seg, wav) in enumerate(zip(segments, waves, strict=True), 1)],
        1000,
        21,
        reviews=reviews,
    )
    np.testing.assert_array_equal(audio[10000:14000], waves[0])
    assert np.all(audio[14000:14120] == 0)
    np.testing.assert_array_equal(audio[14120:15120], waves[1])
    assert reviews[1]["original_start"] == 13
    assert reviews[1]["actual_start"] == 14.12
    assert reviews[1]["actual_end"] == 15.12
    assert build_retimed_subtitle_segments(reviews)[1] == SubtitleSegment(14.12, 15.12, "B")
    assert reviews[1]["timing_shift"] == 1.12
    assert "timing_shift_exceeds_threshold" in reviews[1]["review_reason"]
    assert reviews[2]["timing_shift"] == 0
    assert "prevent voice overlap" in caplog.text


def test_overlay_scales_peak_without_clipping() -> None:
    audio = overlay_tts_items([(1, SubtitleSegment(0, 1, "A"), np.array([0.5, 1.5, 2.0]))], 10, 0)
    np.testing.assert_allclose(audio, [0.25, 0.75, 1.0])


def test_chunk_boundary_rebuild_and_rerun_keep_each_sentence_once(tmp_path, monkeypatch) -> None:
    import soundfile as sf

    segments = [
        SubtitleSegment(299.0, 299.8, "A"),
        SubtitleSegment(300.5, 301.5, "B"),
        SubtitleSegment(302, 303, "C"),
        SubtitleSegment(310, 311, "D"),
    ]
    calls = []

    def generated(**kwargs):
        items, reviews = [], []
        for group in kwargs["groups"]:
            calls.append(group.index)
            for index, segment in group.segments:
                wav = np.full(40 if index == 1 else 10, index / 10, dtype=np.float32)
                wav[-1] = index / 10 + 0.01
                items.append((index, segment, wav))
                reviews.append(_entry(index, segment, len(wav) / 10))
        return items, 10 if items else None, reviews

    monkeypatch.setattr(chunks, "generate_context_group_items", generated)
    loader = mock.Mock(return_value=object())
    monkeypatch.setattr(chunks, "load_qwen_tts_model", loader)
    monkeypatch.setattr(chunks, "get_media_duration_seconds", lambda *args: 311)
    output = tmp_path / "test_tts.wav"
    kwargs = dict(
        segments=segments,
        audio_out=output,
        chunks_dir=tmp_path / "chunks",
        video_path=tmp_path / "video.mp4",
        tts_model_name="local",
        tts_language="English",
        tts_speaker="Aiden",
        tts_instruct="steady",
        device="cuda",
        attn_implementation="sdpa",
        context_max_sentences=2,
    )
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    first_audio, _ = sf.read(output, dtype="float32")
    reviews = [
        json.loads(line)
        for line in (tmp_path / "data/report/tts/test_tts_review.jsonl").read_text().splitlines()
    ]
    assert [entry["subtitle_index"] for entry in reviews] == [1, 2, 3, 4]
    assert reviews[2]["timing_shift"] > 2.0  # Collision crosses chunk owners.
    assert reviews[3]["timing_shift"] == 0.0
    for entry in reviews:
        index = entry["subtitle_index"]
        first, last = entry["cache_start_sample"], entry["cache_end_sample"]
        assert last - first == (40 if index == 1 else 10)
        assert first_audio[last - 1] == pytest.approx(index / 10 + 0.01, abs=0.0001)
    calls.clear()
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    assert calls == [0, 1, 2]
    assert loader.call_count == 2
    np.testing.assert_array_equal(sf.read(output, dtype="float32")[0], first_audio)
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs, rerun_chunk=1)
    assert calls == [0, 1, 2, 0, 1, 2]  # Explicit rerun includes the boundary owner.
    np.testing.assert_array_equal(sf.read(output, dtype="float32")[0], first_audio)


def test_outdated_chunk_review_is_rejected(tmp_path) -> None:
    path = tmp_path / "chunk.review.jsonl"
    segment = SubtitleSegment(0, 1, "A")
    path.write_text('{"subtitle_index": 1, "action": "used_proportional_fallback"}\n')
    assert not core.tts_review_is_current(path, [(1, segment)])
    entry = _entry(1, segment, 0.5)
    overlay_tts_items([(1, segment, np.ones(5))], 10, 1, reviews=[entry])
    write_tts_review_log(path, [entry])
    assert core.tts_review_is_current(path, [(1, segment)])
    assert not core.tts_review_is_current(path, [(1, SubtitleSegment(0, 1, "Changed"))])


@pytest.mark.parametrize("mode", ["replace", "mix"])
def test_mux_does_not_cut_tts_at_video_end(monkeypatch, tmp_path, mode) -> None:
    from transcript_video.processing import media

    runner = mock.Mock()
    monkeypatch.setattr(media, "get_media_duration_seconds", lambda *args: 1.0)
    monkeypatch.setattr(media, "get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(media, "run_command", runner)
    getattr(media, f"mux_audio_into_video_{mode}")(
        tmp_path / "video.mp4", tmp_path / "tts.wav", tmp_path / "out.mp4"
    )
    command = runner.call_args.args[0]
    assert "-t" not in command and "-shortest" not in command
    assert not any("atrim" in str(arg) for arg in command)


@pytest.mark.parametrize("recognized", ["first", "second"])
def test_unrecognized_neighbor_never_leaks_into_aligned_sentence(recognized) -> None:
    group = build_tts_context_groups(
        [SubtitleSegment(0, 1, "first"), SubtitleSegment(1, 2, "second")]
    )[0]
    aligned, failures = align_context_group(group, [WordTiming(recognized, 0.5, 1.0)], 2.0)
    assert not aligned
    assert set(failures) == {1, 2}


def test_only_low_coverage_member_is_regenerated(monkeypatch) -> None:
    segments = [SubtitleSegment(0, 1, "one two"), SubtitleSegment(3, 4, "three four five")]
    context = np.zeros(2000, dtype=np.float32)
    context[:500] = 0.1
    context[1000:1600] = 0.1
    individual = np.full(1500, 0.4, dtype=np.float32)
    model = mock.Mock()
    model.generate_custom_voice.side_effect = [([context], 1000), ([individual], 1000)]
    monkeypatch.setattr(
        core,
        "transcribe_word_timings",
        mock.Mock(
            side_effect=[
                [
                    WordTiming("one", 0, 0.2),
                    WordTiming("two", 0.3, 0.5),
                    WordTiming("three", 1, 1.2),
                    WordTiming("five", 1.4, 1.6),
                ],
                [
                    WordTiming("three", 0, 0.2),
                    WordTiming("four", 0.3, 0.5),
                    WordTiming("five", 0.6, 0.8),
                ],
            ]
        ),
    )
    items, _, reviews = _generate(model, segments, object())
    assert model.generate_custom_voice.call_count == 2
    assert model.generate_custom_voice.call_args.kwargs["text"] == segments[1].text
    assert reviews[0]["action"] == "context_aligned"
    assert reviews[1]["action"] == "regenerated_individual_sentence"
    np.testing.assert_array_equal(items[1][2], individual)


def test_boundary_individual_fallback_is_generated_and_written_once(tmp_path, monkeypatch) -> None:
    import soundfile as sf

    segments = [SubtitleSegment(299.6, 299.9, "first"), SubtitleSegment(300.2, 300.8, "second")]
    model = mock.Mock()
    model.generate_custom_voice.side_effect = [
        ([np.full(100, 0.8)], 100),
        ([np.full(100, 0.2)], 100),
        ([np.full(100, 0.4)], 100),
    ]
    monkeypatch.setattr(chunks, "load_qwen_tts_model", lambda *args: model)
    monkeypatch.setattr(chunks, "get_media_duration_seconds", lambda *args: 302)
    output = tmp_path / "test_tts.wav"
    chunks.synthesize_tts_audio_by_time_chunks(
        segments,
        output,
        tmp_path / "chunks",
        tmp_path / "video.mp4",
        "local",
        "English",
        "Aiden",
        "steady",
        "cuda",
        "sdpa",
        max_speedup=1.0,
    )
    audio, _ = sf.read(output, dtype="float32")
    assert [call.kwargs["text"] for call in model.generate_custom_voice.call_args_list] == [
        "first second",
        "first",
        "second",
    ]
    assert np.count_nonzero(np.isclose(audio, 0.2, atol=0.0001)) == 100
    assert np.count_nonzero(np.isclose(audio, 0.4, atol=0.0001)) == 100
    assert not np.any(np.isclose(audio, 0.8, atol=0.0001))


def test_failed_stretch_keeps_full_waveform_and_applied_speed_one(monkeypatch) -> None:
    segment = SubtitleSegment(0, 1, "one")
    following = SubtitleSegment(1, 2, "two")
    model = mock.Mock()
    model.generate_custom_voice.return_value = ([np.full(4000, 0.3)], 1000)
    monkeypatch.setattr(
        core, "_pitch_preserving_speedup", mock.Mock(side_effect=RuntimeError("ffmpeg failed"))
    )
    items, _, reviews = _generate(model, [segment, following], max_speedup=1.25)
    assert len(items[0][2]) == 4000
    assert reviews[0]["applied_speedup"] == 1.0
    assert "time_stretch_failed" in reviews[0]["review_reason"]


@pytest.mark.integration
def test_real_ffmpeg_speedup_preserves_pitch_and_final_marker() -> None:
    sr = 24000
    time = np.arange(4 * sr) / sr
    wav = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    wav[-2400:] = 0.8 * np.sin(2 * np.pi * 880 * time[-2400:])
    fitted = fit_wav_to_available_duration(wav, sr, 3.0, 1.25)
    assert len(fitted) / sr == pytest.approx(4 / 1.25, abs=0.05)
    assert np.max(np.abs(fitted[-2000:])) > 0.7
    frequencies = np.fft.rfftfreq(sr, 1 / sr)
    assert frequencies[np.argmax(np.abs(np.fft.rfft(fitted[:sr])))] == pytest.approx(440, abs=2)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["replace", "mix"])
def test_real_mux_retains_audio_after_video_ends(tmp_path, mode) -> None:
    import soundfile as sf

    from transcript_video.hardware import get_ffmpeg_exe
    from transcript_video.process_runner import run_ffmpeg
    from transcript_video.processing import media

    video, audio, output, decoded = [
        tmp_path / name for name in ("video.mp4", "tts.wav", "out.mp4", "decoded.wav")
    ]
    run_ffmpeg(
        [
            get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=10:d=0.4",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=mono:d=0.4",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            video,
        ]
    )
    time = np.arange(28800) / 24000
    wav = np.zeros(28800, dtype=np.float32)
    wav[24000:] = 0.5 * np.sin(2 * np.pi * 440 * time[24000:])
    sf.write(audio, wav, 24000)
    getattr(media, f"mux_audio_into_video_{mode}")(video, audio, output)
    run_ffmpeg([get_ffmpeg_exe(), "-y", "-i", output, "-vn", "-c:a", "pcm_f32le", decoded])
    recovered, sr = sf.read(decoded, dtype="float32")
    assert len(recovered) / sr >= 1.19
    assert np.max(np.abs(recovered[round(sr * 1.1) :])) > 0.1


def test_interrupted_chunk_write_invalidates_old_sentence_ranges(tmp_path, monkeypatch) -> None:
    import soundfile as sf

    segment = SubtitleSegment(0, 1, "one")
    model = mock.Mock()
    model.generate_custom_voice.return_value = ([np.full(1000, 0.2)], 1000)
    output = tmp_path / "chunk.wav"
    kwargs = dict(
        model=model,
        chunk_index=0,
        chunk_start=0,
        chunk_end=2,
        chunk_segments=[segment],
        chunk_audio_out=output,
        tts_language="English",
        tts_speaker="Aiden",
        tts_instruct="steady",
    )
    chunks.synthesize_one_fixed_time_chunk(**kwargs)
    review_path = ProjectPaths.from_root(tmp_path).tts_review_path(output, chunk=True)
    assert core.tts_review_is_current(review_path, [(1, segment)])
    monkeypatch.setattr(sf, "write", mock.Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        chunks.synthesize_one_fixed_time_chunk(**kwargs)
    assert not core.tts_review_is_current(review_path, [(1, segment)])


@pytest.mark.parametrize("tail_ms", [100, 150, 200, 250, 300])
def test_information_tail_and_this_onset_stay_separate_through_rebuild(
    tmp_path, monkeypatch, tail_ms
):
    import soundfile as sf

    segments = [
        SubtitleSegment(299.5, 300, "I have this information."),
        SubtitleSegment(300.01, 301, "This information belongs to this topic."),
    ]
    wav = np.zeros(1800, dtype=np.float32)
    wav[100:500] = 0.2
    tail = np.random.default_rng(42).uniform(-0.08, 0.08, tail_ms).astype(np.float32)
    wav[500 : 500 + tail_ms] = tail
    wav[500 + tail_ms : 1000] = 0.003
    wav[1000:1060] = 0.6
    wav[1060:1600] = 0.4
    model = mock.Mock()
    model.generate_custom_voice.return_value = ([wav], 1000)
    monkeypatch.setattr(
        core,
        "transcribe_word_timings",
        lambda *args: [
            WordTiming("I", 0.1, 0.16),
            WordTiming("have", 0.18, 0.26),
            WordTiming("this", 0.28, 0.36),
            WordTiming("information", 0.38, 0.5),
            WordTiming("This", 1.02, 1.08),
            WordTiming("information", 1.1, 1.2),
            WordTiming("belongs", 1.22, 1.3),
            WordTiming("to", 1.32, 1.36),
            WordTiming("this", 1.38, 1.44),
            WordTiming("topic", 1.46, 1.55),
        ],
    )
    monkeypatch.setattr(chunks, "load_qwen_tts_model", lambda *args: model)
    monkeypatch.setattr(chunks, "load_faster_whisper_aligner", lambda *args: object())
    monkeypatch.setattr(chunks, "get_media_duration_seconds", lambda *args: 302)
    paths = ProjectPaths.from_root(tmp_path)
    output = paths.audio_dir / "test_tts.wav"
    reports = paths.tts_review_path(output)
    kwargs = dict(
        segments=segments,
        audio_out=output,
        chunks_dir=paths.audio_dir / "test_tts_chunks",
        video_path=tmp_path / "video.mp4",
        tts_model_name="local",
        tts_language="English",
        tts_speaker="Aiden",
        tts_instruct="steady",
        device="cuda",
        attn_implementation="sdpa",
        alignment_model_name="local-aligner",
        max_speedup=1.0,
        review_log_path=reports,
    )
    from transcript_video.events import RecordingObserver, event_scope

    record = RecordingObserver()
    with event_scope(record, run_id="test", video="test.mp4"):
        retimed_segments = chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    progress = [event for event in record.events if event.context.operation == "chunks"]
    assert progress[-1].current == progress[-1].total == 2
    assert progress[-1].details["sentences"] == progress[-1].details["total_sentences"] == 2
    assert [event.current for event in progress] == sorted(event.current for event in progress)
    assert all(event.context.run_id == "test" for event in record.events)
    entries = [json.loads(line) for line in reports.read_text(encoding="utf-8").splitlines()]
    assert all(entry["action"] == "context_aligned" for entry in entries)
    assert retimed_segments == [
        SubtitleSegment(entry["retimed_start"], entry["retimed_end"], entry["text"])
        for entry in entries
    ]
    audio, sr = sf.read(output, dtype="float32")
    start, end = entries[0]["cache_start_sample"], entries[0]["cache_end_sample"]
    np.testing.assert_allclose(audio[start + 500 : start + 500 + tail_ms], tail, atol=1 / 32768)
    assert not np.any(np.isclose(audio[start:end], 0.6, atol=0.001))
    next_start = entries[1]["cache_start_sample"]
    assert np.all(np.isclose(audio[next_start : next_start + 60], 0.6, atol=0.001))
    assert next_start - end >= round(core.MIN_GAP_SECONDS * sr)
    assert np.all(audio[end:next_start] == 0)
    assert not list(paths.audio_dir.rglob("*.json*"))
    chunk_logs = list((paths.report_dir / "tts/test_tts_chunks").glob("*.review.jsonl"))
    assert len(chunk_logs) == 2
    assert "\n  {" in reports.with_suffix(".pretty.json").read_text(encoding="utf-8")
    assert json.loads(reports.with_suffix(".pretty.json").read_text(encoding="utf-8")) == entries
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    assert model.generate_custom_voice.call_count == 2
    np.testing.assert_array_equal(sf.read(output, dtype="float32")[0], audio)
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs, rerun_chunk=1)
    assert model.generate_custom_voice.call_count == 3
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs, regenerate_all_chunks=True)
    assert model.generate_custom_voice.call_count == 4


def test_information_this_without_acoustic_gap_regenerates_both_sentences(monkeypatch):
    segments = [
        SubtitleSegment(0, 1, "I have this information."),
        SubtitleSegment(1.01, 2, "This information belongs to this topic."),
    ]
    context = np.full(2000, 0.8, dtype=np.float32)
    first = np.full(900, 0.2, dtype=np.float32)
    first[-150:] = 0.09
    second = np.full(900, 0.4, dtype=np.float32)
    model = mock.Mock()
    model.generate_custom_voice.side_effect = [([context], 1000), ([first], 1000), ([second], 1000)]
    monkeypatch.setattr(
        core,
        "transcribe_word_timings",
        mock.Mock(
            side_effect=[
                [
                    WordTiming(word, start, end)
                    for word, start, end in (
                        ("I", 0.1, 0.16),
                        ("have", 0.18, 0.26),
                        ("this", 0.28, 0.36),
                        ("information", 0.38, 0.5),
                        ("This", 0.65, 0.72),
                        ("information", 0.74, 0.84),
                        ("belongs", 0.86, 0.94),
                        ("to", 0.96, 1.0),
                        ("this", 1.02, 1.08),
                        ("topic", 1.1, 1.2),
                    )
                ],
                [
                    WordTiming(word, i * 0.1, i * 0.1 + 0.08)
                    for i, word in enumerate(("I", "have", "this", "information"))
                ],
                [
                    WordTiming(word, i * 0.1, i * 0.1 + 0.08)
                    for i, word in enumerate(
                        ("This", "information", "belongs", "to", "this", "topic")
                    )
                ],
            ]
        ),
    )
    items, _, reviews = _generate(model, segments, object())
    assert model.generate_custom_voice.call_count == 3
    assert all(entry["action"] == "regenerated_individual_sentence" for entry in reviews)
    assert reviews[0]["boundary_safe"] is False
    assert reviews[0]["boundary_reason"] == "no_safe_acoustic_gap"
    np.testing.assert_array_equal(items[0][2], first)
    np.testing.assert_array_equal(items[1][2], second)


def test_context_slice_with_active_waveform_end_falls_back_without_cropping(monkeypatch):
    segment = SubtitleSegment(0, 1, "information")
    context = np.full(1000, 0.2, dtype=np.float32)
    individual = np.full(1200, 0.4, dtype=np.float32)
    model = mock.Mock()
    model.generate_custom_voice.side_effect = [([context], 1000), ([individual], 1000)]
    monkeypatch.setattr(
        core,
        "transcribe_word_timings",
        mock.Mock(
            side_effect=[
                [WordTiming("information", 0.1, 0.5)],
                [WordTiming("information", 0.1, 0.5)],
            ]
        ),
    )

    items, _, reviews = _generate(model, [segment], object())

    assert reviews[0]["action"] == "regenerated_individual_sentence"
    assert reviews[0]["boundary_reason"] == "no_safe_trailing_silence"
    np.testing.assert_array_equal(items[0][2], individual)


def test_release_gap_for_nearby_non_overlapping_sentences():
    segments = [SubtitleSegment(0, 1, "A"), SubtitleSegment(1.01, 2, "B")]
    entries = [_entry(i, segment, 1) for i, segment in enumerate(segments, 1)]
    audio = overlay_tts_items(
        [(i, segment, np.full(1000, 0.2)) for i, segment in enumerate(segments, 1)],
        1000,
        2,
        reviews=entries,
    )
    assert entries[1]["actual_start"] >= 1 + core.MIN_GAP_SECONDS
    assert np.count_nonzero(audio[1000:1120]) == 0


def test_normal_chunk_runs_regenerate_without_provenance(tmp_path, monkeypatch, caplog):
    import soundfile as sf

    paths = ProjectPaths.from_root(tmp_path)
    paths.create_dirs()
    output = paths.audio_dir / "legacy_tts.wav"
    chunk_dir = paths.audio_dir / "legacy_tts_chunks"
    chunk_dir.mkdir()
    segment = SubtitleSegment(0, 1, "old")
    entry = _entry(1, segment, 1)
    overlay_tts_items([(1, segment, np.full(1000, 0.7))], 1000, 2, reviews=[entry])
    entry["schema_version"] = 2
    sf.write(chunk_dir / "legacy_tts_chunk_000.wav", np.full(1000, 0.7), 1000)
    write_tts_review_log(chunk_dir / "legacy_tts_chunk_000.review.jsonl", [entry])
    model = mock.Mock()
    model.generate_custom_voice.return_value = ([np.full(1000, 0.2)], 1000)
    monkeypatch.setattr(chunks, "load_qwen_tts_model", lambda *args: model)
    monkeypatch.setattr(chunks, "get_media_duration_seconds", lambda *args: 2)
    kwargs = dict(
        segments=[segment],
        audio_out=output,
        chunks_dir=chunk_dir,
        video_path=tmp_path / "video.mp4",
        tts_model_name="local",
        tts_language="English",
        tts_speaker="Aiden",
        tts_instruct="steady",
        device="cuda",
        attn_implementation="sdpa",
        max_speedup=1,
    )
    with caplog.at_level("INFO"):
        chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    assert "regenerating safely" in caplog.text
    assert model.generate_custom_voice.call_count == 2
    chunks.synthesize_tts_audio_by_time_chunks(**kwargs)
    assert model.generate_custom_voice.call_count == 4
    assert not list(tmp_path.rglob("*.provenance.json"))
    assert (paths.report_dir / "tts/legacy_tts_chunks/legacy_tts_chunk_000.review.jsonl").exists()


@pytest.mark.parametrize(
    "mode,generation", [("timed", "full"), ("timed", "chunked"), ("simple", "full")]
)
def test_pipeline_routes_reports_and_preserves_tts_modes(tmp_path, monkeypatch, mode, generation):
    from transcript_video.config import RunSettings
    from transcript_video.processing import pipeline
    from transcript_video.processing.subtitles import write_srt

    paths = ProjectPaths.from_root(tmp_path)
    paths.create_dirs()
    settings = RunSettings.defaults()
    settings.tts.enabled, settings.tts.mode, settings.tts.generation_mode = True, mode, generation
    settings.tts.split_audio = False
    video = tmp_path / "lesson.mp4"
    video.touch()
    (tmp_path / "model").mkdir()
    (tmp_path / "model/model.bin").touch()
    source = paths.source_subtitle_dir / "lesson_vi_faster.srt"
    translated = paths.translated_subtitle_dir / "lesson_en.srt"
    write_srt([SubtitleSegment(0, 1, "nguồn")], source)
    write_srt([SubtitleSegment(0, 1, "one")], translated)
    monkeypatch.setattr(pipeline, "burn_subtitles", mock.Mock())
    monkeypatch.setattr(pipeline, "mux_audio_into_video_replace", mock.Mock())
    generators = {
        name: mock.Mock()
        for name in (
            "synthesize_simple_tts_audio",
            "synthesize_timed_tts_audio",
            "synthesize_tts_audio_by_time_chunks",
        )
    }
    for name, generator in generators.items():
        monkeypatch.setattr(pipeline, name, generator)
    pipeline.process_video(video, tmp_path / "model", source, translated, paths, settings)
    chosen = (
        "synthesize_tts_audio_by_time_chunks"
        if generation == "chunked"
        else f"synthesize_{mode}_tts_audio"
    )
    assert generators[chosen].call_count == 1
    assert sum(generator.call_count for generator in generators.values()) == 1
    if mode == "timed":
        assert (
            generators[chosen].call_args.kwargs["review_log_path"]
            == paths.report_dir / "tts/lesson_tts_review.jsonl"
        )

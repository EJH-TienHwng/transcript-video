"""Text fidelity and acoustic review hints, never pronunciation/prosody scores."""

from __future__ import annotations

from difflib import SequenceMatcher


def compare_text(expected: str, recognized: str) -> dict:
    from .core import _normalized_words

    wanted, heard = _normalized_words(expected), _normalized_words(recognized)
    missing, added, matched = [], [], 0
    final_word_present = False
    errors = 0
    for tag, a, b, c, d in SequenceMatcher(None, wanted, heard, autojunk=False).get_opcodes():
        if tag == "equal":
            matched += b - a
            final_word_present |= b == len(wanted)
        else:
            missing.extend(wanted[a:b])
            added.extend(heard[c:d])
            errors += max(b - a, d - c)
    first = bool(wanted and heard and wanted[0] == heard[0])
    last = bool(wanted and heard and wanted[-1] == heard[-1])
    reasons = []
    if wanted and not final_word_present:
        reasons.append("missing_final_word")
    if missing:
        reasons.append("missing_words")
    if added:
        reasons.append("added_words")
    return dict(
        expected_text=expected,
        recognized_text=recognized,
        normalized_expected_words=wanted,
        normalized_recognized_words=heard,
        word_coverage=matched / len(wanted) if wanted else None,
        missing_words=missing,
        added_words=added,
        first_word_match=first,
        last_word_match=last,
        # ponytail: SequenceMatcher edit ratio, use minimum-edit WER if benchmark comparison needs it.
        word_error_ratio=errors / max(1, len(wanted)),
        comparison_method="sequence_matcher_v1",
        text_review_reasons=reasons,
    )


def check_tail(wav, sample_rate: int) -> dict:
    import numpy as np

    audio = np.asarray(wav, dtype=np.float64).reshape(-1)
    window = max(1, round(sample_rate * 0.02))
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    tail = float(np.sqrt(np.mean(audio[-window:] ** 2))) if len(audio) else 0.0
    # ponytail: 20 ms energy heuristic can flag unpadded fricatives; calibrate against listened audio.
    return dict(
        tail_window_seconds=0.02,
        tail_rms=tail,
        waveform_rms=rms,
        possible_truncated_tail=tail > 0.01 and tail > rms * 0.25,
    )


def verify_sentences(audio, sample_rate: int, entries: list[dict], aligner, language: str) -> None:
    """Inspect final placed sentence ranges, excluding timeline/chunk silence."""
    from .core import _add_review_reason, transcribe_word_timings

    for entry in entries:
        wav = audio[entry["cache_start_sample"] : entry["cache_end_sample"]]
        if aligner is None:
            entry["final_verification_status"] = "unavailable"
            _add_review_reason(entry, "final_verification_unavailable")
            continue
        try:
            # No expected-text prompt: the verifier must not be primed to hallucinate the answer.
            words = transcribe_word_timings(aligner, wav, sample_rate, "", language)
        except (RuntimeError, ValueError, OSError) as exc:
            entry["final_verification_status"] = "failed"
            entry["final_verification_error"] = str(exc)
            _add_review_reason(entry, "final_verification_failed")
            continue
        entry.update(compare_text(str(entry["text"]), " ".join(word.text for word in words)))
        entry["final_verification_status"] = "completed"
        for reason in entry["text_review_reasons"]:
            _add_review_reason(entry, reason)


def duration_qa(source: float | None, audio: float | None, output: float | None) -> dict:
    return dict(
        source_video_duration=source,
        tts_audio_duration=audio,
        final_output_duration=output,
        tts_duration_delta=audio - source if audio is not None and source is not None else None,
        duration_delta=output - source if output is not None and source is not None else None,
    )

from __future__ import annotations

import json
import logging
import re
import tempfile
import textwrap
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from math import ceil
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

from ...events import warn

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from ...config import ProjectPaths, SubtitleSegment, find_project_root
from ...context import log_context
from ...events import EventKind, PipelineStage, emit, stage_context
from ...hardware import get_ffmpeg_exe, resolve_torch_device
from ...process_runner import run_ffmpeg
from ..media import get_media_duration_seconds

logger = logging.getLogger(__name__)

# A separate 120 ms release remains even when both waveforms contain speech up to their edges.
MIN_GAP_SECONDS = 0.12
MIN_AVAILABLE_SECONDS = 0.001
ALIGNMENT_MIN_CONFIDENCE = 1.0
ALIGNMENT_START_PADDING_SECONDS = 0.04
# Regression budget: a 150 ms final fricative plus a 30 ms guard.
# If this budget reaches another word, regenerate instead of shortening the tail.
ALIGNMENT_END_PADDING_SECONDS = 0.18
# Half a second is visible relative to subtitle cues; diagnose it instead of dropping speech.
TIMING_SHIFT_REVIEW_SECONDS = 0.5
TTS_REVIEW_VERSION = 3


@dataclass(slots=True)
class TTSContextGroup:
    index: int
    segments: list[tuple[int, SubtitleSegment]]

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for _, segment in self.segments)


@dataclass(slots=True)
class AlignedTTSSegment:
    subtitle_index: int
    segment: SubtitleSegment
    source_start: float
    source_end: float
    confidence: float


@dataclass(slots=True)
class WordTiming:
    text: str
    start: float
    end: float


def split_text_for_tts(text: str, max_chars: int = 450) -> list[str]:
    """Split long text into smaller chunks for untimed TTS generation."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in filter(
        None, (part.strip() for part in re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", text))
    ):
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                textwrap.wrap(
                    sentence,
                    width=max_chars,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )
        elif len(current) + len(sentence) + bool(current) <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def build_tts_context_groups(
    segments: list[SubtitleSegment],
    max_sentences: int = 4,
    max_chars: int = 450,
    hard_break_seconds: float = 3.0,
) -> list[TTSContextGroup]:
    """Group complete SRT sentences for acoustic context without changing timestamps."""
    if max_sentences < 1 or max_chars < 1 or hard_break_seconds < 0:
        raise ValueError("TTS context limits are outside their supported ranges.")
    indexed = sorted(
        ((index, segment) for index, segment in enumerate(segments, 1) if segment.text.strip()),
        key=lambda item: item[1].start,
    )
    groups: list[TTSContextGroup] = []
    current: list[tuple[int, SubtitleSegment]] = []
    current_chars = 0
    for item in indexed:
        text = item[1].text.strip()
        gap = item[1].start - current[-1][1].end if current else 0.0
        if current and (
            len(current) >= max_sentences
            or current_chars + 1 + len(text) > max_chars
            or gap > hard_break_seconds
        ):
            groups.append(TTSContextGroup(len(groups), current))
            current = []
            current_chars = 0
        current.append(item)
        current_chars += len(text) + bool(current_chars)
    if current:
        groups.append(TTSContextGroup(len(groups), current))
    return groups


@stage_context(
    PipelineStage.TTS, "Loading Qwen TTS model", operation="load_qwen", completed="Qwen TTS ready"
)
def load_qwen_tts_model(tts_model_name: str, device: str, attn_implementation: str):
    """Load Qwen lazily so non-TTS commands do not require its runtime."""
    import torch
    from qwen_tts import Qwen3TTSModel

    device = resolve_torch_device(device, "Qwen TTS")
    kwargs = {
        "device_map": "cuda:0" if device == "cuda" else "cpu",
        "dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    logger.info("Loading Qwen TTS model: %s", tts_model_name)
    return Qwen3TTSModel.from_pretrained(tts_model_name, **kwargs)


@stage_context(
    PipelineStage.TTS,
    "Loading Whisper aligner",
    operation="load_aligner",
    completed="Whisper aligner initialization finished",
)
def load_faster_whisper_aligner(model_name: str | Path):
    """Use CPU alignment so the aligner does not compete with Qwen for VRAM."""
    try:
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper aligner on CPU: %s", model_name)
        return WhisperModel(str(model_name), device="cpu", compute_type="int8")
    except Exception as exc:
        warn(logger, "Could not load faster-whisper alignment; using reviewed fallback: %s", exc)
        return None


def generate_qwen_custom_voice(
    model, text: str, language: str, speaker: str, instruct: str
) -> tuple[NDArray[np.float32], int]:
    """Generate one waveform with Qwen CustomVoice."""
    wavs, sr = model.generate_custom_voice(
        text=text, language=language, speaker=speaker, instruct=instruct
    )
    if wavs is None or len(wavs) == 0:
        raise ValueError("Qwen TTS returned no waveform.")
    if isinstance(sr, bool) or not isinstance(sr, Integral) or sr <= 0:
        raise ValueError(f"Invalid TTS sample rate: {sr}")
    import numpy as np

    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    if not len(wav) or not np.isfinite(wav).all() or not np.any(wav):
        raise ValueError("Qwen TTS returned empty, silent, or non-finite audio.")
    return wav, int(sr)


def _normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    return re.findall(r"\w+(?:'\w+)*", normalized, flags=re.UNICODE)


def align_context_group(
    group: TTSContextGroup,
    words: list[WordTiming],
    audio_duration: float,
    min_confidence: float = ALIGNMENT_MIN_CONFIDENCE,
    confidences: dict[int, float] | None = None,
) -> tuple[list[AlignedTTSSegment], dict[int, str]]:
    """Match normalized ASR words monotonically back to the original sentences."""
    expected: list[str] = []
    owners: list[int] = []
    owner_positions: dict[int, tuple[int, int]] = {}
    for subtitle_index, segment in group.segments:
        sentence_words = _normalized_words(segment.text)
        first = len(expected)
        expected.extend(sentence_words)
        owners.extend([subtitle_index] * len(sentence_words))
        owner_positions[subtitle_index] = (first, len(expected) - 1)

    recognized: list[str] = []
    valid_words: list[WordTiming] = []
    for word in words:
        normalized = _normalized_words(word.text)
        if normalized and 0 <= word.start < word.end <= audio_duration + 1e-3:
            recognized.extend(normalized)
            valid_words.extend([word] * len(normalized))

    matched: dict[int, list[tuple[int, WordTiming]]] = {}
    recognized_positions: dict[int, int] = {}
    for block in SequenceMatcher(None, expected, recognized, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            expected_position = block.a + offset
            recognized_positions[expected_position] = block.b + offset
            matched.setdefault(owners[expected_position], []).append(
                (expected_position, valid_words[block.b + offset])
            )

    aligned: list[AlignedTTSSegment] = []
    failures: dict[int, str] = {}
    previous_end = 0.0
    for position, (subtitle_index, segment) in enumerate(group.segments):
        matches = matched.get(subtitle_index, [])
        confidence = len(matches) / max(1, len(_normalized_words(segment.text)))
        if confidences is not None:
            confidences[subtitle_index] = confidence
        if not matches:
            failures[subtitle_index] = "alignment_failed"
            continue
        if confidence < min_confidence or (
            owner_positions[subtitle_index] != (matches[0][0], matches[-1][0])
        ):
            failures[subtitle_index] = "alignment_low_confidence"
            continue
        first_word, last_word = matches[0][1], matches[-1][1]
        first_rec = recognized_positions[matches[0][0]]
        last_rec = recognized_positions[matches[-1][0]]
        source_start = max(0.0, first_word.start - ALIGNMENT_START_PADDING_SECONDS)
        source_end = min(audio_duration, last_word.end + ALIGNMENT_END_PADDING_SECONDS)
        # Use transcript order: filtering by time would hide overlapping ASR word ranges.
        preceding = valid_words[:first_rec]
        following = valid_words[last_rec + 1 :]
        if position == 0 and not preceding:
            source_start = 0.0
        if position == len(group.segments) - 1 and not following:
            source_end = audio_duration
        neighbor_boundaries = []
        if position > 0:
            neighbor_boundaries.append(owner_positions[group.segments[position - 1][0]][1])
        if position + 1 < len(group.segments):
            neighbor_boundaries.append(owner_positions[group.segments[position + 1][0]][0])
        # Never resolve uncertain boundaries by cropping either sentence's speech.
        if (
            source_end <= source_start
            or any(boundary not in recognized_positions for boundary in neighbor_boundaries)
            or last_rec - first_rec + 1 != len(matches)
            or source_start < previous_end
            or any(word.end > source_start for word in preceding)
            or any(word.start - ALIGNMENT_START_PADDING_SECONDS < source_end for word in following)
            # The previous tail needs protection even if that sentence was regenerated.
            or (
                position > 0
                and preceding
                and preceding[-1].end + ALIGNMENT_END_PADDING_SECONDS > source_start
            )
            or any(left[1].end > right[1].start for left, right in pairwise(matches))
        ):
            failures[subtitle_index] = "invalid_audio_range"
            continue
        aligned.append(
            AlignedTTSSegment(subtitle_index, segment, source_start, source_end, confidence)
        )
        previous_end = source_end
    return aligned, failures


def _alignment_language(language: str) -> str | None:
    names = {
        "english": "en",
        "vietnamese": "vi",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "russian": "ru",
        "portuguese": "pt",
        "italian": "it",
    }
    value = language.strip().casefold()
    return names.get(value, value if len(value) in {2, 3} else None)


def transcribe_word_timings(
    aligner, wav, sample_rate: int, expected_text: str, language: str
) -> list[WordTiming]:
    """Get word timestamps from faster-whisper; its ndarray input must be 16 kHz."""
    import numpy as np

    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    if sample_rate != 16000 and len(audio) > 1:
        output_length = max(1, round(len(audio) * 16000 / sample_rate))
        audio = np.interp(
            np.linspace(0, len(audio), output_length, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    segments, _ = aligner.transcribe(
        audio,
        language=_alignment_language(language),
        beam_size=1,
        word_timestamps=True,
        vad_filter=False,
        initial_prompt=expected_text,
        condition_on_previous_text=False,
    )
    return [
        WordTiming(word.word, float(word.start), float(word.end))
        for segment in segments
        for word in (segment.words or [])
        if word.start is not None and word.end is not None
    ]


def _pitch_preserving_speedup(wav, sample_rate: int, speed_factor: float):
    """Change tempo with the already-required FFmpeg without shifting pitch."""
    import numpy as np
    import soundfile as sf

    if speed_factor < 1.0:
        raise ValueError("Timed TTS only supports speed-up, never slow-down.")
    with tempfile.TemporaryDirectory(prefix="transcript-video-atempo-") as directory:
        source = Path(directory) / "source.wav"
        output = Path(directory) / "output.wav"
        sf.write(str(source), np.asarray(wav, dtype=np.float32), sample_rate, subtype="FLOAT")
        run_ffmpeg(
            [
                get_ffmpeg_exe(),
                "-y",
                "-i",
                source,
                "-filter:a",
                f"atempo={speed_factor:.8f}",
                "-c:a",
                "pcm_f32le",
                output,
            ]
        )
        stretched, stretched_rate = sf.read(str(output), dtype="float32")
    if stretched_rate != sample_rate:
        raise ValueError(f"FFmpeg changed sample rate: {stretched_rate} != {sample_rate}")
    return np.asarray(stretched, dtype=np.float32)


def fit_wav_to_available_duration(
    wav,
    sample_rate: int,
    available_duration: float,
    max_speedup: float = 1.15,
    fade_out_seconds: float = 0.04,
):
    """Pitch-preserving speed-up bounded by max_speedup; never truncate speech."""
    import numpy as np

    del fade_out_seconds  # Compatibility with older Python callers.
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero.")
    if available_duration <= 0:
        raise ValueError("available_duration must be greater than zero.")
    if max_speedup < 1.0:
        raise ValueError("max_speedup must be at least 1.0.")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    duration = len(wav) / sample_rate
    if duration <= available_duration or len(wav) < 2 or max_speedup == 1.0:
        return wav
    return _pitch_preserving_speedup(
        wav, sample_rate, min(duration / available_duration, max_speedup)
    )


def generate_individual_sentence_fallback(
    model,
    aligner,
    subtitle_index: int,
    segment: SubtitleSegment,
    language: str,
    speaker: str,
    instruct: str,
    expected_sample_rate: int | None,
) -> tuple[NDArray[np.float32] | None, int | None, dict[str, object]]:
    """Keep the complete best candidate, with at most one additional Qwen attempt."""
    best_wav, best_rate = None, None
    best_score = -1.0
    metadata: dict[str, object] = {"generation_failures": 0}
    errors: list[str] = []
    group = TTSContextGroup(0, [(subtitle_index, segment)])
    for attempt in range(1, 3):
        metadata["individual_attempts"] = attempt
        try:
            wav, sr = generate_qwen_custom_voice(model, segment.text, language, speaker, instruct)
            if expected_sample_rate is not None and sr != expected_sample_rate:
                raise ValueError(f"Inconsistent sample rate: {sr} != {expected_sample_rate}")
        except Exception as exc:
            logger.info(
                "TTS subtitle #%d: individual attempt %d failed: %s", subtitle_index, attempt, exc
            )
            errors.append(str(exc))
            metadata["generation_failures"] = len(errors)
            continue
        score = 0.0
        verified = False
        verification_reason = "verification_unavailable"
        if aligner is not None:
            try:
                words = transcribe_word_timings(aligner, wav, sr, segment.text, language)
                confidences: dict[int, float] = {}
                aligned, _ = align_context_group(
                    group, words, len(wav) / sr, confidences=confidences
                )
                score = confidences.get(subtitle_index, 0.0)
                verified = bool(aligned)
                verification_reason = "" if verified else "individual_coverage_low"
            except Exception as exc:
                logger.info(
                    "TTS subtitle #%d: coverage verification failed: %s", subtitle_index, exc
                )
                verification_reason = "verification_failed"
        if best_wav is None or (score, len(wav) / sr) > (best_score, len(best_wav) / best_rate):
            best_wav, best_rate, best_score = wav, sr, score
            metadata.update(verification_confidence=score, verification_reason=verification_reason)
        if verified:
            best_wav, best_rate = wav, sr
            metadata.update(verification_confidence=score, verification_reason="")
            break
        if verification_reason in {"verification_unavailable", "verification_failed"}:
            break  # Retrying cannot repair an unavailable verifier.
        logger.info("TTS subtitle #%d: coverage %.2f on attempt %d", subtitle_index, score, attempt)
    if errors:
        metadata["generation_errors"] = errors
    return best_wav, best_rate, metadata


def _review_entry(
    *,
    subtitle_index: int,
    segment: SubtitleSegment,
    next_start: float | None,
    available_duration: float,
    generated_duration: float,
    max_speedup: float,
    group_index: int,
    source_start: float | None,
    source_end: float | None,
    action: str,
    reason: str,
    confidence: float | None = None,
) -> dict[str, object]:
    return {
        "schema_version": TTS_REVIEW_VERSION,
        "subtitle_index": subtitle_index,
        "text": segment.text,
        "start": segment.start,
        "end": segment.end,
        "next_start": next_start,
        "available_duration": available_duration,
        "raw_generated_duration": generated_duration,
        "generated_speech_duration": generated_duration,
        "required_speedup": generated_duration / available_duration,
        "applied_speedup": 1.0,
        "final_audio_duration": generated_duration,
        "overflow_duration": max(0.0, generated_duration - available_duration),
        "max_speedup": max_speedup,
        "original_start": segment.start,
        "actual_start": segment.start,
        "timing_shift": 0.0,
        "context_group_index": group_index,
        "alignment_confidence": confidence,
        "alignment_source_start": source_start,
        "alignment_source_end": source_end,
        "action": action,
        "review_reason": reason,
        "generation_failures": 0,
    }


def _add_review_reason(entry: dict[str, object], reason: str) -> None:
    reasons = str(entry.get("review_reason", "")).split(";")
    entry["review_reason"] = ";".join(dict.fromkeys(value for value in [*reasons, reason] if value))


def generate_context_group_items(
    *,
    model,
    aligner,
    groups: list[TTSContextGroup],
    all_segments: list[SubtitleSegment],
    language: str,
    speaker: str,
    instruct: str,
    max_speedup: float,
    video_duration: float | None,
) -> tuple[
    list[tuple[int, SubtitleSegment, NDArray[np.float32]]], int | None, list[dict[str, object]]
]:
    """Generate, reliably extract or regenerate, then fit without cutting speech."""
    ordered = sorted(
        ((index, segment) for index, segment in enumerate(all_segments, 1) if segment.text.strip()),
        key=lambda item: item[1].start,
    )
    next_starts = {
        index: ordered[position + 1][1].start if position + 1 < len(ordered) else None
        for position, (index, _) in enumerate(ordered)
    }
    generated: list[tuple[int, SubtitleSegment, NDArray[np.float32]]] = []
    reviews: list[dict[str, object]] = []
    sample_rate: int | None = None
    sentence_total = sum(len(item.segments) for item in groups)
    for group in groups:
        group_wav = None
        group_error = None
        aligned: list[AlignedTTSSegment] = []
        confidences: dict[int, float] = {}
        failures = {index: "alignment_failed" for index, _ in group.segments}
        try:
            group_wav, sr = generate_qwen_custom_voice(
                model, group.text, language, speaker, instruct
            )
            if sample_rate is not None and sr != sample_rate:
                raise ValueError(f"Inconsistent sample rate: {sr} != {sample_rate}")
            sample_rate = sr
        except Exception as exc:
            group_wav = None
            group_error = str(exc)
            logger.info("TTS context group %d failed: %s", group.index, exc)
        if group_wav is not None and aligner is not None:
            try:
                words = transcribe_word_timings(aligner, group_wav, sr, group.text, language)
                aligned, failures = align_context_group(
                    group, words, len(group_wav) / sr, confidences=confidences
                )
            except Exception as exc:
                logger.info("Alignment failed for context group %d: %s", group.index, exc)
        aligned_by_index = {item.subtitle_index: item for item in aligned}
        for subtitle_index, segment in group.segments:
            with log_context(subtitle=subtitle_index):
                item = aligned_by_index.get(subtitle_index)
                reason = failures.get(subtitle_index, "")
                sentence_wav = None
                metadata: dict[str, object] = {}
                if item is not None:
                    first, last = round(item.source_start * sr), round(item.source_end * sr)
                    if 0 <= first < last <= len(group_wav):
                        sentence_wav = group_wav[first:last].copy()
                    else:
                        reason = "invalid_audio_range"
                action = "context_aligned"
                if sentence_wav is None:
                    reason = reason or "alignment_failed"
                    logger.info(
                        "TTS subtitle #%d: %s (confidence %s); regenerating sentence individually",
                        subtitle_index,
                        reason,
                        confidences.get(subtitle_index),
                    )
                    sentence_wav, individual_rate, metadata = generate_individual_sentence_fallback(
                        model,
                        aligner,
                        subtitle_index,
                        segment,
                        language,
                        speaker,
                        instruct,
                        sample_rate,
                    )
                    action = (
                        "regenerated_individual_sentence"
                        if sentence_wav is not None
                        else "generation_failed"
                    )
                    if individual_rate is not None:
                        sample_rate = individual_rate
                next_start = next_starts[subtitle_index]
                slot_end = (
                    next_start - MIN_GAP_SECONDS
                    if next_start is not None
                    else max(segment.end + 1.0, video_duration or 0.0)
                )
                available = max(MIN_AVAILABLE_SECONDS, slot_end - segment.start)
                raw_duration = len(sentence_wav) / sample_rate if sentence_wav is not None else 0.0
                entry = _review_entry(
                    subtitle_index=subtitle_index,
                    segment=segment,
                    next_start=next_start,
                    available_duration=available,
                    generated_duration=raw_duration,
                    max_speedup=max_speedup,
                    group_index=group.index,
                    source_start=item.source_start if item else None,
                    source_end=item.source_end if item else None,
                    action=action,
                    reason=reason,
                    confidence=confidences.get(subtitle_index),
                )
                entry.update(metadata)
                if group_error:
                    entry["context_generation_error"] = group_error
                    _add_review_reason(entry, "context_generation_failed")
                if metadata.get("verification_reason"):
                    _add_review_reason(entry, str(metadata["verification_reason"]))
                if metadata.get("generation_failures"):
                    _add_review_reason(entry, "individual_generation_error")
                reviews.append(entry)
                if sentence_wav is None:
                    _add_review_reason(entry, "individual_generation_failed")
                    continue
                try:
                    sentence_wav = fit_wav_to_available_duration(
                        sentence_wav, sample_rate, available, max_speedup
                    )
                    entry["applied_speedup"] = min(max(1.0, raw_duration / available), max_speedup)
                except Exception as exc:
                    logger.info(
                        "Could not time-stretch subtitle %d; preserving raw audio: %s",
                        subtitle_index,
                        exc,
                    )
                    _add_review_reason(entry, "time_stretch_failed")
                final_duration = len(sentence_wav) / sample_rate
                entry.update(
                    final_audio_duration=final_duration,
                    overflow_duration=max(0.0, final_duration - available),
                )
                if final_duration > available + 1 / sample_rate:
                    _add_review_reason(
                        entry,
                        "exceeds_max_speedup"
                        if raw_duration / available > max_speedup
                        else "timing_overflow",
                    )
                    logger.info(
                        "TTS subtitle #%d: requires %.2fx, applied %.2fx; preserving full speech with +%.2fs overflow",
                        subtitle_index,
                        raw_duration / available,
                        entry["applied_speedup"],
                        final_duration - available,
                    )
                generated.append((subtitle_index, segment, sentence_wav))
        emit(
            PipelineStage.TTS,
            "Sentences generated",
            current=len(reviews),
            total=sentence_total,
            details={"unit": "sentences"},
        )
    return generated, sample_rate, reviews


def overlay_tts_items(
    items: list[tuple[int, SubtitleSegment, object]],
    sample_rate: int,
    minimum_duration: float,
    offset: float = 0.0,
    reviews: list[dict[str, object]] | None = None,
):
    """Place whole sentences without overlap; original starts recover unused silence."""
    import numpy as np

    entries = {entry["subtitle_index"]: entry for entry in reviews or []}
    placements = []
    previous_end = None
    gap = ceil(MIN_GAP_SECONDS * sample_rate)
    for index, segment, wav in sorted(items, key=lambda item: (item[1].start, item[0])):
        original = max(0, round((segment.start - offset) * sample_rate))
        start = max(original, previous_end + gap) if previous_end is not None else original
        end = start + len(wav)
        previous_end = end
        placements.append((start, end, wav))
        shift = (start - original) / sample_rate
        if entry := entries.get(index):
            # Exact sample ranges also let cache rebuild recover every complete sentence.
            entry.update(
                cache_start_sample=start,
                cache_end_sample=end,
                original_start=segment.start,
                actual_start=offset + start / sample_rate,
                timing_shift=shift,
            )
            reasons = str(entry.get("review_reason", "")).split(";")
            entry["review_reason"] = ";".join(
                reason for reason in reasons if reason != "timing_shift_exceeds_threshold"
            )
            if shift > TIMING_SHIFT_REVIEW_SECONDS:
                _add_review_reason(entry, "timing_shift_exceeds_threshold")
        if shift > 0:
            logger.info("TTS subtitle #%d: shifted +%.3fs to prevent voice overlap", index, shift)
    audio = np.zeros(
        max(1, round(minimum_duration * sample_rate), previous_end or 0), dtype=np.float32
    )
    for start, end, wav in placements:
        audio[start:end] = wav
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        # Linear gain preserves the waveform; hard clipping would distort loud samples.
        logger.info("TTS peak %.3f exceeds full scale; applying uniform gain", peak)
        audio /= peak
    return audio


def write_tts_review_log(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    path.with_suffix(".pretty.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def invalidate_tts_review_log(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".pretty.json").unlink(missing_ok=True)


def tts_review_is_current(path: Path, segments: list[tuple[int, SubtitleSegment]]) -> bool:
    """Old review-only logs cannot prove safe cached sentence boundaries."""
    try:
        entries = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected = {index: segment for index, segment in segments}
        if len(entries) != len(expected) or {entry["subtitle_index"] for entry in entries} != set(
            expected
        ):
            return False
        return all(
            entry["schema_version"] == TTS_REVIEW_VERSION
            and entry["action"] != "generation_failed"
            and entry["text"] == expected[entry["subtitle_index"]].text
            and entry["start"] == expected[entry["subtitle_index"]].start
            and entry["end"] == expected[entry["subtitle_index"]].end
            and isinstance(entry["cache_start_sample"], int)
            and isinstance(entry["cache_end_sample"], int)
            and 0 <= entry["cache_start_sample"] < entry["cache_end_sample"]
            for entry in entries
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def log_tts_summary(total: int, entries: list[dict[str, object]], review_path: Path) -> None:
    counts = {
        "sentences": total,
        "aligned": sum(entry["action"] == "context_aligned" for entry in entries),
        "regenerated": sum(
            entry["action"] == "regenerated_individual_sentence" for entry in entries
        ),
        "shifted": sum(entry.get("timing_shift", 0) > 0 for entry in entries),
        "overflow": sum(entry.get("overflow_duration", 0) > 0 for entry in entries),
        "failures": sum(entry["action"] == "generation_failed" for entry in entries),
        "flagged": sum(bool(entry.get("review_reason")) for entry in entries),
    }
    for entry in entries:
        if entry.get("review_reason"):
            with log_context(subtitle=entry["subtitle_index"], operation="review"):
                emit(
                    PipelineStage.TTS,
                    str(entry["review_reason"]),
                    kind=EventKind.REVIEW,
                    details={
                        key: entry.get(key)
                        for key in ("action", "timing_shift", "overflow_duration")
                    },
                )
    emit(
        PipelineStage.TTS,
        "TTS review report",
        kind=EventKind.ARTIFACT,
        artifact=review_path.with_suffix(".pretty.json"),
        details={"category": "Review"},
    )
    with log_context(operation="quality", subtitle=None, chunk=None):
        emit(
            PipelineStage.TTS,
            "TTS quality",
            kind=EventKind.REVIEW,
            details=counts,
            artifact=review_path.with_suffix(".pretty.json"),
        )


def tts_review_counts(total: int, entries: list[dict[str, object]]) -> tuple[int, int]:
    reviewed = {entry["subtitle_index"] for entry in entries if entry.get("review_reason")}
    not_aligned = {
        entry["subtitle_index"]
        for entry in entries
        if entry["action"]
        in {"regenerated_individual_sentence", "generation_failed", "skipped_audio"}
    }
    return total - len(not_aligned), len(reviewed)


def synthesize_simple_tts_audio(
    segments: list[SubtitleSegment],
    audio_out: Path,
    tts_model_name: str,
    tts_language: str,
    tts_speaker: str,
    tts_instruct: str,
    device: str,
    attn_implementation: str,
) -> None:
    """Generate an untimed continuous voice-over."""
    import numpy as np
    import soundfile as sf

    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    chunks = split_text_for_tts(text)
    if not chunks:
        raise ValueError("No text is available for TTS generation.")
    model = load_qwen_tts_model(tts_model_name, device, attn_implementation)
    wav_list = []
    sample_rate = None
    for chunk in chunks:
        wav, sr = generate_qwen_custom_voice(model, chunk, tts_language, tts_speaker, tts_instruct)
        if sample_rate is not None and sr != sample_rate:
            raise ValueError(f"Inconsistent sample rate: {sr} != {sample_rate}")
        sample_rate = sr
        wav_list.append(np.asarray(wav, dtype=np.float32))
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_out), np.clip(np.concatenate(wav_list), -1.0, 1.0), sample_rate)


def synthesize_timed_tts_audio(
    segments: list[SubtitleSegment],
    audio_out: Path,
    video_path: Path,
    tts_model_name: str,
    tts_language: str,
    tts_speaker: str,
    tts_instruct: str,
    device: str,
    attn_implementation: str,
    alignment_model_name: str | Path | None = None,
    max_speedup: float = 1.15,
    context_max_sentences: int = 4,
    context_max_chars: int = 450,
    context_break_seconds: float = 3.0,
    review_log_path: Path | None = None,
) -> None:
    """Generate contextual TTS and place complete sentences as close to SRT starts as possible."""
    import soundfile as sf

    groups = build_tts_context_groups(
        segments, context_max_sentences, context_max_chars, context_break_seconds
    )
    if not groups:
        raise ValueError("No subtitle segments are available for TTS generation.")
    video_duration = get_media_duration_seconds(video_path) or 0.0
    model = load_qwen_tts_model(tts_model_name, device, attn_implementation)
    aligner = load_faster_whisper_aligner(alignment_model_name) if alignment_model_name else None
    items, sample_rate, reviews = generate_context_group_items(
        model=model,
        aligner=aligner,
        groups=groups,
        all_segments=segments,
        language=tts_language,
        speaker=tts_speaker,
        instruct=tts_instruct,
        max_speedup=max_speedup,
        video_duration=video_duration,
    )
    review_path = review_log_path or ProjectPaths.from_root(find_project_root()).tts_review_path(
        audio_out
    )
    if sample_rate is None or any(entry["action"] == "generation_failed" for entry in reviews):
        write_tts_review_log(review_path, reviews)
        log_tts_summary(sum(len(group.segments) for group in groups), reviews, review_path)
        raise ValueError(f"Qwen TTS could not generate every sentence; see {review_path}")
    minimum_duration = max(video_duration, max(segment.end for segment in segments) + 1.0)
    audio = overlay_tts_items(items, sample_rate, minimum_duration, reviews=reviews)
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    # A failed WAV write must not leave old ranges authorizing reuse of partial audio.
    invalidate_tts_review_log(review_path)
    sf.write(str(audio_out), audio, sample_rate)
    write_tts_review_log(review_path, reviews)
    total = sum(len(group.segments) for group in groups)
    log_tts_summary(total, reviews, review_path)

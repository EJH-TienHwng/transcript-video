from __future__ import annotations

import json
import logging
import re
import tempfile
import textwrap
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from numbers import Integral
from pathlib import Path

from ...config import SubtitleSegment
from ...hardware import get_ffmpeg_exe, resolve_torch_device
from ...process_runner import run_ffmpeg
from ..media import get_media_duration_seconds

logger = logging.getLogger(__name__)

MIN_GAP_SECONDS = 0.08
MIN_AVAILABLE_SECONDS = 0.001
ALIGNMENT_MIN_CONFIDENCE = 0.6
ALIGNMENT_PADDING_SECONDS = 0.04


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


def load_faster_whisper_aligner(model_name: str | Path):
    """Use CPU alignment so the aligner does not compete with Qwen for VRAM."""
    try:
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper aligner on CPU: %s", model_name)
        return WhisperModel(str(model_name), device="cpu", compute_type="int8")
    except Exception as exc:
        logger.warning("Could not load faster-whisper alignment; using reviewed fallback: %s", exc)
        return None


def generate_qwen_custom_voice(
    model, text: str, language: str, speaker: str, instruct: str
) -> tuple[object, int]:
    """Generate one waveform with Qwen CustomVoice."""
    wavs, sr = model.generate_custom_voice(
        text=text, language=language, speaker=speaker, instruct=instruct
    )
    if wavs is None or len(wavs) == 0:
        raise ValueError("Qwen TTS returned no waveform.")
    if not isinstance(sr, Integral) or sr <= 0:
        raise ValueError(f"Invalid TTS sample rate: {sr}")
    return wavs[0], int(sr)


def _normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    return re.findall(r"\w+(?:'\w+)*", normalized, flags=re.UNICODE)


def align_context_group(
    group: TTSContextGroup,
    words: list[WordTiming],
    audio_duration: float,
    min_confidence: float = ALIGNMENT_MIN_CONFIDENCE,
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
    for block in SequenceMatcher(None, expected, recognized, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            expected_position = block.a + offset
            matched.setdefault(owners[expected_position], []).append(
                (expected_position, valid_words[block.b + offset])
            )

    aligned: list[AlignedTTSSegment] = []
    failures: dict[int, str] = {}
    previous_end = 0.0
    for subtitle_index, segment in group.segments:
        matches = matched.get(subtitle_index, [])
        confidence = len(matches) / max(1, len(_normalized_words(segment.text)))
        if not matches:
            failures[subtitle_index] = "alignment_failed"
            continue
        if confidence < min_confidence or (
            owner_positions[subtitle_index] != (matches[0][0], matches[-1][0])
        ):
            failures[subtitle_index] = "alignment_low_confidence"
            continue
        source_start = max(previous_end, matches[0][1].start - ALIGNMENT_PADDING_SECONDS)
        source_end = min(audio_duration, matches[-1][1].end + ALIGNMENT_PADDING_SECONDS)
        if source_end <= source_start:
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


def proportional_alignment_fallback(
    group: TTSContextGroup, audio_duration: float
) -> list[AlignedTTSSegment]:
    """Last-resort split used only with an explicit review entry."""
    weights = [max(1, len(_normalized_words(segment.text))) for _, segment in group.segments]
    total = sum(weights)
    cursor = 0.0
    aligned: list[AlignedTTSSegment] = []
    for (subtitle_index, segment), weight in zip(group.segments, weights, strict=True):
        end = (
            audio_duration
            if len(aligned) + 1 == len(weights)
            else cursor + audio_duration * weight / total
        )
        aligned.append(AlignedTTSSegment(subtitle_index, segment, cursor, end, 0.0))
        cursor = end
    return aligned


def _pitch_preserving_speedup(wav, sample_rate: int, speed_factor: float):
    """Change tempo with the already-required FFmpeg without shifting pitch."""
    import numpy as np
    import soundfile as sf

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
    if duration <= available_duration or len(wav) < 2:
        return wav
    return _pitch_preserving_speedup(
        wav, sample_rate, min(duration / available_duration, max_speedup)
    )


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
) -> dict[str, object]:
    return {
        "subtitle_index": subtitle_index,
        "text": segment.text,
        "start": segment.start,
        "end": segment.end,
        "next_start": next_start,
        "available_duration": round(available_duration, 6),
        "generated_speech_duration": round(generated_duration, 6),
        "required_speedup": round(generated_duration / available_duration, 6),
        "max_speedup": max_speedup,
        "context_group_index": group_index,
        "alignment_source_start": source_start,
        "alignment_source_end": source_end,
        "action": action,
        "review_reason": reason,
    }


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
) -> tuple[list[tuple[int, SubtitleSegment, object]], int | None, list[dict[str, object]]]:
    """Generate, align, extract, and duration-fit context-group sentences."""
    import numpy as np

    ordered = sorted(
        ((index, segment) for index, segment in enumerate(all_segments, 1) if segment.text.strip()),
        key=lambda item: item[1].start,
    )
    next_starts = {
        index: ordered[position + 1][1].start if position + 1 < len(ordered) else None
        for position, (index, _) in enumerate(ordered)
    }
    generated: list[tuple[int, SubtitleSegment, object]] = []
    reviews: list[dict[str, object]] = []
    sample_rate: int | None = None

    for group in groups:
        try:
            group_wav, sr = generate_qwen_custom_voice(
                model, group.text, language, speaker, instruct
            )
            group_wav = np.asarray(group_wav, dtype=np.float32).reshape(-1)
            if sample_rate is not None and sr != sample_rate:
                raise ValueError(f"Inconsistent sample rate: {sr} != {sample_rate}")
            sample_rate = sr
        except Exception as exc:
            logger.warning("TTS context group %d failed: %s", group.index, exc)
            for subtitle_index, segment in group.segments:
                next_start = next_starts[subtitle_index]
                available = max(
                    MIN_AVAILABLE_SECONDS,
                    (next_start - MIN_GAP_SECONDS if next_start is not None else segment.end + 1.0)
                    - segment.start,
                )
                reviews.append(
                    _review_entry(
                        subtitle_index=subtitle_index,
                        segment=segment,
                        next_start=next_start,
                        available_duration=available,
                        generated_duration=0.0,
                        max_speedup=max_speedup,
                        group_index=group.index,
                        source_start=None,
                        source_end=None,
                        action="skipped_audio",
                        reason="unexpected_generation_error",
                    )
                )
            continue

        audio_duration = len(group_wav) / sr
        aligned: list[AlignedTTSSegment] = []
        alignment_reason: dict[int, str]
        if aligner is not None:
            try:
                words = transcribe_word_timings(aligner, group_wav, sr, group.text, language)
                aligned, alignment_reason = align_context_group(group, words, audio_duration)
            except Exception as exc:
                logger.warning("Alignment failed for context group %d: %s", group.index, exc)
                alignment_reason = {index: "alignment_failed" for index, _ in group.segments}
        else:
            alignment_reason = {index: "alignment_failed" for index, _ in group.segments}

        aligned_by_index = {item.subtitle_index: item for item in aligned}
        fallback_by_index = {
            item.subtitle_index: item
            for item in proportional_alignment_fallback(group, audio_duration)
        }
        for subtitle_index, segment in group.segments:
            item = aligned_by_index.get(subtitle_index)
            reason = alignment_reason.get(subtitle_index)
            if item is None:
                item = fallback_by_index[subtitle_index]
                reason = reason or "alignment_failed"
            start_sample = max(0, round(item.source_start * sr))
            end_sample = min(len(group_wav), round(item.source_end * sr))
            if end_sample <= start_sample:
                item = fallback_by_index[subtitle_index]
                start_sample = max(0, round(item.source_start * sr))
                end_sample = min(len(group_wav), round(item.source_end * sr))
                reason = "invalid_audio_range"
            sentence_wav = group_wav[start_sample:end_sample].copy()

            next_start = next_starts[subtitle_index]
            slot_end = (
                next_start - MIN_GAP_SECONDS
                if next_start is not None
                else max(segment.end + 1.0, video_duration or 0.0)
            )
            available = max(MIN_AVAILABLE_SECONDS, slot_end - segment.start)
            generated_duration = len(sentence_wav) / sr
            if reason:
                reviews.append(
                    _review_entry(
                        subtitle_index=subtitle_index,
                        segment=segment,
                        next_start=next_start,
                        available_duration=available,
                        generated_duration=generated_duration,
                        max_speedup=max_speedup,
                        group_index=group.index,
                        source_start=item.source_start,
                        source_end=item.source_end,
                        action="used_proportional_fallback",
                        reason=reason,
                    )
                )
            if generated_duration > available:
                try:
                    sentence_wav = fit_wav_to_available_duration(
                        sentence_wav, sr, available, max_speedup
                    )
                except Exception as exc:
                    logger.warning("Could not time-stretch subtitle %d: %s", subtitle_index, exc)
                    reviews.append(
                        _review_entry(
                            subtitle_index=subtitle_index,
                            segment=segment,
                            next_start=next_start,
                            available_duration=available,
                            generated_duration=generated_duration,
                            max_speedup=max_speedup,
                            group_index=group.index,
                            source_start=item.source_start,
                            source_end=item.source_end,
                            action="kept_unmodified_audio",
                            reason="unexpected_generation_error",
                        )
                    )
                if len(sentence_wav) / sr > available + 1 / sr:
                    reviews.append(
                        _review_entry(
                            subtitle_index=subtitle_index,
                            segment=segment,
                            next_start=next_start,
                            available_duration=available,
                            generated_duration=generated_duration,
                            max_speedup=max_speedup,
                            group_index=group.index,
                            source_start=item.source_start,
                            source_end=item.source_end,
                            action="kept_overflow_after_pitch_preserving_speedup",
                            reason="exceeds_max_speedup",
                        )
                    )
            generated.append((subtitle_index, segment, sentence_wav))
    return generated, sample_rate, reviews


def overlay_tts_items(
    items: list[tuple[int, SubtitleSegment, object]],
    sample_rate: int,
    minimum_duration: float,
    offset: float = 0.0,
):
    """Place extracted sentences at original SRT starts, never sequentially."""
    import numpy as np

    required_samples = max(
        [round(minimum_duration * sample_rate), 1]
        + [round((segment.start - offset) * sample_rate) + len(wav) for _, segment, wav in items]
    )
    audio = np.zeros(required_samples, dtype=np.float32)
    for _, segment, wav in items:
        start = max(0, round((segment.start - offset) * sample_rate))
        end = start + len(wav)
        if end > len(audio):
            audio = np.pad(audio, (0, end - len(audio)))
        audio[start:end] += wav
    return np.clip(audio, -1.0, 1.0)


def write_tts_review_log(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def log_tts_summary(total: int, aligned: int, reviews: int, review_path: Path) -> None:
    logger.info(
        "TTS summary: %d sentence(s), %d aligned, %d marked for review; review log: %s",
        total,
        aligned,
        reviews,
        review_path,
    )


def tts_review_counts(total: int, entries: list[dict[str, object]]) -> tuple[int, int]:
    reviewed = {entry["subtitle_index"] for entry in entries}
    not_aligned = {
        entry["subtitle_index"]
        for entry in entries
        if entry["review_reason"]
        in {"alignment_failed", "alignment_low_confidence", "invalid_audio_range"}
        or entry["action"] == "skipped_audio"
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
    """Generate contextual TTS, then restore each sentence to its SRT start."""
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
    if sample_rate is None:
        raise ValueError("Qwen TTS did not generate any audio.")
    minimum_duration = max(video_duration, max(segment.end for segment in segments) + 1.0)
    audio = overlay_tts_items(items, sample_rate, minimum_duration)
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_out), audio, sample_rate)
    review_path = review_log_path or audio_out.with_name(f"{audio_out.stem}_review.jsonl")
    write_tts_review_log(review_path, reviews)
    total = sum(len(group.segments) for group in groups)
    aligned, review_count = tts_review_counts(total, reviews)
    log_tts_summary(total, aligned, review_count, review_path)

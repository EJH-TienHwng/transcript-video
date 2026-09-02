from __future__ import annotations

import logging
import re
import textwrap
from numbers import Integral
from pathlib import Path

from ...config import SubtitleSegment
from ...hardware import resolve_torch_device
from ..media import get_media_duration_seconds


def split_text_for_tts(text: str, max_chars: int = 450) -> list[str]:
    """Split long text into smaller chunks for TTS generation."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentence_parts = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentence_parts:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(
                textwrap.wrap(
                    sentence,
                    width=max_chars,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


def load_qwen_tts_model(
    tts_model_name: str,
    device: str,
    attn_implementation: str,
):
    """Load Qwen TTS model lazily so the script can still run without TTS."""
    import torch
    from qwen_tts import Qwen3TTSModel

    device = resolve_torch_device(device, "Qwen TTS")

    kwargs = {
        "device_map": "cuda:0" if device == "cuda" else "cpu",
        "dtype": torch.float16 if device == "cuda" else torch.float32,
    }

    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation

    logging.info("Loading Qwen TTS model: %s", tts_model_name)
    return Qwen3TTSModel.from_pretrained(tts_model_name, **kwargs)


def generate_qwen_custom_voice(
    model,
    text: str,
    language: str,
    speaker: str,
    instruct: str,
) -> tuple[object, int]:
    """Generate one waveform with Qwen CustomVoice."""
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )
    if wavs is None or len(wavs) == 0:
        raise ValueError("Qwen TTS returned no waveform.")
    if not isinstance(sr, Integral) or sr <= 0:
        raise ValueError(f"Invalid TTS sample rate: {sr}")
    return wavs[0], int(sr)


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
    """
    Generate a simple continuous audio file from all subtitle text.

    This is easy to generate, but it is not timestamp-aligned.
    For most video dubbing use-cases, synthesize_timed_tts_audio is better.
    """
    import numpy as np
    import soundfile as sf

    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    chunks = split_text_for_tts(text)
    if not chunks:
        raise ValueError("No text is available for TTS generation.")

    model = load_qwen_tts_model(tts_model_name, device, attn_implementation)

    wav_list = []
    sample_rate = None
    for index, chunk in enumerate(chunks, start=1):
        logging.info("TTS chunk %d/%d", index, len(chunks))
        wav, sr = generate_qwen_custom_voice(
            model=model,
            text=chunk,
            language=tts_language,
            speaker=tts_speaker,
            instruct=tts_instruct,
        )
        wav = np.asarray(wav, dtype=np.float32)
        wav_list.append(wav)
        sample_rate = sr if sample_rate is None else sample_rate
        if sr != sample_rate:
            raise ValueError(f"Inconsistent sample rate: {sr} != {sample_rate}")

    full_audio = np.concatenate(wav_list)
    full_audio = np.clip(full_audio, -1.0, 1.0)
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_out), full_audio, sample_rate)


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
) -> None:
    """
    Generate one WAV file by placing every TTS segment at its subtitle timestamp.

    Important fix:
    - Each TTS segment is only allowed to play until before the next subtitle starts.
    - If the generated TTS is too long, it is sped up.
    - If it is still too long, it is trimmed with a small fade-out.
    """
    import numpy as np
    import soundfile as sf

    # ===== Timing control =====
    MIN_GAP_SECONDS = 0.08  # small silence before next voice starts
    MAX_SPEEDUP = 1.35  # do not speed up too much, or voice becomes unnatural
    MIN_SLOT_SECONDS = 0.35  # minimum allowed slot
    FADE_OUT_SECONDS = 0.04  # fade out when trimming

    def speedup_wav_by_resample(wav: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        Simple in-memory speed-up.
        Note: this changes pitch slightly, but avoids overlap without extra dependencies.
        """
        if speed_factor <= 1.0 or len(wav) < 2:
            return wav

        new_length = max(1, int(len(wav) / speed_factor))
        old_positions = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)

        return np.interp(new_positions, old_positions, wav).astype(np.float32)

    def trim_with_fade_out(wav: np.ndarray, max_samples: int, sample_rate: int) -> np.ndarray:
        """Trim wav to max_samples and apply a small fade-out to avoid clicking."""
        if max_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        if len(wav) <= max_samples:
            return wav

        wav = wav[:max_samples].copy()
        fade_samples = min(len(wav), int(FADE_OUT_SECONDS * sample_rate))

        if fade_samples > 0:
            wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

        return wav

    valid_segments = [segment for segment in segments if segment.text.strip()]
    if not valid_segments:
        raise ValueError("No subtitle segments are available for TTS generation.")

    model = load_qwen_tts_model(tts_model_name, device, attn_implementation)

    generated_chunks = []
    sample_rate = None
    video_duration = get_media_duration_seconds(video_path)

    for index, segment in enumerate(valid_segments):
        text = segment.text.strip()
        logging.info("TTS segment %d/%d: %s", index + 1, len(valid_segments), text[:80])

        wav, sr = generate_qwen_custom_voice(
            model=model,
            text=text,
            language=tts_language,
            speaker=tts_speaker,
            instruct=tts_instruct,
        )

        wav = np.asarray(wav, dtype=np.float32)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Inconsistent sample rate: {sr} != {sample_rate}")

        # Slot end = before next subtitle starts.
        if index + 1 < len(valid_segments):
            next_start = valid_segments[index + 1].start
            slot_end = next_start - MIN_GAP_SECONDS
        else:
            # Last segment can run until video end, or at least its own subtitle end.
            slot_end = video_duration if video_duration else segment.end + 1.0

        available_duration = max(MIN_SLOT_SECONDS, slot_end - segment.start)
        available_samples = int(available_duration * sample_rate)

        original_duration = len(wav) / sample_rate

        if original_duration > available_duration:
            needed_speedup = original_duration / available_duration
            speedup = min(needed_speedup, MAX_SPEEDUP)

            logging.warning(
                "TTS segment %d too long: %.2fs > %.2fs. Speedup %.2fx.",
                index + 1,
                original_duration,
                available_duration,
                speedup,
            )

            wav = speedup_wav_by_resample(wav, speedup)

            if len(wav) > available_samples:
                logging.warning(
                    "TTS segment %d still too long after speedup. Trim to %.2fs.",
                    index + 1,
                    available_duration,
                )
                wav = trim_with_fade_out(wav, available_samples, sample_rate)

        generated_chunks.append((segment, wav))

    subtitle_duration = max(segment.end for segment in valid_segments) + 1.0
    total_duration = max(video_duration or 0.0, subtitle_duration)
    total_samples = int(total_duration * sample_rate) + sample_rate
    full_audio = np.zeros(total_samples, dtype=np.float32)

    for segment, wav in generated_chunks:
        start_sample = max(0, int(segment.start * sample_rate))
        end_sample = start_sample + len(wav)

        if start_sample >= len(full_audio):
            continue

        if end_sample > len(full_audio):
            wav = wav[: len(full_audio) - start_sample]
            end_sample = len(full_audio)

        # No overlap now because each wav was already fitted to its slot.
        full_audio[start_sample:end_sample] += wav

    full_audio = np.clip(full_audio, -1.0, 1.0)
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_out), full_audio, sample_rate)


def fit_wav_to_available_duration(
    wav,
    sample_rate: int,
    available_duration: float,
    max_speedup: float = 1.15,
    fade_out_seconds: float = 0.04,
):
    """Fit a generated TTS waveform into an available time slot.

    If the waveform is longer than the slot, it is sped up slightly by
    resampling. If it is still too long, it is trimmed with a small fade-out.
    """
    import numpy as np

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero.")
    if available_duration <= 0:
        raise ValueError("available_duration must be greater than zero.")
    if max_speedup < 1.0:
        raise ValueError("max_speedup must be at least 1.0.")

    wav = np.asarray(wav, dtype=np.float32)
    available_samples = max(1, int(available_duration * sample_rate))

    if len(wav) <= available_samples:
        return wav

    original_duration = len(wav) / sample_rate
    needed_speedup = original_duration / max(available_duration, 1e-6)
    speedup = min(max(1.0, needed_speedup), max_speedup)

    if speedup > 1.0 and len(wav) > 1:
        new_length = max(1, int(len(wav) / speedup))
        old_positions = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        wav = np.interp(new_positions, old_positions, wav).astype(np.float32)

    if len(wav) > available_samples:
        wav = wav[:available_samples].copy()
        fade_samples = min(len(wav), int(fade_out_seconds * sample_rate))
        if fade_samples > 0:
            wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    return wav

# Quality implementation and validation

The existing CLI/Wizard/TUI → application → processing/course architecture is preserved.
TTS review schema is now v4; cache provenance is v1; semantic events remain v1.
No model inference or manual listening acceptance is implied by mocked/unit test results.

## Local environment, 2026-09-07

- Python 3.14.7; project pin 3.14 and packaging `>=3.14,<3.15`.
- Torch 2.14.0+cu132, CUDA 13.2 local wheel; torchaudio 2.11.0 local wheel.
- qwen-tts 0.1.1 and Transformers 4.57.3 installed.
- GTX 1650 Ti, compute capability 7.5; `flash_attn` absent.
- No `models/`, `data/input/`, or local MP4/WAV test material found in `data/`.
- FFmpeg 7.1 (imageio bundled) is available, with working NVENC and libass probes. `ffprobe`
  is missing from PATH and beside FFmpeg; Doctor correctly exits 1 for this and missing models.
- The installed Qwen import prints third-party optional FlashAttention/SoX warnings. The
  project's speech-duration adjustment still uses FFmpeg atempo and does not require SoX.

The [official FA2 implementation](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)
lists Ampere/Ada/Hopper support for its CUDA backend and treats Windows compilation as less
tested. Its PyTorch minimum alone does not establish compatibility with this custom
Python/Torch/CUDA wheel combination. This host's 7.5 GPU cannot validate that backend.
No package installation or real TTS FA2 benchmark was attempted; SDPA remains the preset.

The [Qwen project](https://github.com/QwenLM/Qwen3-TTS) exposes the attention implementation
through model loading. The installed 0.1.1 source was inspected: its speech tokenizer is a
separate wrapper, so the runtime explicitly parks/resumes codec weights as well as the talker.
Whisper uses the installed CTranslate2 `unload_model(to_cpu=True)` / `load_model()` API,
also documented in the [upstream Whisper binding](https://github.com/OpenNMT/CTranslate2/blob/master/python/cpp/whisper.cc).
Real offload/resume throughput and host-RAM use still need measurement with model weights.

## Acceptance limits

The final verifier reads persisted PCM WAV ranges after fitting and assembly, using CPU
faster-whisper with no expected-text prompt. Alignment's existing prompted recognition remains
separate. Text agreement is evidence of fidelity only, not pronunciation/prosody quality.
SequenceMatcher reports an explicit edit ratio, not minimum-edit WER. Tail RMS is a review
heuristic, especially uncertain for unpadded fricatives. There are no synthetic voice scores.

Listen to all flagged sentences, including the preceding/following sentence, on a representative
short and long source video. Verify technical terms, final syllables, timbre/register, pauses,
speed-adjusted, shifted and overflowing sentences. This work remains unchecked in `todo.md`.

Media integration tests generate synthetic audio/video and exercise real FFmpeg, atempo,
tail-marker preservation and muxing beyond the source's duration. These do not substitute for
Qwen generation, real Whisper recognition or human listening.

Partial output tests cover exceptions, KeyboardInterrupt, failed FFmpeg publication and stale
WAV manifests. Hard-kill orphans use recognizable names and are ignored, rather than deleting
files that might belong to another live writer. Atomic replacement covers individual files;
report/WAV/manifest are not a multi-file transaction. Version/range checks and WAV identity
prevent incomplete or mismatched generations from authorizing reuse.

## Checks executed

- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed after formatting two mixed-line-ending edits.
- `uv run pytest`: 278 passed on Windows, including synthetic real-FFmpeg tests and the
  installed Qwen/Transformers token-flow check without model weights.
- `uv build`: source distribution and wheel built successfully.
- Both shipped profiles pass `config validate`; the Python consistency script passes.
- Doctor: expected exit 1 for missing ffprobe/ASR/Qwen models; FA2 reports optional WARN.
- No real Qwen/Whisper inference, representative-media listening, FA2 performance benchmark,
  or Linux execution was performed. The CI Windows/Linux matrix remains configured.

## Files changed in this implementation

The staged work present when the task started was preserved. This table describes the
additional implementation, not all changes already staged on `dev`.

| File(s) | Purpose |
| --- | --- |
| `src/transcript_video/processing/tts/qa.py` | Text comparison, final PCM verification, acoustic heuristic, duration metrics. |
| `src/transcript_video/processing/tts/core.py` | Review v4, speed observability, final verifier, atomic WAV/reports, runtime/FA2 hooks. |
| `src/transcript_video/processing/tts/chunks.py` | Per-chunk provenance, preserved selective reruns, atomic rebuild, final verification. |
| `src/transcript_video/processing/provenance.py` | Versioned source/config/model manifests and semantic invalidation. |
| `src/transcript_video/processing/runtime.py` | Scoped backend reuse, GPU parking/resume and cleanup. |
| `src/transcript_video/processing/transcription.py` | Reusable Whisper, HF ASR and translation loaders. |
| `src/transcript_video/processing/pipeline.py` | Provenance decisions, published SRT timestamps, style/QA routing, duration reports. |
| `src/transcript_video/artifacts.py` | Atomic writers and recognizable partial-file naming. |
| `src/transcript_video/processing/subtitles.py` | Atomic SRT output. |
| `src/transcript_video/processing/media.py` | Atomic FFmpeg publication, subtitle-style serializer, partial discovery exclusion. |
| `src/transcript_video/config.py` | Validated subtitle style and optional final-verifier setting. |
| `src/transcript_video/application/settings.py` | Shared validation for new settings. |
| `src/transcript_video/application/processing.py` | Runtime lifecycle, planned duration report, reject partial inputs. |
| `src/transcript_video/application/diagnostics.py` | Optional FA2 Doctor status. |
| `src/transcript_video/hardware.py` | FA2 capability/import/kernel probe. |
| `src/transcript_video/cli.py` | Final-verification CLI override. |
| `src/transcript_video/course/builder.py`, `course/media.py` | Atomic course MP4/copy publication. |
| `src/transcript_video/course/config.py` | Reject incomplete session artifacts. |
| `src/transcript_video/course/wizard.py` | Compact table/summary hierarchy; ignore partial media. |
| `src/transcript_video/tui/app.py` | Responsive session layout, focus borders and headings. |
| `src/transcript_video/ui/progress.py` | Display `speed_adjusted`. |
| `configs/transcription.toml`, `configs/profiles/srt.toml`, `configs/profiles/tts-review.toml` | Verifier default and two workflow presets. |
| `scripts/check_python_version.py`, `.github/workflows/quality.yml` | Canonical-pin consistency script and CI invocation. |
| `tests/test_artifacts.py`, `test_attention.py`, `test_cache.py`, `test_profiles_version.py`, `test_quality.py`, `test_runtime.py`, `test_style.py` | New focused regression coverage. |
| `tests/test_observability.py`, `test_qol.py`, `test_tts.py`, `test_tui.py` | Updated schema/provenance fixtures and layout checks. |
| `README.md`, `docs/README.md`, `docs/README.vi.md`, `docs/observability.md`, `docs/todo.md`, this report | Behavior, migration, validation evidence and unchecked acceptance work. |

# Qwen generation PAD: implementation and validation

Inspected on branch `dev`, using the installed `qwen-tts==0.1.1` and
`transformers==4.57.3` in `.venv/Lib/site-packages`.

## Root cause and actual call chain

```text
generate_qwen_custom_voice()
  → Qwen3TTSModel.generate_custom_voice(**kwargs)
  → Qwen3TTSModel._merge_generate_kwargs(**kwargs)
  → Qwen3TTSForConditionalGeneration.generate(**gen_kwargs)
  → Qwen3TTSTalkerForConditionalGeneration.generate(**talker_kwargs)
  → Transformers GenerationMixin._prepare_special_tokens()
```

The installed wrapper forwards arbitrary kwargs, including `pad_token_id`, to
the outer conditional-generation model. However, that model builds its own
`talker_kwargs` dictionary and **drops PAD**. Setting only the wrapper kwarg
does not prevent the warning in this package version.

The relevant PAD lives at `model.model.talker.generation_config.pad_token_id`.
Qwen explicitly passes `model.model.config.talker_config.codec_eos_token_id`
as the talker's generation EOS when the caller does not override EOS. Thus
`talker.generation_config.eos_token_id` is not the effective EOS for this path.
The wrapper's `generate_defaults` dictionary only supplies named sampling
defaults; it does not supply PAD/EOS here.

When PAD is missing, Transformers selects the first effective EOS and emits
the reported warning. It prepares a local config copy, so implicit fallback
does not persist across calls. Qwen supplies an attention mask already.

Source evidence, relative to `.venv/Lib/site-packages`:

| Source | Relevant locations |
| --- | --- |
| `qwen_tts/inference/qwen3_tts_model.py` | `from_pretrained`: 112–123; `_merge_generate_kwargs`: 287–352; CustomVoice forwarding: 827–837 |
| `qwen_tts/core/models/modeling_qwen3_tts.py` | talker construction: 1820; `talker_kwargs`: 2044–2067; actual talker call: 2272 |
| `transformers/generation/utils.py` | `_prepare_special_tokens`: 2054; first-EOS fallback and warning: 2094–2102 |

## Implementation and files

- `src/transcript_video/processing/tts/core.py`: small typed PAD resolver,
  initialization when loading Qwen, and explicit PAD at the shared call.
- `tests/test_tts.py`: resolver, loader, waveform, routing and installed-package
  regression checks.
- This document: source evidence and validation limits.

Resolution prefers the existing talker generation PAD, including zero. Only
when it is absent does it use Qwen's effective codec-vocabulary EOS. A valid
list/tuple of EOS IDs supplies its first element, matching Transformers.
Booleans, negative/non-integer IDs, empty or malformed sequences and arbitrary
objects return `None`; malformed dedicated PAD is not replaced silently.
Missing generation config also returns `None`. The caller then omits the PAD
kwarg, retaining the library's existing handling.

This deliberately does not consult `tts_pad_token_id`, `codec_pad_id`, outer
generation PAD or the talker's overridden EOS. No production token ID is
hard-coded.

Because Qwen 0.1.1 drops PAD before the actual Transformers call, the loader
initializes **only a missing PAD on the loaded talker's generation config**.
This is the required exception to avoiding config mutation. It affects that
model instance, not process-wide defaults. Existing PAD is preserved. The
shared generator additionally passes the resolved PAD explicitly to the
wrapper, as requested. No dependency patch, logging suppression, TOML setting,
sampling change or per-subtitle token log was introduced.

Before: missing talker PAD → Transformers infers PAD from EOS → warning on
every generation.

After: load model → resolve and initialize missing talker PAD → shared call
passes explicit PAD → Transformers finds PAD already configured.

All project TTS entry points use the shared loader/generator: context groups,
individual fallback/retries, full timed generation, time chunks/reruns and
simple full-text generation. The other production `.generate()` call belongs
to translation and is unrelated. Wrappers loaded outside the project's loader
still encounter Qwen 0.1.1's dropped-kwarg limitation.

## Tests and validation

Tests cover existing PAD, effective EOS precedence, absent config/tokens,
multi-EOS lists/tuples, zero, NumPy integral conversion, invalid values,
explicit kwargs/omission and initialization without changing other config.
Waveform checks retain float32 conversion, flattening, sample-rate validation,
missing/empty/silent audio and NaN/Inf rejection. Route checks exercise context,
individual retries, simple/full timed output, chunk generation and reruns.

The installed-package regression uses the real Qwen wrapper, real Qwen outer
generation method and real Transformers special-token preparation. Tiny
synthetic embeddings replace weights; execution stops before talker inference
and speech decoding. It reproduces the warning before loader initialization,
then verifies three subsequent calls produce the same effective PAD without
that warning. A test-only log capture handler observes warnings without
suppressing them. This test is marked `integration` and skips if the optional
heavy runtime is absent from a lightweight test environment.

Final validation:

| Command | Result |
| --- | --- |
| `uv run ruff format .` | Pass; final run required no changes |
| `uv run ruff check .` | All checks passed |
| `uv run pytest tests/test_tts.py -v` | 88 passed, 17.09 seconds |
| `uv run pytest -v` | 229 passed, 43.11 seconds |
| `git diff --check` | Pass |

Self-review found no hard-coded `2150` in production and no added warning
filters or verbosity changes. Existing subtitle timestamp, chunk-cache and
pitch/tail-preservation regressions pass. Full speech synthesis remains
unverified for the reason below.

## Runtime token values and manual TTS limits

The synthetic runtime fixture verifies PAD `None`, effective EOS `2150`, then
resolved PAD `2150`; its talker generation-config EOS is deliberately `999`
to prove Qwen's effective EOS takes precedence. These are **fixture values**.

The current checkpoint's actual values have not been verified:
`models/Qwen3-TTS-12Hz-1.7B-CustomVoice` and `test.mp4` are absent locally.
No weights were downloaded. Therefore the real speech command below could
not be validated, and audio quality/output equivalence on that checkpoint is
not claimed:

```powershell
uv run transcript-video -v process test.mp4 --enable-tts
```

Once the checkpoint/input are available, this remains the final manual check
for audible speech and absence of the fallback warning. The model-aware fix
uses the loaded checkpoint's values rather than assuming the warning's `2150`.

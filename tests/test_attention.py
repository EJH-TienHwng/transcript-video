import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from transcript_video.hardware import flash_attention_status
from transcript_video.processing.tts import core


def test_unsupported_gpu_falls_back_without_importing_flash(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda _: (7, 5))
        ),
    )
    assert flash_attention_status("cuda")[0] is False
    assert flash_attention_status("cpu")[0] is False


@pytest.mark.parametrize(
    "supported,fail_initialization", [(False, False), (True, True), (True, False)]
)
def test_qwen_optional_attention_fallback(monkeypatch, supported, fail_initialization):
    model = SimpleNamespace(model=SimpleNamespace())
    factory = Mock(
        side_effect=[ImportError("flash attention ABI mismatch"), model]
        if fail_initialization
        else None,
        return_value=model,
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=factory)),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16", float32="float32"))
    monkeypatch.setattr(core, "resolve_torch_device", lambda *a: "cuda")
    monkeypatch.setattr(
        core, "flash_attention_status", lambda *a: (supported, "optional backend unavailable")
    )
    assert core.load_qwen_tts_model("local", "cuda", "flash_attention_2") is model
    assert factory.call_args.kwargs["attn_implementation"] == (
        "flash_attention_2" if supported and not fail_initialization else "sdpa"
    )

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from transcript_video.processing.runtime import model_runtime, reuse_model


def test_three_video_batch_loads_whisper_once_and_cleans_scope(tmp_path, monkeypatch):
    from transcript_video.application.processing import build_process_plan, execute_process_plan
    from transcript_video.config import RunSettings
    from transcript_video.processing import transcription

    model = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        transcribe=lambda *a, **kw: ([SimpleNamespace(start=0, end=1, text="one")], None),
    )
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    monkeypatch.setattr(transcription, "resolve_torch_device", lambda *a: "cpu")
    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)
    settings.project.model = "model"
    settings.hardware.device = "cpu"
    settings.transcription.skip_burn = True
    (tmp_path / "model").mkdir()
    (tmp_path / "model/model.bin").touch()
    videos = [tmp_path / f"{i}.mp4" for i in range(3)]
    for video in videos:
        video.touch()
    result = execute_process_plan(build_process_plan(settings, videos))
    assert result.succeeded == 3, result.failures
    assert factory.call_count == 1
    transcription.load_whisper(tmp_path / "model", "cpu", "int8")
    assert factory.call_count == 2  # Invocation scope cannot leak into the next caller.


def test_gpu_models_and_codec_are_parked_before_loading_next_backend(monkeypatch):
    monkeypatch.setattr("transcript_video.processing.runtime._empty_cuda_cache", lambda: None)
    whisper = SimpleNamespace(model=Mock(device="cuda"))
    translation = (object(), Mock(device="cuda:0"))
    qwen = SimpleNamespace(device="cuda:0", model=Mock())
    aligner = SimpleNamespace(model=SimpleNamespace(device="cpu"))
    factories = {
        name: Mock(return_value=value)
        for name, value in dict(
            whisper=whisper, translation=translation, qwen=qwen, aligner=aligner
        ).items()
    }
    loaders = {name: reuse_model(name)(factory) for name, factory in factories.items()}
    with model_runtime() as runtime:
        for _ in range(3):
            loaders["whisper"]()
            loaders["translation"]()
            assert whisper.model.unload_model.call_args.kwargs == {"to_cpu": True}
            loaders["qwen"]()
            assert translation[1].to.call_args.args == ("cpu",)
            loaders["aligner"]()
        assert all(factory.call_count == 1 for factory in factories.values())
        assert whisper.model.load_model.call_count == 2
        assert qwen.model.to.call_count == qwen.model.speech_tokenizer.model.to.call_count == 4
    assert runtime.models == {} and runtime.active is None


def test_runtime_cleanup_after_error_and_configuration_change(monkeypatch):
    monkeypatch.setattr("transcript_video.processing.runtime._empty_cuda_cache", lambda: None)
    factory = Mock(return_value=SimpleNamespace(model=SimpleNamespace(device="cpu")))
    loader = reuse_model("whisper")(factory)
    with pytest.raises(RuntimeError), model_runtime() as runtime:
        loader("A")
        loader("A")
        loader("B")
        assert factory.call_count == 2
        raise RuntimeError("video failed")
    assert not runtime.models

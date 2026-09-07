from pathlib import Path

import pytest

from transcript_video.artifacts import atomic_output
from transcript_video.processing import media


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_atomic_failure_preserves_previous_and_cleans_partial(tmp_path, error):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"previous")
    with pytest.raises(error), atomic_output(output) as temporary:
        temporary.write_bytes(b"partial")
        raise error()
    assert output.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [output]
    with atomic_output(output) as temporary:
        temporary.write_bytes(b"complete")
    assert output.read_bytes() == b"complete"


def test_ffmpeg_publish_only_after_success(tmp_path, monkeypatch):
    output = tmp_path / "render.mp4"
    output.write_bytes(b"old")

    def fail(command):
        assert Path(command[-1]) != output and str(command[-1]).endswith(".mp4")
        Path(command[-1]).write_bytes(b"partial")
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(media, "run_ffmpeg", fail)
    with pytest.raises(RuntimeError):
        media.run_command(["ffmpeg", "-y", str(output)])
    assert output.read_bytes() == b"old" and len(list(tmp_path.iterdir())) == 1


def test_replaced_wav_cannot_use_previous_manifest(tmp_path):
    from transcript_video.events import PipelineStage
    from transcript_video.processing.provenance import cache_matches, write_provenance

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"old")
    write_provenance(audio, {"speaker": "Aiden"})
    with atomic_output(audio) as temporary:
        temporary.write_bytes(b"new voice")
    assert not cache_matches(audio, {"speaker": "Aiden"}, PipelineStage.TTS)


def test_orphan_partial_files_are_never_discovered_as_completed_media(tmp_path):
    from transcript_video.course.wizard import _scan_videos

    complete, partial = tmp_path / "a.mp4", tmp_path / ".a.dead.partial.mp4"
    complete.touch()
    partial.touch()
    assert media.find_videos(tmp_path) == [complete]
    assert _scan_videos(tmp_path) == [complete]
    with pytest.raises(ValueError, match="Incomplete artifact"):
        media.find_videos(tmp_path, partial.name)

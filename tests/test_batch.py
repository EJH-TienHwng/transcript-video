from copy import deepcopy
from pathlib import Path

import pytest

from transcript_video.application.batch import BatchSettings, VideoOverride
from transcript_video.events import EventKind, PipelineStage, RecordingObserver


def test_batch_settings_resolve_isolated_profiles_and_overrides(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text("", encoding="utf-8")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "srt.toml").write_text("[tts]\nenabled = false\n", encoding="utf-8")
    (profiles / "voice.toml").write_text(
        "[tts]\nenabled = true\nspeaker = 'Ryan'\n", encoding="utf-8"
    )
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    batch = BatchSettings(
        [first, second],
        config_path=config,
        profile="srt",
        values={"hardware.device": "cpu"},
        overrides={second: VideoOverride(profile="voice", values={"tts.verify_final_audio": True})},
    )

    items = batch.resolve_items()

    assert not items[0].settings.tts.enabled
    assert items[1].settings.tts.enabled
    assert items[1].settings.tts.speaker == "Ryan"
    assert items[1].settings.tts.verify_final_audio
    assert all(item.settings.hardware.device == "cpu" for item in items)
    items[0].settings.tts.enabled = True
    assert items[1].settings.tts.enabled


def test_heterogeneous_batch_preserves_order_and_continues_after_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from transcript_video.application import processing
    from transcript_video.application.batch import ProcessItem
    from transcript_video.config import RunSettings

    source_dir = tmp_path / "data/subtitles/source"
    source_dir.mkdir(parents=True)
    videos = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    for video in videos:
        video.touch()
        (source_dir / f"{video.stem}_vi_faster.srt").write_text("", encoding="utf-8")
    first = RunSettings.defaults()
    first.project.root = str(tmp_path)
    first.hardware.device = "cpu"
    second = deepcopy(first)
    second.hardware.device = "cuda"
    seen = []

    def process(video, _model, _source, _translated, _paths, settings, _observer):
        seen.append((video.name, settings.hardware.device))
        if video.name == "first.mp4":
            raise RuntimeError("intentional")

    monkeypatch.setattr(processing, "process_video", process)
    observer = RecordingObserver()
    plan = processing.build_batch_process_plan(
        [ProcessItem(videos[0], first), ProcessItem(videos[1], second)]
    )

    summary = processing.execute_batch_process_plan(plan, observer)

    assert seen == [("first.mp4", "cpu"), ("second.mp4", "cuda")]
    assert summary.total == 2 and summary.succeeded == 1 and len(summary.failures) == 1
    video_events = [event for event in observer.events if event.stage == PipelineStage.VIDEO]
    assert [event.context.video for event in video_events if event.kind == EventKind.START] == [
        "first.mp4",
        "second.mp4",
    ]
    assert any(event.kind == EventKind.FAILURE for event in video_events)


def test_interactive_controller_applies_and_resets_sparse_overrides(tmp_path: Path) -> None:
    from transcript_video.interactive import InteractiveBatchController

    config = tmp_path / "run.toml"
    config.write_text("", encoding="utf-8")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "srt.toml").write_text("[tts]\nenabled = false\n", encoding="utf-8")
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    controller = InteractiveBatchController(tmp_path, config)
    controller.select([first, second])
    controller.update_defaults({"tts.enabled": True, "tts.speaker": "Aiden"})
    controller.apply([second], {"tts.speaker": "Ryan"}, profile="srt")

    items = controller.resolved_items()

    assert [item.settings.tts.speaker for item in items] == ["Aiden", "Ryan"]
    assert all(item.settings.tts.enabled for item in items)
    controller.reset([second])
    assert [item.settings.tts.speaker for item in controller.resolved_items()] == ["Aiden", "Aiden"]


def test_batch_plan_rejects_duplicate_video(tmp_path: Path) -> None:
    from transcript_video.application.batch import ProcessItem
    from transcript_video.application.processing import build_batch_process_plan
    from transcript_video.config import RunSettings

    source_dir = tmp_path / "data/subtitles/source"
    source_dir.mkdir(parents=True)
    video = tmp_path / "lesson.mp4"
    video.touch()
    (source_dir / "lesson_vi_faster.srt").write_text("", encoding="utf-8")
    settings = RunSettings.defaults()
    settings.project.root = str(tmp_path)

    with pytest.raises(ValueError, match="selected more than once"):
        build_batch_process_plan(
            [ProcessItem(video, settings), ProcessItem(video, deepcopy(settings))]
        )

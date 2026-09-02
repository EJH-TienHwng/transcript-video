from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transcript_video.cli import parse_args
from transcript_video.config import (
    RunSettings,
    SubtitleSegment,
    load_run_settings,
    save_run_settings,
)
from transcript_video.course.config import CourseConfig, load_course_config
from transcript_video.hardware import get_ffmpeg_exe, video_encoder_args
from transcript_video.processing.models import detect_model_type, read_transformers_model_config
from transcript_video.processing.subtitles import (
    parse_srt_timestamp,
    read_srt,
    remove_repeated_hallucination_segments,
)


class ModelUtilsTests(unittest.TestCase):
    def test_detects_sharded_transformers_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "whisper"}),
                encoding="utf-8",
            )
            (model_dir / "model-00001-of-00002.safetensors").touch()

            self.assertEqual(detect_model_type(model_dir), "huggingface")

    def test_reports_invalid_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not read model config"):
                read_transformers_model_config(model_dir)


class SubtitleTests(unittest.TestCase):
    def test_rejects_out_of_range_timestamp(self) -> None:
        with self.assertRaises(ValueError):
            parse_srt_timestamp("00:60:00,000")
        with self.assertRaises(ValueError):
            parse_srt_timestamp("00:00:60,000")

    def test_reads_optional_srt_positioning_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            srt_path = Path(temp_dir) / "positioned.srt"
            srt_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,500 position:50%\nHello\n",
                encoding="utf-8",
            )

            self.assertEqual(
                read_srt(srt_path),
                [SubtitleSegment(1.0, 2.5, "Hello")],
            )

    def test_only_filters_consecutive_repetitions(self) -> None:
        segments = [
            SubtitleSegment(index, index + 1, text)
            for index, text in enumerate(["Again", "Other", "Again", "Again", "Again", "Again"])
        ]

        cleaned = remove_repeated_hallucination_segments(
            segments,
            max_same_text_count=3,
        )

        self.assertEqual(
            [segment.text for segment in cleaned],
            ["Again", "Other", "Again", "Again", "Again"],
        )


class CourseConfigTests(unittest.TestCase):
    def _write_config(self, root: Path, overrides: dict) -> Path:
        config = {"sessions": [{"number": 1, "title": "Session", "video": "data/input.mp4"}]}
        config.update(overrides)
        config_dir = root / "courses"
        config_dir.mkdir()
        config_path = config_dir / "course.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_rejects_string_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                {"add_chapters": "false"},
            )

            with self.assertRaisesRegex(ValueError, "add_chapters"):
                load_course_config(config_path)

    def test_rejects_output_that_overwrites_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                {"output": "data/input.mp4"},
            )

            with self.assertRaisesRegex(ValueError, "overwrite"):
                load_course_config(config_path)

    def test_rejects_unknown_video_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                {"render": {"video_encoder": "unknown"}},
            )

            with self.assertRaisesRegex(ValueError, "render.video_encoder"):
                load_course_config(config_path)


class CourseMediaTests(unittest.TestCase):
    def test_normalization_uses_source_video_duration(self) -> None:
        from transcript_video.course import media

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_in = root / "input.mp4"
            video_in.touch()
            video_out = root / "output.mp4"
            config = CourseConfig(
                title="Course",
                output=root / "course.mp4",
                theme_image=None,
                sessions=[],
            )

            with (
                mock.patch.object(
                    media,
                    "get_media_duration_seconds",
                    return_value=12.5,
                ),
                mock.patch.object(media, "media_has_audio", return_value=True),
                mock.patch.object(media, "get_ffmpeg_exe", return_value="ffmpeg"),
                mock.patch.object(
                    media,
                    "video_encoder_args",
                    return_value=["-c:v", "h264_nvenc"],
                ),
                mock.patch.object(media, "run_command") as run_command,
            ):
                media.normalize_session_video(video_in, video_out, config)

            command = run_command.call_args.args[0]
            duration_index = command.index("-t") + 1
            self.assertEqual(command[duration_index], "12.500")
            self.assertIn("h264_nvenc", command)


class HardwareTests(unittest.TestCase):
    def test_explicit_ffmpeg_path_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ffmpeg_path = Path(temp_dir) / "ffmpeg.exe"
            ffmpeg_path.touch()

            with mock.patch.dict(
                "os.environ",
                {"TRANSCRIPT_VIDEO_FFMPEG": str(ffmpeg_path)},
                clear=False,
            ):
                selected = get_ffmpeg_exe()

        self.assertEqual(selected, str(ffmpeg_path.resolve()))

    def test_auto_video_encoder_prefers_nvenc(self) -> None:
        with mock.patch(
            "transcript_video.hardware.ffmpeg_encoder_available",
            return_value=True,
        ):
            args = video_encoder_args("ffmpeg", "auto", bitrate="8M")

        self.assertEqual(args[:4], ["-c:v", "h264_nvenc", "-preset", "p4"])
        self.assertIn("8M", args)

    def test_auto_video_encoder_falls_back_to_libx264(self) -> None:
        with mock.patch(
            "transcript_video.hardware.ffmpeg_encoder_available",
            return_value=False,
        ):
            args = video_encoder_args("ffmpeg", "auto")

        self.assertEqual(args[:4], ["-c:v", "libx264", "-preset", "medium"])


class RunSettingsTests(unittest.TestCase):
    def test_saved_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "run.toml"
            settings = RunSettings.defaults()
            settings.project.video = "lesson.mp4"
            settings.tts.enabled = True

            save_run_settings(settings, config_path)

            self.assertEqual(load_run_settings(config_path), settings)

    def test_cli_values_override_saved_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "run.toml"
            save_run_settings(RunSettings.defaults(), config_path)

            _args, settings = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--enable-tts",
                ]
            )

            self.assertEqual(settings.hardware.device, "cpu")
            self.assertTrue(settings.tts.enabled)

    def test_rejects_invalid_toml_value_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "run.toml"
            config_path.write_text(
                '[tts]\nchunk_minutes = "five"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "tts.chunk_minutes must be an integer"):
                parse_args(["--config", str(config_path)])

    def test_allows_empty_language_for_auto_detection(self) -> None:
        _args, settings = parse_args(["--language", ""])

        self.assertEqual(settings.transcription.language, "")

    def test_migrates_legacy_hardware_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "legacy.toml"
            config_path.write_text(
                '[transcription]\ndevice = "cpu"\ncompute_type = "int8"\n',
                encoding="utf-8",
            )

            settings = load_run_settings(config_path)

            self.assertEqual(settings.hardware.device, "cpu")
            self.assertEqual(settings.hardware.compute_type, "int8")

    def test_rejects_unsupported_compute_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.toml"
            config_path.write_text(
                '[hardware]\ncompute_type = "fastest"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "hardware.compute_type"):
                parse_args(["--config", str(config_path)])


if __name__ == "__main__":
    unittest.main()

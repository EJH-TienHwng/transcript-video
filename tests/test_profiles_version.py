from pathlib import Path

import pytest
from scripts.check_python_version import check_versions

from transcript_video.application.settings import resolve_settings


def test_project_python_versions_agree():
    check_versions(Path(__file__).resolve().parents[1])


@pytest.mark.parametrize(
    "ci,requirement",
    [("3.99", ">=3.99,<3.100"), ("3.98", ">=3.99,<3.100"), ("3.99", ">=3.98,<3.99")],
)
def test_python_version_mismatch(tmp_path, ci, requirement):
    (tmp_path / ".python-version").write_text("3.99\n")
    (tmp_path / "pyproject.toml").write_text(f'[project]\nrequires-python = "{requirement}"\n')
    workflow = tmp_path / ".github/workflows/quality.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(f"with:\n  python-version: '{ci}'\n")
    if ci != "3.99" or "3.98" in requirement:
        with pytest.raises(ValueError, match=r"\.python-version is 3\.99 but"):
            check_versions(tmp_path)
    else:
        check_versions(tmp_path)


def test_shipped_profiles_and_cli_precedence():
    base = Path(__file__).resolve().parents[1] / "configs/transcription.toml"
    srt = resolve_settings(config_path=base, profile="srt")
    assert srt.settings.transcription.skip_burn and not srt.settings.tts.enabled
    review = resolve_settings(config_path=base, profile="tts-review")
    assert review.settings.tts.verify_final_audio and not review.settings.transcription.skip_burn
    assert review.settings.transcription.language == "vi"
    assert review.settings.tts.language == "English"
    overridden = resolve_settings(
        config_path=base,
        profile="tts-review",
        overrides={"tts.speaker": "Ryan", "tts.verify_final_audio": False},
    )
    assert (
        overridden.settings.tts.speaker == "Ryan" and not overridden.settings.tts.verify_final_audio
    )
    assert overridden.sources["tts.verify_final_audio"] == "command line"

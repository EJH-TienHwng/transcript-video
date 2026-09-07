import pytest

from transcript_video.application.settings import resolve_settings
from transcript_video.config import SubtitleStyle
from transcript_video.processing.media import build_force_style


def test_style_default_and_explicit_values():
    assert build_force_style(SubtitleStyle()) == "MarginV=25"
    assert (
        build_force_style(
            SubtitleStyle(
                font_name="Noto Sans",
                font_size=18,
                primary_color="&H00FFFFFF",
                outline=1.5,
                margin_left=10,
            )
        )
        == "FontName=Noto Sans,FontSize=18,PrimaryColour=&H00FFFFFF,Outline=1.5,MarginV=25,MarginL=10"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("font_name", "Font,MarginV=0"),
        ("font_name", "Font':evil"),
        ("font_name", "a\\b"),
        ("font_name", "a\nb"),
        ("font_size", 0),
        ("font_size", float("nan")),
        ("primary_color", "red"),
        ("outline_color", "&H00FFFFFZ"),
        ("border_style", 2),
        ("alignment", 10),
        ("outline", -1),
        ("margin_left", True),
        ("margin_vertical", None),
        ("shadow", "1"),
    ],
)
def test_style_rejects_invalid_values_and_filter_injection(key, value):
    with pytest.raises(ValueError, match="subtitle_style"):
        build_force_style(SubtitleStyle(**{key: value}))


def test_style_config_flows_through_shared_validation(tmp_path):
    config = tmp_path / "style.toml"
    config.write_text('[subtitle_style]\nfont_name="Arial"\nfont_size=18\nmargin_vertical=30\n')
    result = resolve_settings(config_path=config)
    assert (
        build_force_style(result.settings.subtitle_style) == "FontName=Arial,FontSize=18,MarginV=30"
    )
    config.write_text("[subtitle_style]\nalignment=0\n")
    with pytest.raises(ValueError, match="alignment"):
        resolve_settings(config_path=config)

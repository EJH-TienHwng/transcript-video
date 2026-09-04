from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..config import find_project_root
from .config import CourseConfig
from .timeline import SessionTimeline, format_video_timestamp, session_number

logger = logging.getLogger(__name__)
TOC_BACKGROUND = Path("assets/table_of_content.png")


def _find_default_font(bold: bool = False) -> Path | None:
    """Find a common system font without shipping font files with the project."""
    candidates = []

    if os.name == "nt":
        windows = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates.extend(
            [
                windows / ("arialbd.ttf" if bold else "arial.ttf"),
                windows / ("segoeuib.ttf" if bold else "segoeui.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                    if bold
                    else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ),
                Path(
                    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
                    if bold
                    else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
                ),
            ]
        )

    return next((path for path in candidates if path.exists()), None)


def _load_font(config: CourseConfig, size: int, bold: bool = False):
    font_path = config.render.font_path or _find_default_font(bold=bold)
    if font_path and font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)

    logger.warning("TrueType font not found; using Pillow's default font.")
    return ImageFont.load_default()


def _cover_image(image: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.fit(
        image.convert("RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def _make_background(
    config: CourseConfig,
    image_path: Path | None = None,
    darken: bool = True,
) -> Image.Image:
    width, height = config.render.width, config.render.height
    image_path = image_path or config.theme_image

    if image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"Background image not found: {image_path}")
        with Image.open(image_path) as source:
            background = _cover_image(source, width, height)
    else:
        background = Image.new("RGB", (width, height), (24, 24, 24))

    if not darken:
        return background.convert("RGBA")

    # Dark transparent overlay improves text readability on arbitrary theme images.
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 96))
    return Image.alpha_composite(background.convert("RGBA"), overlay)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    font,
    center_x: float,
    start_y: float,
    fill,
    line_gap: int,
) -> float:
    y = start_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        draw.text((center_x - line_width / 2, y), line, font=font, fill=fill)
        y += line_height + line_gap
    return y


def render_session_card(
    config: CourseConfig,
    timeline_item: SessionTimeline,
    position: int,
    output_path: Path,
) -> None:
    """Render one static PNG title card for a session."""
    image = _make_background(config)
    draw = ImageDraw.Draw(image)

    width, height = image.size
    number = session_number(timeline_item.session, position)

    label_font = _load_font(config, max(30, width // 45), bold=True)
    title_font = _load_font(config, max(42, width // 24), bold=True)
    time_font = _load_font(config, max(24, width // 60), bold=False)

    label = f"SESSION {number:02d}"
    label_bbox = draw.textbbox((0, 0), label, font=label_font)
    label_width = label_bbox[2] - label_bbox[0]

    draw.text(
        ((width - label_width) / 2, height * 0.34),
        label,
        font=label_font,
        fill="white",
    )

    title_lines = _wrap_text(
        draw,
        timeline_item.session.title,
        title_font,
        max_width=int(width * 0.72),
    )
    y = _draw_centered_lines(
        draw=draw,
        lines=title_lines,
        font=title_font,
        center_x=width / 2,
        start_y=height * 0.44,
        fill="white",
        line_gap=max(12, height // 90),
    )

    start_text = f"Starts at {format_video_timestamp(timeline_item.content_start)}"
    time_bbox = draw.textbbox((0, 0), start_text, font=time_font)
    time_width = time_bbox[2] - time_bbox[0]
    draw.text(
        ((width - time_width) / 2, min(y + height * 0.04, height * 0.78)),
        start_text,
        font=time_font,
        fill=(225, 225, 225, 255),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def render_toc_pages(
    config: CourseConfig,
    timeline: Sequence[SessionTimeline],
    output_dir: Path,
) -> list[Path]:
    """Render one or more TOC PNG pages with absolute timestamps."""
    if not config.toc.enabled:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    items_per_page = config.toc.items_per_page
    page_paths: list[Path] = []
    toc_background = find_project_root(config.work_dir) / TOC_BACKGROUND
    text_color = "black"

    for page_index, start in enumerate(range(0, len(timeline), items_per_page), start=1):
        page_items = timeline[start : start + items_per_page]
        image = _make_background(config, toc_background, darken=False)
        draw = ImageDraw.Draw(image)
        width, height = image.size

        heading_font = _load_font(config, max(40, width // 28), bold=True)
        row_font = _load_font(config, max(27, width // 52), bold=False)
        time_font = _load_font(config, max(27, width // 52), bold=True)

        heading = config.toc.heading
        heading_bbox = draw.textbbox((0, 0), heading, font=heading_font)
        heading_width = heading_bbox[2] - heading_bbox[0]
        draw.text(
            ((width - heading_width) / 2, height * 0.13),
            heading,
            font=heading_font,
            fill=text_color,
        )

        if len(timeline) > items_per_page:
            page_label = f"{page_index} / {(len(timeline) + items_per_page - 1) // items_per_page}"
            page_font = _load_font(config, max(20, width // 75), bold=False)
            page_bbox = draw.textbbox((0, 0), page_label, font=page_font)
            draw.text(
                (width - (page_bbox[2] - page_bbox[0]) - width * 0.06, height * 0.08),
                page_label,
                font=page_font,
                fill=text_color,
            )

        left_x = width * 0.14
        title_x = width * 0.20
        time_right_x = width * 0.86
        top_y = height * 0.28
        usable_height = height * 0.58
        row_height = usable_height / max(1, len(page_items))

        for local_index, item in enumerate(page_items):
            absolute_position = start + local_index + 1
            number = session_number(item.session, absolute_position)
            y = top_y + local_index * row_height
            timestamp = format_video_timestamp(item.content_start)

            number_text = f"{number:02d}"
            draw.text(
                (left_x, y),
                number_text,
                font=time_font,
                fill=text_color,
            )

            max_title_width = int(time_right_x - title_x - width * 0.13)
            title_lines = _wrap_text(
                draw,
                item.session.title,
                row_font,
                max_width=max_title_width,
            )
            # TOC rows are deliberately compact; keep at most two visual lines.
            title_lines = title_lines[:2]
            _draw_centered_lines(
                draw=draw,
                lines=title_lines,
                font=row_font,
                center_x=title_x + max_title_width / 2,
                start_y=y,
                fill=text_color,
                line_gap=4,
            )

            timestamp_bbox = draw.textbbox((0, 0), timestamp, font=time_font)
            timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
            draw.text(
                (time_right_x - timestamp_width, y),
                timestamp,
                font=time_font,
                fill=text_color,
            )

        page_path = output_dir / f"toc_{page_index:03d}.png"
        image.convert("RGB").save(page_path, quality=95)
        page_paths.append(page_path)

    return page_paths

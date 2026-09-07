"""Small versioned cache manifests; no full-video hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..artifacts import write_text
from ..context import log_context
from ..events import EventKind, emit
from .models import detect_model_type

CACHE_VERSION = 1


def file_identity(path: Path) -> dict:
    path = path.expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_identity(value: str | Path) -> dict:
    path = Path(value).expanduser()
    if not path.exists():
        return {"name": str(value)}
    if path.is_file():
        return file_identity(path)
    return {
        "path": str(path.resolve()),
        "files": [file_identity(p) for p in sorted(path.rglob("*")) if p.is_file()],
    }


def subtitle_provenance(video, model, translation, settings) -> dict:
    return dict(
        input=file_identity(video),
        model=model_identity(model),
        model_type=detect_model_type(model),
        translation_model=model_identity(translation) if translation else None,
        # Version covers fixed decoding/filtering defaults in transcription/subtitles.
        transcription={
            k: v
            for k, v in asdict(settings.transcription).items()
            if k not in {"overwrite_srt", "skip_burn"}
        },
        device=settings.hardware.device,
        compute_type=settings.hardware.compute_type,
    )


def tts_provenance(segments, config: dict) -> dict:
    from .tts.core import TTS_REVIEW_VERSION

    return dict(
        review_version=TTS_REVIEW_VERSION,
        subtitles=[
            dict(start=float(segment.start), end=float(segment.end), text=segment.text)
            for segment in segments
        ],
        **{
            k: v
            for k, v in config.items()
            if k not in {"enabled", "overwrite", "rerun_chunk", "audio_mode", "split_audio"}
        },
    )


def manifest_path(artifact: Path) -> Path:
    if artifact.suffix == ".wav":
        audio_root = next(
            (p for p in artifact.parents if p.name == "audio" and p.parent.name == "data"), None
        )
        if audio_root:
            return (
                audio_root.parent
                / "report/tts/provenance"
                / artifact.relative_to(audio_root).with_suffix(".json")
            )
        return artifact.parent / ".provenance" / (artifact.name + ".json")
    return artifact.with_suffix(artifact.suffix + ".provenance.json")


def fingerprint(provenance: dict) -> str:
    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def cache_matches(artifact: Path, provenance: dict, stage) -> bool:
    if not artifact.is_file():
        return False
    reason = "missing or outdated provenance"
    try:
        saved = json.loads(manifest_path(artifact).read_text(encoding="utf-8"))
        if saved["schema_version"] == CACHE_VERSION:
            if saved["fingerprint"] == fingerprint(provenance) and (
                artifact.suffix != ".wav"
                or saved.get("artifact_identity") == file_identity(artifact)
            ):
                return True
            old = saved["provenance"]
            if not isinstance(old, dict):
                raise ValueError("Invalid provenance object")
            changed = [key for key in provenance if old.get(key) != provenance[key]]
            reason = (
                ", ".join(changed[:3]) + " changed"
                if changed
                else "artifact replaced since provenance was written"
            )
    except (OSError, ValueError, KeyError, TypeError):
        pass  # Unreadable metadata cannot authorize cache reuse.
    with log_context(operation="cache"):
        emit(
            stage,
            f"{stage.value.upper()} cache invalidated: {reason}",
            kind=EventKind.WARNING,
            artifact=artifact,
            details={"decision": "invalidated", "reason": reason},
        )
    return False


def write_provenance(artifact: Path, provenance: dict) -> None:
    path = manifest_path(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(
        path,
        json.dumps(
            dict(
                schema_version=CACHE_VERSION,
                fingerprint=fingerprint(provenance),
                provenance=provenance,
                artifact_identity=file_identity(artifact) if artifact.is_file() else None,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

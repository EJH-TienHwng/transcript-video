"""Check the canonical project pin against packaging and hosted CI."""

import re
import tomllib
from pathlib import Path


def check_versions(root: Path) -> None:
    pin = (root / ".python-version").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+", pin):
        raise ValueError(".python-version must contain a major.minor project pin")
    major, minor = map(int, pin.split("."))
    with (root / "pyproject.toml").open("rb") as stream:
        requirement = tomllib.load(stream)["project"]["requires-python"]
    expected = f">={pin},<{major}.{minor + 1}"
    if requirement.replace(" ", "") != expected:
        raise ValueError(
            f".python-version is {pin} but requires-python is {requirement}; expected {expected}"
        )
    workflow = (root / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    versions = re.findall(r"^\s+python-version:\s*['\"]?([\d.]+)['\"]?\s*$", workflow, re.MULTILINE)
    if not versions:
        raise ValueError(
            "GitHub Actions has no literal python-version to compare with .python-version"
        )
    for version in versions:
        if version != pin:
            raise ValueError(f".python-version is {pin} but GitHub Actions uses {version}")


if __name__ == "__main__":
    check_versions(Path(__file__).resolve().parents[1])
    print("Python version declarations agree with .python-version")

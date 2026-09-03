from __future__ import annotations

import sys
import warnings


def main() -> None:
    warnings.warn(
        "transcript-course-config is deprecated; use 'transcript-video course create'.",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.argv[1:1] = ["course", "create"]
    from ..cli import main as unified_main

    unified_main()


if __name__ == "__main__":
    main()

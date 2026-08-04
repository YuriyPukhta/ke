"""Deprecated path: use `prosody_steer/optimize_prosody_pso.py` instead."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "prosody_steer" / "optimize_prosody_pso.py"


def main() -> None:
    print(
        "NOTE: moved to prosody_steer/optimize_prosody_pso.py — redirecting.",
        flush=True,
    )
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")


if __name__ == "__main__":
    main()

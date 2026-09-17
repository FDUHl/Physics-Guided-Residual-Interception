"""Add the repository and bundled simulator to the Python import path."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

for path in (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "demo",
    REPOSITORY_ROOT / "src" / "aerial_gym_simulator",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

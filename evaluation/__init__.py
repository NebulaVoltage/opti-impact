"""Dataset analysis and evaluation package."""

from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
PLOTS_DIR = EVALUATION_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

"""Run the config-driven MS MARCO GenQA experiment pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msmarco_genqa.cli.pipeline import main


if __name__ == "__main__":
    main()

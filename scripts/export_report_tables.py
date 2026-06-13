"""Export LaTeX report tables from checked report artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from msmarco_genqa.reporting.latex_tables import export_tables, validate_sidecar_current


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "reports" / "generated" / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated" / "tables"


def _default_artifacts() -> list[Path]:
    return sorted(DEFAULT_ARTIFACT_DIR.glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        dest="artifacts",
        help="Artifact JSON path. Defaults to reports/generated/artifacts/*.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check-sources",
        action="store_true",
        help="Check existing .sources.json files instead of writing table fragments.",
    )
    args = parser.parse_args()

    artifacts = args.artifacts if args.artifacts else _default_artifacts()
    if not artifacts:
        print(f"No artifacts found under {DEFAULT_ARTIFACT_DIR}", file=sys.stderr)
        return 1

    if args.check_sources:
        for sidecar in sorted(args.output_dir.glob("*.sources.json")):
            validate_sidecar_current(sidecar)
        print("Table source sidecars are current")
        return 0

    written = export_tables(artifact_paths=artifacts, output_dir=args.output_dir)
    for path in written:
        print(path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


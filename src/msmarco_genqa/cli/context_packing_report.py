"""``mgq-context-packing-report`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.evaluation.context_packing_report import (
    build_context_packing_report,
    load_prediction_jsonl,
    write_context_packing_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--compressed-predictions", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--compressed-name", default="compressed")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    return (
        path.relative_to(PROJECT_ROOT).as_posix()
        if path.is_relative_to(PROJECT_ROOT)
        else str(path)
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    baseline_path = _resolve(args.baseline_predictions)
    compressed_path = _resolve(args.compressed_predictions)
    for label, path in (
        ("baseline predictions", baseline_path),
        ("compressed predictions", compressed_path),
    ):
        if not path.exists():
            raise SystemExit(f"{label} file not found: {path}")

    report = build_context_packing_report(
        load_prediction_jsonl(baseline_path),
        load_prediction_jsonl(compressed_path),
        baseline_name=args.baseline_name,
        compressed_name=args.compressed_name,
    )
    report["inputs"] = {
        "baseline_predictions": _display_path(baseline_path),
        "compressed_predictions": _display_path(compressed_path),
    }
    output_dir = _resolve(args.output_dir)
    write_context_packing_outputs(report, output_dir=output_dir)
    print(f"Wrote context packing comparison to {output_dir}")


if __name__ == "__main__":
    main()

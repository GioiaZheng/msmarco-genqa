"""Build comparison tables from local experiment-tracking events."""

from __future__ import annotations

import argparse
from pathlib import Path

from msmarco_genqa.util.sweep_summary import discover_event_files, write_sweep_summary


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mgq-sweep-summary", description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Tracking directories or events.jsonl files to summarize.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional sweep name to record in summary outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/tracking_summaries",
        help="Directory for sweep_summary.{json,csv,md}.",
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.paths:
        raise SystemExit("provide at least one tracking directory or events.jsonl file")

    tracking_paths = [_resolve(path) for path in args.paths]
    try:
        event_files = discover_event_files(tracking_paths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not event_files:
        raise SystemExit("no events.jsonl files found")

    output_dir = _resolve(args.output_dir)
    paths = write_sweep_summary(event_files, output_dir, sweep_name=args.name)
    print(f"sweep summary: {len(event_files)} runs, output={output_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

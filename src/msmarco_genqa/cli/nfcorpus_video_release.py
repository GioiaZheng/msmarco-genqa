"""Manage the public NFCorpus video query-representation release bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msmarco_genqa.reproducibility.nfcorpus_video_release import (
    ReleaseArtifactError,
    build_release_bundle,
    evaluate_release_bundle,
    fetch_release_bundle,
    verify_release_archive,
)


DEFAULT_POINTER = Path("artifacts/nfcorpus_video_query_representation_v1.json")
DEFAULT_OUTPUT = Path("outputs/reproductions/nfcorpus_video_query_representation_v1")


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--qrels",
        type=Path,
        help="Optional local NFCorpus qrels (format auto-detected); otherwise use ir_datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional IR_DATASETS_HOME used to recover the public qrels.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the deterministic release ZIP.")
    build.add_argument(
        "--source-record",
        type=Path,
        default=Path(
            "reports/generated/artifacts/nfcorpus_video_query_representation.json"
        ),
    )
    build.add_argument("--project-root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify an existing release ZIP.")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--sha256")
    verify.add_argument("--bytes", type=int)

    fetch = subparsers.add_parser("fetch", help="Download and extract the pinned release.")
    fetch.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    fetch.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Recompute aggregate and paired metrics from an extracted bundle.",
    )
    evaluate.add_argument("--bundle-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    _add_evaluation_arguments(evaluate)

    reproduce = subparsers.add_parser(
        "reproduce",
        help="Download, verify, and recompute the published ablation.",
    )
    reproduce.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    reproduce.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    _add_evaluation_arguments(reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_release_bundle(
                source_record_path=args.source_record,
                project_root=args.project_root,
                output_archive=args.output,
            )
        elif args.command == "verify":
            result = verify_release_archive(
                args.archive,
                expected_sha256=args.sha256,
                expected_bytes=args.bytes,
            )
        elif args.command == "fetch":
            result = fetch_release_bundle(
                pointer_path=args.pointer,
                output_dir=args.output_dir,
            )
        elif args.command == "evaluate":
            result = evaluate_release_bundle(
                bundle_dir=args.bundle_dir,
                output_dir=args.output_dir,
                qrels_path=args.qrels,
                cache_dir=args.cache_dir,
            )
        else:
            fetched = fetch_release_bundle(
                pointer_path=args.pointer,
                output_dir=args.output_dir,
            )
            result = evaluate_release_bundle(
                bundle_dir=fetched["bundle_dir"],
                output_dir=Path(args.output_dir) / "evaluation",
                qrels_path=args.qrels,
                cache_dir=args.cache_dir,
            )
    except (OSError, ReleaseArtifactError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

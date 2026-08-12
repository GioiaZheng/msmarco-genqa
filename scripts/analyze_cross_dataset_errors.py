"""Compare fixed NFCorpus and SciFact first-stage error diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from msmarco_genqa.evaluation.cross_dataset_errors import (
    CrossDatasetErrorAnalysisError,
    assert_cross_dataset_fingerprint,
    build_cross_dataset_error_analysis,
    render_cross_dataset_error_markdown,
    summarize_taxonomy_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NFCORPUS_SUMMARY = (
    PROJECT_ROOT / "outputs" / "analysis" / "nfcorpus_first_stage" / "summary.json"
)
DEFAULT_SCIFACT_SUMMARY = (
    PROJECT_ROOT / "outputs" / "analysis" / "scifact_first_stage" / "summary.json"
)
DEFAULT_TAXONOMY = (
    PROJECT_ROOT
    / "reports"
    / "annotations"
    / "nfcorpus_first_stage_review_v1.csv"
)
DEFAULT_SCIFACT_REVIEW = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "scifact_first_stage"
    / "review"
    / "review_summary.json"
)
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "cross_dataset_error_analysis.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "cross_dataset_errors"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrossDatasetErrorAnalysisError(f"{_display(path)}: expected object")
    return value


def _load_taxonomy(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return summarize_taxonomy_rows(list(csv.DictReader(handle)))
    except OSError as exc:
        raise CrossDatasetErrorAnalysisError(
            f"cannot read taxonomy CSV {_display(path)}: {exc}"
        ) from exc


def _load_optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json_object(path)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfcorpus-summary", type=Path, default=DEFAULT_NFCORPUS_SUMMARY)
    parser.add_argument("--scifact-summary", type=Path, default=DEFAULT_SCIFACT_SUMMARY)
    parser.add_argument("--nfcorpus-taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--scifact-review", type=Path, default=DEFAULT_SCIFACT_REVIEW)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    try:
        nfcorpus_summary = _load_json_object(_resolve(args.nfcorpus_summary))
        scifact_summary = _load_json_object(_resolve(args.scifact_summary))
        taxonomy = _load_taxonomy(_resolve(args.nfcorpus_taxonomy))
        scifact_review = _load_optional_json_object(_resolve(args.scifact_review))
        report = build_cross_dataset_error_analysis(
            {
                "NFCorpus": nfcorpus_summary,
                "SciFact": scifact_summary,
            },
            nfcorpus_taxonomy=taxonomy,
            scifact_residual_review=scifact_review,
        )
        contract = _load_json_object(_resolve(args.contract))
        assert_cross_dataset_fingerprint(
            report,
            contract.get("expected_fingerprint"),
        )
    except (
        CrossDatasetErrorAnalysisError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cross-dataset error analysis failed: {exc}", file=sys.stderr)
        return 1

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", report)
    markdown = render_cross_dataset_error_markdown(
        report,
        nfcorpus_doc="docs/nfcorpus_first_stage_error_analysis.md",
        scifact_doc="docs/scifact_first_stage_error_analysis.md",
        scifact_review_doc="docs/scifact_failure_review.md",
    )
    (output_dir / "report.md").write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )

    comparison = report["comparison"]
    print(
        "Cross-dataset first-stage analysis: "
        "SciFact - NFCorpus Recall@100 = "
        f"{comparison['recall_at_100_gap_scifact_minus_nfcorpus']:+.4f}"
    )
    print(
        "No-hit-at-100 share gap (NFCorpus - SciFact) = "
        f"{comparison['no_relevant_top_100_share_gap_nfcorpus_minus_scifact']:+.1%}"
    )
    print(f"Wrote {_display(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

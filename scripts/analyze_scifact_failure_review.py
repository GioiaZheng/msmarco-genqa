"""Review residual SciFact first-stage failures from frozen BM25 outputs."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from msmarco_genqa.evaluation.first_stage_coverage import (
    FirstStageCoverageError,
    analyze_first_stage_coverage,
    assert_first_stage_diagnostic_fingerprint,
)
from msmarco_genqa.evaluation.retrieval_contract import (
    RetrievalContractError,
    verify_retrieval_contract,
)
from msmarco_genqa.evaluation.scifact_failure_review import (
    SciFactFailureReviewError,
    assert_scifact_failure_fingerprint,
    build_scifact_failure_cases,
    render_scifact_failure_review_markdown,
    summarize_scifact_failure_review,
)
from msmarco_genqa.evaluation.trec import QrelsFormatError, read_qrels
from msmarco_genqa.reranking.io import RunTsvFormatError, read_run_tsv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "scifact_first_stage_contract.json"
DEFAULT_REVIEW_CONTRACT = PROJECT_ROOT / "configs" / "scifact_failure_review.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "analysis" / "scifact_first_stage" / "review"
)


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
        raise SciFactFailureReviewError(f"{_display(path)}: expected object")
    return value


def _load_jsonl_member(
    archive_path: Path,
    member: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
                for line_number, raw_line in enumerate(text, start=1):
                    value = json.loads(raw_line)
                    item_id = value.get("_id") if isinstance(value, dict) else None
                    if not isinstance(item_id, str) or not item_id:
                        raise SciFactFailureReviewError(
                            f"{member}:{line_number}: invalid _id"
                        )
                    if item_id in records:
                        raise SciFactFailureReviewError(
                            f"{member}:{line_number}: duplicate _id {item_id!r}"
                        )
                    records[item_id] = value
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile, KeyError) as exc:
        raise SciFactFailureReviewError(
            f"cannot read source member {member!r}: {exc}"
        ) from exc
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--review-contract", type=Path, default=DEFAULT_REVIEW_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    try:
        contract_path = _resolve(args.contract).resolve()
        contract_report = verify_retrieval_contract(
            contract_path,
            project_root=PROJECT_ROOT,
        )
        contract = _load_json_object(contract_path)
        review_contract = _load_json_object(_resolve(args.review_contract).resolve())
        source_record = contract["inputs"]["source_archive"]
        source_archive = (
            PROJECT_ROOT / contract_report["inputs"]["source_archive"]["path"]
        )
        query_records = _load_jsonl_member(
            source_archive,
            str(source_record["query_member"]),
        )
        corpus = _load_jsonl_member(
            source_archive,
            str(source_record["corpus_member"]),
        )
        queries = {
            qid: str(record.get("text") or "")
            for qid, record in query_records.items()
        }
        qrels = read_qrels(
            PROJECT_ROOT / contract_report["inputs"]["qrels"]["path"],
            qrels_format=str(contract["qrels_format"]),
        )
        run = read_run_tsv(
            PROJECT_ROOT / contract_report["inputs"]["bm25_run"]["path"]
        )
        analysis = analyze_first_stage_coverage(
            run,
            qrels,
            queries,
            rel_threshold=int(contract["binary_relevance_threshold"]),
        )
        assert_first_stage_diagnostic_fingerprint(
            analysis,
            contract["expected_first_stage_diagnostics"],
        )
        cases = build_scifact_failure_cases(
            analysis["per_query"],
            run,
            qrels,
            corpus,
            rel_threshold=int(contract["binary_relevance_threshold"]),
        )
        summary = summarize_scifact_failure_review(cases)
        assert_scifact_failure_fingerprint(
            summary,
            review_contract.get("expected_fingerprint"),
        )
    except (
        FirstStageCoverageError,
        SciFactFailureReviewError,
        RetrievalContractError,
        QrelsFormatError,
        RunTsvFormatError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"SciFact failure review failed: {exc}", file=sys.stderr)
        return 1

    output_dir = _resolve(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "review_cases.jsonl", cases)
    _write_json(output_dir / "review_summary.json", summary)
    markdown = render_scifact_failure_review_markdown(cases, summary)
    (output_dir / "review.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(
        "SciFact residual failure review: "
        f"{summary['n_cases']} no-hit@100 cases; "
        f"{summary['cohort_counts']['depth_recoverable_101_1000']} "
        "depth-recoverable; "
        f"{summary['cohort_counts']['miss_top_1000']} top-1000 misses"
    )
    print(f"Wrote {_display(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

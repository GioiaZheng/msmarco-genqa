"""Analyze first-stage coverage from frozen published BM25 outputs."""

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
    deterministic_bucket_samples,
    render_first_stage_coverage_markdown,
)
from msmarco_genqa.evaluation.retrieval_contract import (
    RetrievalContractError,
    verify_retrieval_contract,
)
from msmarco_genqa.evaluation.trec import QrelsFormatError, read_qrels
from msmarco_genqa.reranking.io import RunTsvFormatError, read_run_tsv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "nfcorpus_first_stage_contract.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "analysis" / "nfcorpus_first_stage"
)
DEFAULT_SAMPLE_SEED = "nfcorpus-first-stage-errors-v1"
DEFAULT_LABEL = "NFCorpus"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FirstStageCoverageError(f"{path}: expected a JSON object")
    return value


def _load_query_texts(archive_path: Path, member: str) -> dict[str, str]:
    queries: dict[str, str] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
                for line_number, raw_line in enumerate(text, start=1):
                    value = json.loads(raw_line)
                    qid = value.get("_id") if isinstance(value, dict) else None
                    query = value.get("text") if isinstance(value, dict) else None
                    if not isinstance(qid, str) or not qid:
                        raise FirstStageCoverageError(
                            f"{member}:{line_number}: invalid query _id"
                        )
                    if not isinstance(query, str) or not query.strip():
                        raise FirstStageCoverageError(
                            f"{member}:{line_number}: missing query text"
                        )
                    if qid in queries:
                        raise FirstStageCoverageError(
                            f"{member}:{line_number}: duplicate query id {qid!r}"
                        )
                    queries[qid] = query
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile, KeyError) as exc:
        raise FirstStageCoverageError(
            f"cannot read query member {member!r}: {exc}"
        ) from exc
    return queries


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--sample-seed", default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    args = parser.parse_args(argv)

    contract_path = _resolve(args.contract).resolve()
    output_dir = _resolve(args.output_dir).resolve()
    try:
        contract_report = verify_retrieval_contract(
            contract_path,
            project_root=PROJECT_ROOT,
        )
        contract = _load_json_object(contract_path)
        source_record = contract["inputs"]["source_archive"]
        query_member = str(source_record["query_member"])
        source_archive = (
            PROJECT_ROOT / contract_report["inputs"]["source_archive"]["path"]
        )
        qrels_path = PROJECT_ROOT / contract_report["inputs"]["qrels"]["path"]
        bm25_path = PROJECT_ROOT / contract_report["inputs"]["bm25_run"]["path"]
        queries = _load_query_texts(source_archive, query_member)
        qrels = read_qrels(
            qrels_path,
            qrels_format=str(contract["qrels_format"]),
        )
        bm25_run = read_run_tsv(bm25_path)
        report = analyze_first_stage_coverage(
            bm25_run,
            qrels,
            queries,
            rel_threshold=int(contract["binary_relevance_threshold"]),
        )
        expected = contract_report["expected_bm25_metrics"]
        tolerance = float(contract_report["metric_tolerance"])
        for metric in ("recall@100", "recall@1000"):
            delta = abs(
                float(report["macro_recall"][metric]) - float(expected[metric])
            )
            if delta > tolerance:
                raise FirstStageCoverageError(
                    f"{metric}: analysis drift {delta:.3g} exceeds {tolerance:g}"
                )
        assert_first_stage_diagnostic_fingerprint(
            report,
            contract.get("expected_first_stage_diagnostics"),
        )
        samples = deterministic_bucket_samples(
            report["per_query"],
            per_bucket=args.samples_per_bucket,
            seed=args.sample_seed,
        )
    except (
        FirstStageCoverageError,
        RetrievalContractError,
        QrelsFormatError,
        RunTsvFormatError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"First-stage analysis failed: {exc}", file=sys.stderr)
        return 1

    per_query = list(report.pop("per_query"))
    report["dataset_id"] = contract_report["dataset_id"]
    report["provenance"] = {
        "analysis_base_commit": contract_report["analysis_base_commit"],
        "experiment_commit": contract_report["experiment_commit"],
        "contract": contract_report["contract"],
        "release": contract_report["release"],
        "inputs": contract_report["inputs"],
    }
    report["sampling"] = {
        "method": "smallest SHA-256(seed, dimension, bucket, qid)",
        "seed": args.sample_seed,
        "per_bucket": args.samples_per_bucket,
        "n_rows": len(samples),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", report)
    _write_jsonl(output_dir / "per_query.jsonl", per_query)
    _write_jsonl(output_dir / "examples.jsonl", samples)
    markdown = render_first_stage_coverage_markdown(
        report,
        dataset_id=str(contract_report["dataset_id"]),
        contract_path=str(contract_report["contract"]),
        release_tag=str(contract_report["release"]["tag"]),
        samples=samples,
    )
    (output_dir / "report.md").write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )

    candidate = report["candidate_set_diagnostic"]
    depth = report["depth_100_to_1000_diagnostic"]
    print(
        f"{args.label} first-stage coverage: "
        f"{candidate['queries_with_no_relevant_in_top_100']}/"
        f"{report['scope']['n_queries']} queries have no relevant BM25 top-100 "
        "candidate"
    )
    print(
        "Macro recall: "
        f"@100={report['macro_recall']['recall@100']:.10f}, "
        f"@1000={report['macro_recall']['recall@1000']:.10f}, "
        f"gain={depth['macro_recall_gain']:+.10f}"
    )
    print(f"Wrote {_display_path(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

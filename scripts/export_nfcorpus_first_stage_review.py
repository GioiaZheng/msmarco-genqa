"""Export and validate the pre-declared NFCorpus first-stage review cohorts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from msmarco_genqa.evaluation.first_stage_coverage import (
    FirstStageCoverageError,
    analyze_first_stage_coverage,
    assert_first_stage_diagnostic_fingerprint,
)
from msmarco_genqa.evaluation.first_stage_review import (
    FirstStageReviewError,
    build_first_stage_review_cases,
    load_review_taxonomy,
    partition_query_ids_by_source,
    summarize_query_source_diagnostics,
    validate_review_annotations,
)
from msmarco_genqa.evaluation.retrieval_contract import (
    RetrievalContractError,
    verify_retrieval_contract,
)
from msmarco_genqa.evaluation.trec import (
    QrelsFormatError,
    evaluate_trec_retrieval,
    read_qrels,
)
from msmarco_genqa.reranking.io import RunTsvFormatError, read_run_tsv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "nfcorpus_first_stage_contract.json"
DEFAULT_TAXONOMY = (
    PROJECT_ROOT / "configs" / "nfcorpus_retrieval_review_taxonomy.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "analysis" / "nfcorpus_first_stage" / "review"
)
ANNOTATION_FIELDS = (
    "qid",
    "cohort",
    "review_status",
    "primary_label",
    "secondary_label",
    "evidence_note",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FirstStageReviewError(f"{path}: expected a JSON object")
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
                        raise FirstStageReviewError(
                            f"{member}:{line_number}: invalid _id"
                        )
                    if item_id in records:
                        raise FirstStageReviewError(
                            f"{member}:{line_number}: duplicate _id {item_id!r}"
                        )
                    records[item_id] = value
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile, KeyError) as exc:
        raise FirstStageReviewError(
            f"cannot read source member {member!r}: {exc}"
        ) from exc
    return records


def _annotation_template(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "qid": str(case["qid"]),
            "cohort": str(case["cohort"]),
            "review_status": "pending",
            "primary_label": "",
            "secondary_label": "",
            "evidence_note": "",
        }
        for case in cases
    ]


def _write_annotations(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ANNOTATION_FIELDS})


def _read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(ANNOTATION_FIELDS):
            raise FirstStageReviewError(
                f"{path}: expected columns {list(ANNOTATION_FIELDS)}, "
                f"got {reader.fieldnames}"
            )
        return [dict(row) for row in reader]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _escape_markdown(value: Any) -> str:
    return (
        " ".join(str(value).split())
        .replace("\\", "\\\\")
        .replace("|", "\\|")
    )


def _render_document(document: Mapping[str, Any]) -> str:
    rank = document.get("rank")
    relevance = document.get("relevance")
    context = []
    if rank is not None:
        context.append(f"rank {rank}")
    if relevance is not None:
        context.append(f"rel {relevance}")
    context.append(
        "query-token recall "
        f"{float(document['query_token_recall']):.2f}"
    )
    title = _escape_markdown(document.get("title") or "(untitled)")
    snippet = _escape_markdown(document["snippet"])
    return (
        f"- `{document['doc_id']}` ({', '.join(context)}): "
        f"**{title}** — {snippet}"
    )


def _render_review_guide(
    cases: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
) -> str:
    lines = [
        "# NFCorpus first-stage retrieval review guide",
        "",
        "This is a full census of the pre-declared 24 depth-recoverable and "
        "48 top-1000-miss queries. The ordering is deterministic. Evidence "
        "snippets are generated from the public NFCorpus archive and must stay "
        "under ignored `outputs/`; commit only the compact annotation file.",
        "",
        "## Label definitions",
        "",
        "| label | definition | experiment implication |",
        "|---|---|---|",
    ]
    for label, record in taxonomy["labels"].items():
        lines.append(
            f"| `{label}` | {_escape_markdown(record['definition'])} | "
            f"{_escape_markdown(record['experiment_implication'])} |"
        )
    lines.extend(
        [
            "",
            "Use one primary label only when the evidence note supports it. "
            "Use `needs_adjudication` with no primary label when the available "
            "text is insufficient or two explanations cannot be separated.",
            "",
            "## Review cases",
            "",
        ]
    )
    for case in cases:
        first_rank = case["first_relevant_rank"]
        lines.extend(
            [
                (
                    f"### {case['review_order']}. `{case['qid']}` — "
                    f"`{case['cohort']}`"
                ),
                "",
                f"- **Query:** {_escape_markdown(case['query'])}",
                (
                    "- **Query source:** "
                    f"`{_escape_markdown(case['query_source_type'])}`; "
                    f"{_escape_markdown(case['query_source_url'])}"
                ),
                (
                    "- **Coverage:** first relevant rank "
                    f"{first_rank if first_rank is not None else 'none'}; "
                    f"Recall@100 {float(case['recall@100']):.3f}; "
                    f"Recall@1000 {float(case['recall@1000']):.3f}; "
                    f"{case['n_positive_qrels']} positive qrels"
                ),
                (
                    "- **Qrel levels:** "
                    + ", ".join(
                        f"rel {level}: {count}"
                        for level, count in case[
                            "positive_qrels_by_relevance"
                        ].items()
                    )
                ),
                (
                    "- **Surface evidence:** query content tokens "
                    f"`{', '.join(case['query_content_tokens']) or '(none)'}`; "
                    "maximum positive-qrel query-token recall "
                    f"{float(case['max_positive_qrel_query_token_recall']):.2f}; "
                    "maximum top-result query-token recall "
                    f"{float(case['max_top_result_query_token_recall']):.2f}"
                ),
                (
                    "- **Diagnostic flags:** "
                    + (
                        ", ".join(f"`{flag}`" for flag in case["diagnostic_flags"])
                        or "(none)"
                    )
                ),
                "",
                "**Representative judged-relevant documents**",
                "",
            ]
        )
        lines.extend(
            _render_document(document)
            for document in case["representative_relevant_documents"]
        )
        lines.extend(["", "**BM25 top documents**", ""])
        lines.extend(
            _render_document(document) for document in case["top_bm25_documents"]
        )
        lines.extend(
            [
                "",
                "- **Review status:** `pending`",
                "- **Primary label:**",
                "- **Secondary label:**",
                "- **Evidence note:**",
                "",
            ]
        )
    return "\n".join(lines)


def _render_summary(
    summary: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> str:
    lines = [
        "# NFCorpus first-stage taxonomy review summary",
        "",
        "## Review status",
        "",
        (
            f"Reviewed **{summary['n_reviewed']} of {summary['n_cases']}** "
            f"cases ({float(summary['review_coverage']):.1%}). Label shares "
            "below use reviewed cases only; pending and adjudication cases are "
            "never folded into a residual category."
        ),
        "",
        "| status | cases |",
        "|---|---:|",
    ]
    for status in taxonomy["review_statuses"]:
        lines.append(f"| `{status}` | {summary['status_counts'].get(status, 0)} |")
    lines.extend(
        [
            "",
            "## Primary labels among reviewed cases",
            "",
            "| label | cases | share of reviewed |",
            "|---|---:|---:|",
        ]
    )
    denominator = int(summary["n_reviewed"])
    for label in taxonomy["labels"]:
        count = int(summary["primary_label_counts"].get(label, 0))
        share = count / denominator if denominator else 0.0
        lines.append(f"| `{label}` | {count} | {share:.1%} |")
    lines.extend(["", "## Cohort coverage", "", "| cohort | cases | reviewed | coverage |"])
    lines.append("|---|---:|---:|---:|")
    for cohort, record in summary["cohorts"].items():
        lines.append(
            f"| `{cohort}` | {record['n_cases']} | {record['n_reviewed']} | "
            f"{float(record['review_coverage']):.1%} |"
        )
    lines.extend(
        [
            "",
            "No causal or architectural conclusion should be drawn until each "
            "reported cohort has adequate review coverage and unresolved cases "
            "are adjudicated.",
            "",
            "## Objective qrel evidence in the 72-case census",
            "",
            "These counts describe the published relevance levels and do not "
            "replace human review.",
            "",
            "| evidence group | cases |",
            "|---|---:|",
        ]
    )
    for group, count in summary["objective_qrel_evidence_counts"].items():
        lines.append(f"| `{group}` | {count} |")
    lines.append("")
    source_diagnostics = summary.get("query_source_diagnostics")
    if isinstance(source_diagnostics, Mapping):
        lines.extend(
            [
                "## Objective coverage by query source",
                "",
                "These are deterministic retrieval diagnostics over all test "
                "queries, not human taxonomy labels.",
                "",
                "| source | queries | no relevant top 100 | rate | "
                "recoverable 101–1000 | miss top 1000 | macro R@100 | "
                "macro R@1000 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for source_type, record in source_diagnostics.items():
            lines.append(
                f"| `{source_type}` | {record['n_queries']} | "
                f"{record['no_relevant_top_100']} | "
                f"{float(record['no_relevant_top_100_rate']):.1%} | "
                f"{record['depth_recoverable_101_1000']} | "
                f"{record['miss_top_1000']} | "
                f"{float(record['macro_recall@100']):.4f} | "
                f"{float(record['macro_recall@1000']):.4f} |"
            )
        lines.append("")
    system_metrics = summary.get("query_source_system_metrics")
    if isinstance(system_metrics, Mapping):
        lines.extend(
            [
                "## Retrieval metrics by query source",
                "",
                "Both systems are evaluated from the fixed published runs. "
                "The cross-encoder only reorders the BM25 top 100.",
                "",
                "| source | system | queries | MRR@10 | nDCG@10 | R@100 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for source_type, systems in system_metrics.items():
            for system_name, metrics in systems.items():
                lines.append(
                    f"| `{source_type}` | `{system_name}` | "
                    f"{int(metrics['n_queries'])} | "
                    f"{float(metrics['mrr@10']):.4f} | "
                    f"{float(metrics['ndcg@10']):.4f} | "
                    f"{float(metrics['recall@100']):.4f} |"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional completed annotation CSV to validate and summarize.",
    )
    parser.add_argument(
        "--initialize-annotations",
        type=Path,
        help="Write a new pending annotation CSV; refuses to overwrite.",
    )
    args = parser.parse_args(argv)

    contract_path = _resolve(args.contract).resolve()
    taxonomy_path = _resolve(args.taxonomy).resolve()
    output_dir = _resolve(args.output_dir).resolve()
    try:
        contract_report = verify_retrieval_contract(
            contract_path,
            project_root=PROJECT_ROOT,
        )
        contract = _load_json_object(contract_path)
        taxonomy = load_review_taxonomy(taxonomy_path)
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
        qrels = read_qrels(
            PROJECT_ROOT / contract_report["inputs"]["qrels"]["path"],
            qrels_format=str(contract["qrels_format"]),
        )
        run = read_run_tsv(
            PROJECT_ROOT / contract_report["inputs"]["bm25_run"]["path"]
        )
        ce_run = read_run_tsv(
            PROJECT_ROOT / contract_report["inputs"]["ce_run"]["path"]
        )
        queries = {
            qid: str(record.get("text") or "")
            for qid, record in query_records.items()
        }
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
        cases = build_first_stage_review_cases(
            analysis["per_query"],
            run,
            qrels,
            corpus,
            taxonomy,
            query_records=query_records,
            rel_threshold=int(contract["binary_relevance_threshold"]),
        )
        template = _annotation_template(cases)
        if args.initialize_annotations is not None:
            initialize_path = _resolve(args.initialize_annotations).resolve()
            if initialize_path.exists():
                raise FirstStageReviewError(
                    f"refusing to overwrite existing annotations: {initialize_path}"
                )
            _write_annotations(initialize_path, template)
        annotations = (
            _read_annotations(_resolve(args.annotations).resolve())
            if args.annotations is not None
            else template
        )
        summary = validate_review_annotations(annotations, cases, taxonomy)
        summary["query_source_diagnostics"] = summarize_query_source_diagnostics(
            analysis["per_query"],
            query_records,
        )
        query_source_groups = partition_query_ids_by_source(
            sorted(qrels),
            query_records,
        )
        query_source_groups["nontopic_combined"] = sorted(
            qid
            for source_type, qids in query_source_groups.items()
            if source_type != "topic"
            for qid in qids
        )
        source_system_metrics: dict[str, dict[str, dict[str, float]]] = {}
        for source_type, qids in query_source_groups.items():
            qid_set = set(qids)
            source_qrels = {
                qid: judgments
                for qid, judgments in qrels.items()
                if qid in qid_set
            }
            source_system_metrics[source_type] = {}
            for system_name, system_run in (
                ("bm25", run),
                ("bm25_ce", ce_run),
            ):
                ranked_docs = {
                    qid: [doc_id for doc_id, _score in system_run[qid]]
                    for qid in qids
                }
                source_system_metrics[source_type][system_name] = (
                    evaluate_trec_retrieval(
                        ranked_docs,
                        source_qrels,
                        rel_threshold=int(
                            contract["binary_relevance_threshold"]
                        ),
                        ks_mrr=(10,),
                        ks_ndcg=(10,),
                        ks_recall=(100,),
                    )
                )
        summary["query_source_system_metrics"] = source_system_metrics
    except (
        FirstStageCoverageError,
        FirstStageReviewError,
        RetrievalContractError,
        QrelsFormatError,
        RunTsvFormatError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"NFCorpus review export failed: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "review_cases.jsonl", cases)
    _write_annotations(output_dir / "review_template.csv", template)
    _write_json(output_dir / "review_summary.json", summary)
    (output_dir / "review_guide.md").write_text(
        _render_review_guide(cases, taxonomy),
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "review_summary.md").write_text(
        _render_summary(summary, taxonomy),
        encoding="utf-8",
        newline="\n",
    )
    print(
        "NFCorpus review census exported: "
        f"{summary['n_cases']} cases, {summary['n_reviewed']} reviewed"
    )
    print(f"Wrote {output_dir.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the public NFCorpus video-query ablation release workflow."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from msmarco_genqa.evaluation.trec import evaluate_internal_trec_scope, read_qrels
from msmarco_genqa.reproducibility.nfcorpus_video_release import (
    BUNDLE_SCHEMA,
    REPRESENTATIONS,
    ReleaseArtifactError,
    _paired_comparisons,
    _qid_sha256,
    build_release_bundle,
    evaluate_release_bundle,
    fetch_release_bundle,
    verify_release_archive,
)
from msmarco_genqa.reproducibility.trec_release import (
    POINTER_SCHEMA,
    load_release_pointer,
)
from msmarco_genqa.reranking.io import read_run_tsv


def _write_run(path: Path, rows: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for qid in sorted(rows):
        for rank, doc_id in enumerate(rows[qid], start=1):
            lines.append(f"{qid}\tQ0\t{doc_id}\t{rank}\t{3-rank}.0\ttest")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_release_sources(root: Path) -> tuple[Path, Path]:
    qrels_path = root / "qrels.trec"
    qrels_path.write_text(
        "q1 0 d1 2\nq1 0 d2 1\nq2 0 d3 2\n",
        encoding="utf-8",
        newline="\n",
    )
    qrels = read_qrels(qrels_path, qrels_format="trec")
    bm25_rows = {
        "title": {"q1": ["miss1", "d1"], "q2": ["miss2", "d3"]},
        "description": {"q1": ["d1", "miss1"], "q2": ["d3", "miss2"]},
        "title_plus_description": {"q1": ["d1", "d2"], "q2": ["d3", "miss2"]},
    }

    results: dict[str, object] = {}
    runs: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {}
    for representation in REPRESENTATIONS:
        root_dir = root / "outputs" / "beir_nfcorpus_video" / representation
        bm25_path = root_dir / "bm25" / "run.tsv"
        rerank_path = root_dir / "cross_encoder_rerank" / "run.tsv"
        _write_run(bm25_path, bm25_rows[representation])
        _write_run(
            rerank_path,
            {
                qid: [doc_ids[0]]
                for qid, doc_ids in bm25_rows[representation].items()
            },
        )
        bm25_run = read_run_tsv(bm25_path)
        rerank_run = read_run_tsv(rerank_path)
        runs[(representation, "bm25")] = bm25_run
        runs[(representation, "bm25_ce")] = rerank_run
        bm25_metrics = evaluate_internal_trec_scope(bm25_run, qrels, rel_threshold=1)
        rerank_metrics = evaluate_internal_trec_scope(
            rerank_run,
            qrels,
            rel_threshold=1,
        )
        rerank_metrics.pop("recall@1000")
        results[representation] = {
            "bm25": bm25_metrics,
            "bm25_ce": rerank_metrics,
            "artifact_paths": {
                "bm25_run": bm25_path.relative_to(root).as_posix(),
                "rerank_run": rerank_path.relative_to(root).as_posix(),
            },
            "artifact_sha256": {
                "bm25_run": _sha256_file(bm25_path),
                "rerank_run": _sha256_file(rerank_path),
            },
        }

    paired = _paired_comparisons(runs, qrels, n_resamples=20, seed=7)
    description = results["description"]
    combined = results["title_plus_description"]
    assert isinstance(description, dict)
    assert isinstance(combined, dict)
    description["paired_vs_title"] = {
        "recall@100": paired["bm25"]["description_vs_title"]["metrics"][
            "recall@100"
        ]
    }
    combined["paired_vs_title"] = {
        "recall@100": paired["bm25"]["title_plus_description_vs_title"][
            "metrics"
        ]["recall@100"],
        "bm25_mrr@10": paired["bm25"]["title_plus_description_vs_title"][
            "metrics"
        ]["mrr@10"],
        "bm25_ndcg@10": paired["bm25"]["title_plus_description_vs_title"][
            "metrics"
        ]["ndcg@10"],
        "bm25_ce_mrr@10": paired["bm25_ce"][
            "title_plus_description_vs_title"
        ]["metrics"]["mrr@10"],
        "bm25_ce_ndcg@10": paired["bm25_ce"][
            "title_plus_description_vs_title"
        ]["metrics"]["ndcg@10"],
    }

    source_record = {
        "schema": "msmarco-genqa.table-artifact.v1",
        "query_set": {
            "n_queries": 2,
            "qid_sha256": _qid_sha256({"q1", "q2"}),
        },
        "model_revision": {
            "first_stage": "test-bm25",
            "reranker": "test-ce",
            "reranker_revision": "a" * 40,
            "candidate_depth": 1,
        },
        "experiment": {
            "git_commits": {"bm25": "a" * 40, "reranker": "b" * 40},
            "dataset": "beir/nfcorpus/test",
            "subset": "official_test_video",
            "bm25": {"retrieval_depth": 2},
            "reranker": {"depth": 1},
            "evaluation": {
                "bootstrap_resamples": 20,
                "bootstrap_seed": 7,
                "confidence_level": 0.95,
            },
        },
        "results": results,
    }
    source_path = (
        root
        / "reports"
        / "generated"
        / "artifacts"
        / "nfcorpus_video_query_representation.json"
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(source_record, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return source_path, qrels_path


def test_release_build_is_deterministic_and_text_only(tmp_path: Path) -> None:
    source_record, _qrels = _make_release_sources(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = build_release_bundle(
        source_record_path=source_record,
        project_root=tmp_path,
        output_archive=first,
    )
    second_result = build_release_bundle(
        source_record_path=source_record,
        project_root=tmp_path,
        output_archive=second,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_result["sha256"] == second_result["sha256"]
    verified = verify_release_archive(first)
    with zipfile.ZipFile(first) as archive:
        assert len(archive.namelist()) == 8
        assert set(archive.namelist()) == {
            "README.md",
            "bundle_manifest.json",
            "runs/nfcorpus_video_title_bm25.tsv",
            "runs/nfcorpus_video_title_bm25_ce.tsv",
            "runs/nfcorpus_video_description_bm25.tsv",
            "runs/nfcorpus_video_description_bm25_ce.tsv",
            "runs/nfcorpus_video_title_plus_description_bm25.tsv",
            "runs/nfcorpus_video_title_plus_description_bm25_ce.tsv",
        }
        contents = b"\n".join(archive.read(name) for name in archive.namelist())
    assert verified["manifest"]["schema"] == BUNDLE_SCHEMA
    assert verified["manifest"]["source_record"]["serialization"] == (
        "canonical-json-utf8-sorted-keys-compact"
    )
    assert b"query text" in contents
    assert b"C:\\Users" not in contents


def test_fetch_and_evaluate_recomputes_aggregates_and_bootstrap(
    tmp_path: Path,
) -> None:
    source_record, _qrels_path = _make_release_sources(tmp_path)
    qrels_path = tmp_path / "qrels.irds.tsv"
    qrels_path.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t2\n"
        "q1\td2\t1\n"
        "q2\td3\t2\n",
        encoding="utf-8",
        newline="\n",
    )
    archive = tmp_path / "release.zip"
    built = build_release_bundle(
        source_record_path=source_record,
        project_root=tmp_path,
        output_archive=archive,
    )
    pointer = {
        "schema": POINTER_SCHEMA,
        "artifact_id": built["manifest"]["artifact_id"],
        "release": {
            "repository": "GioiaZheng/msmarco-genqa",
            "tag": "test",
            "asset": archive.name,
        },
        "download": {
            "url": archive.as_uri(),
            "sha256": built["sha256"],
            "bytes": built["bytes"],
        },
    }
    pointer_path = tmp_path / "pointer.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    fetched = fetch_release_bundle(
        pointer_path=pointer_path,
        output_dir=tmp_path / "recovered",
    )
    report = evaluate_release_bundle(
        bundle_dir=fetched["bundle_dir"],
        output_dir=tmp_path / "evaluation",
        qrels_path=qrels_path,
    )

    assert report["verified"] is True
    assert report["aggregate_max_abs_delta"] == 0.0
    assert report["paired_max_abs_delta"] == 0.0
    assert report["candidate_set_checks"] == 6
    assert len(report["results"]) == 6
    assert "recall@1000" not in report["results"]["title_bm25_ce"]["metrics"]
    assert (tmp_path / "evaluation" / "metrics.json").is_file()
    table = (tmp_path / "evaluation" / "metrics.md").read_text(encoding="utf-8")
    assert "Title + description" in table


def test_release_build_rejects_run_hash_drift(tmp_path: Path) -> None:
    source_record, _qrels = _make_release_sources(tmp_path)
    run = (
        tmp_path
        / "outputs"
        / "beir_nfcorpus_video"
        / "title"
        / "bm25"
        / "run.tsv"
    )
    run.write_text(run.read_text(encoding="utf-8") + "q3 Q0 d3 1 1.0 test\n")

    with pytest.raises(ReleaseArtifactError, match="SHA-256 mismatch"):
        build_release_bundle(
            source_record_path=source_record,
            project_root=tmp_path,
            output_archive=tmp_path / "release.zip",
        )


def test_tracked_release_pointer_is_internally_consistent() -> None:
    project_root = Path(__file__).parents[1]
    pointer = load_release_pointer(
        project_root / "artifacts" / "nfcorpus_video_query_representation_v1.json"
    )
    release = pointer["release"]
    download = pointer["download"]

    assert release["repository"] == "GioiaZheng/msmarco-genqa"
    assert f"/{release['tag']}/{release['asset']}" in download["url"]
    assert pointer["contents"]["run_files"] == 6
    assert pointer["reproduce"] == {
        "command": "make reproduce-nfcorpus-video-eval",
        "requires_private_credentials": False,
        "rebuilds_corpus_indexes": False,
        "reruns_retrieval": False,
        "reruns_cross_encoder": False,
    }

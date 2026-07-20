"""Tests for the public BEIR release and recovery workflow."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from msmarco_genqa.evaluation.trec import evaluate_internal_trec_scope, read_qrels
from msmarco_genqa.reproducibility.beir_release import (
    BUNDLE_SCHEMA,
    ReleaseArtifactError,
    build_release_bundle,
    evaluate_release_bundle,
    fetch_release_bundle,
    verify_release_archive,
)
from msmarco_genqa.reproducibility.trec_release import (
    POINTER_SCHEMA,
    load_release_pointer,
    verify_release_archive as verify_trec_release_archive,
)
from msmarco_genqa.reranking.io import read_run_tsv


def _write_run(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{qid}\tQ0\t{doc_id}\t1\t1.0\ttest" for qid, doc_id in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _make_release_sources(root: Path) -> tuple[Path, dict[str, Path]]:
    qrels_paths: dict[str, Path] = {}
    datasets = {
        "nfcorpus": {
            "dataset_id": "beir/nfcorpus/test",
            "qrels": "q1 0 d1 2\nq1 0 dx 0\nq2 0 d2 1\n",
            "bm25": [("q1", "d1"), ("q2", "miss")],
            "bm25_ce": [("q1", "d1"), ("q2", "d2")],
        },
        "scifact": {
            "dataset_id": "beir/scifact/test",
            "qrels": "q3 0 d3 2\nq3 0 dy 1\nq4 0 d4 1\n",
            "bm25": [("q3", "miss"), ("q4", "d4")],
            "bm25_ce": [("q3", "d3"), ("q4", "d4")],
        },
    }
    results: dict[str, object] = {}
    for dataset_key, values in datasets.items():
        qrels_path = root / "qrels" / f"{dataset_key}.qrels"
        qrels_path.parent.mkdir(parents=True, exist_ok=True)
        qrels_path.write_text(str(values["qrels"]), encoding="utf-8", newline="\n")
        qrels_paths[dataset_key] = qrels_path
        qrels = read_qrels(qrels_path, qrels_format="trec")

        output_root = root / "outputs" / f"beir_{dataset_key}_test"
        bm25_path = output_root / "bm25" / "run.tsv"
        rerank_path = output_root / "cross_encoder_rerank" / "run.tsv"
        _write_run(bm25_path, values["bm25"])
        _write_run(rerank_path, values["bm25_ce"])
        bm25_metrics = evaluate_internal_trec_scope(
            read_run_tsv(bm25_path), qrels, rel_threshold=1
        )
        rerank_metrics = evaluate_internal_trec_scope(
            read_run_tsv(rerank_path), qrels, rel_threshold=1
        )
        rerank_metrics.pop("recall@1000")
        results[dataset_key] = {
            "dataset_id": values["dataset_id"],
            "judged_topics": 2,
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

    source_record = {
        "schema": "msmarco-genqa.table-artifact.v1",
        "experiment": {
            "git_commits": {
                "nfcorpus_bm25": "a" * 40,
                "nfcorpus_bm25_ce": "a" * 40,
                "scifact_bm25": "a" * 40,
                "scifact_bm25_ce": "b" * 40,
            }
        },
        "results": results,
    }
    record_path = root / "reports" / "generated" / "artifacts" / "beir.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(source_record, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return record_path, qrels_paths


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    assert first_result["sha256"] == second_result["sha256"]
    assert first.read_bytes() == second.read_bytes()
    verified = verify_release_archive(first)
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == {
            "README.md",
            "bundle_manifest.json",
            "runs/beir_nfcorpus_bm25.tsv",
            "runs/beir_nfcorpus_bm25_ce.tsv",
            "runs/beir_scifact_bm25.tsv",
            "runs/beir_scifact_bm25_ce.tsv",
        }
        contents = b"\n".join(archive.read(name) for name in archive.namelist())
    assert verified["manifest"]["schema"] == BUNDLE_SCHEMA
    assert b"document text" in contents
    assert b"C:\\Users" not in contents


def test_trec_verifier_rejects_beir_schema(tmp_path: Path) -> None:
    source_record, _qrels = _make_release_sources(tmp_path)
    archive = tmp_path / "release.zip"
    build_release_bundle(
        source_record_path=source_record,
        project_root=tmp_path,
        output_archive=archive,
    )

    with pytest.raises(ReleaseArtifactError, match="trec-dl-release-bundle"):
        verify_trec_release_archive(archive)


def test_fetch_and_evaluate_without_private_credentials(tmp_path: Path) -> None:
    source_record, qrels_paths = _make_release_sources(tmp_path)
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
        qrels_paths=qrels_paths,
    )

    assert report["verified"] is True
    assert report["max_abs_delta"] == 0.0
    assert set(report["results"]) == {
        "nfcorpus_bm25",
        "nfcorpus_bm25_ce",
        "scifact_bm25",
        "scifact_bm25_ce",
    }
    assert "recall@1000" not in report["results"]["nfcorpus_bm25_ce"]["metrics"]
    assert report["results"]["scifact_bm25"]["metrics"]["n_queries"] == 2
    assert (tmp_path / "evaluation" / "metrics.json").is_file()
    table = (tmp_path / "evaluation" / "metrics.md").read_text(encoding="utf-8")
    assert "Reproduced BEIR" in table
    assert "Max abs. delta" in table


def test_release_build_rejects_run_hash_drift(tmp_path: Path) -> None:
    source_record, _qrels = _make_release_sources(tmp_path)
    run = tmp_path / "outputs" / "beir_nfcorpus_test" / "bm25" / "run.tsv"
    run.write_text(run.read_text(encoding="utf-8") + "q3 Q0 d3 1 1.0 test\n")

    with pytest.raises(ReleaseArtifactError, match="SHA-256 mismatch"):
        build_release_bundle(
            source_record_path=source_record,
            project_root=tmp_path,
            output_archive=tmp_path / "release.zip",
        )


def test_tracked_release_pointer_is_internally_consistent() -> None:
    project_root = Path(__file__).parents[1]
    pointer = load_release_pointer(project_root / "artifacts" / "beir_cross_domain_v1.json")
    release = pointer["release"]
    download = pointer["download"]

    assert release["repository"] == "GioiaZheng/msmarco-genqa"
    assert f"/{release['tag']}/{release['asset']}" in download["url"]
    assert pointer["reproduce"] == {
        "command": "make reproduce-beir-eval",
        "requires_private_credentials": False,
        "rebuilds_corpus_indexes": False,
        "reruns_cross_encoder": False,
    }

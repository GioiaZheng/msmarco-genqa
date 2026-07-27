from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.run_reranker import (
    _validate_query_representation_args as validate_reranker_representation_args,
    _validate_upstream_query_representation,
    parse_args as parse_reranker_args,
)
from experiments.run_retrieval import (
    _index_fingerprint,
    _select_video_query_cohort,
    _validate_query_representation_args,
    _validate_representation_resume,
    parse_args,
)
from msmarco_genqa.data.benchmark import BenchmarkQueries, get_benchmark_spec
from msmarco_genqa.data.nfcorpus_video import (
    NFCorpusVideoContractError,
    load_nfcorpus_video_query_representation,
    validate_frozen_title_metrics,
    validate_frozen_title_reranker_metrics,
    write_nfcorpus_video_query_artifacts,
)
from msmarco_genqa.retrieval.bm25 import BM25Retriever


def _digest(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def _canonical_qid_hash(query_ids: list[str]) -> str:
    text = "".join(f"{query_id}\n" for query_id in sorted(query_ids))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_record_hash(
    titles: dict[str, str],
    descriptions: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(titles):
        row = {
            "description": descriptions[query_id],
            "qid": query_id,
            "title": titles[query_id],
        }
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(f"{encoded}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_tar_member(
    archive: tarfile.TarFile,
    name: str,
    rows: dict[str, str],
) -> None:
    payload = "".join(
        f"{query_id}\t{value}\n" for query_id, value in rows.items()
    ).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def _make_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    titles = {
        "q1": "Vitamin B12 — Benefits ?",
        "q2": "Organic foods : safety",
    }
    descriptions = {
        "q1": "  A bounded   description about B12.  ",
        "q2": "Evidence about organic foods.",
    }
    archive_path = tmp_path / "cache" / "nfcorpus.tar.gz"
    archive_path.parent.mkdir(parents=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        _write_tar_member(
            archive,
            "nfcorpus/test.vid-titles.queries",
            titles,
        )
        _write_tar_member(
            archive,
            "nfcorpus/test.vid-desc.queries",
            descriptions,
        )

    contract = {
        "schema": "msmarco-genqa.nfcorpus-video-query-representation.v1",
        "inputs": {
            "official_nfcorpus_v1": {
                "dataset_id": "nfcorpus/test/video",
                "path": "cache/nfcorpus.tar.gz",
                "url": "https://example.test/nfcorpus.tar.gz",
                "bytes": archive_path.stat().st_size,
                "md5": _digest(archive_path, "md5"),
                "sha256": _digest(archive_path, "sha256"),
            }
        },
        "cohort": {
            "n_queries": 2,
            "qid_sha256": _canonical_qid_hash(list(titles)),
            "official_query_records_sha256": _canonical_record_hash(
                titles,
                descriptions,
            ),
        },
        "query_representations": {
            "title": {},
            "description": {},
            "title_plus_description": {},
        },
        "frozen_title_baseline": {
            "bm25": {
                "mrr@10": 0.5,
                "ndcg@10": 0.4,
                "recall@100": 0.3,
                "recall@1000": 0.6,
            },
            "bm25_ce": {
                "mrr@10": 0.6,
                "ndcg@10": 0.5,
                "recall@100": 0.3,
            },
            "positive_score_bm25": {
                "positive_score_recall@100": 0.3,
                "positive_score_recall@1000": 0.6,
            },
            "deterministic_tie_bm25": {
                "mrr@10": 0.5,
                "ndcg@10": 0.4,
                "recall@100": 0.3,
                "recall@1000": 0.6,
            },
            "deterministic_tie_bm25_ce": {
                "mrr@10": 0.6,
                "ndcg@10": 0.5,
                "recall@100": 0.3,
            },
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline_queries = {
        "q1": "Vitamin B12 - Benefits?",
        "q2": "Organic foods: safety",
        "other": "not in the video cohort",
    }
    return contract_path, baseline_queries


@pytest.mark.parametrize(
    ("representation", "expected"),
    [
        (
            "title",
            {
                "q1": "Vitamin B12 - Benefits?",
                "q2": "Organic foods: safety",
            },
        ),
        (
            "description",
            {
                "q1": "A bounded description about B12.",
                "q2": "Evidence about organic foods.",
            },
        ),
        (
            "title_plus_description",
            {
                "q1": "Vitamin B12 - Benefits?\nA bounded description about B12.",
                "q2": "Organic foods: safety\nEvidence about organic foods.",
            },
        ),
    ],
)
def test_loader_constructs_only_predeclared_video_queries(
    tmp_path: Path,
    representation: str,
    expected: dict[str, str],
) -> None:
    contract_path, baseline_queries = _make_fixture(tmp_path)

    bundle = load_nfcorpus_video_query_representation(
        baseline_queries,
        representation=representation,
        contract_path=contract_path,
        project_root=tmp_path,
        download_if_missing=False,
    )

    assert bundle.queries == expected
    assert bundle.summary["n_queries"] == 2
    assert bundle.summary["title_alignment"] == {"matched": 2, "mismatched": 0}
    assert bundle.summary["leakage_boundary"]["excluded"] == [
        "qrels",
        "corpus_documents",
        "ranked_outputs",
        "metrics",
        "manual_rewrites",
    ]


def test_loader_rejects_archive_hash_drift(tmp_path: Path) -> None:
    contract_path, baseline_queries = _make_fixture(tmp_path)
    archive_path = tmp_path / "cache" / "nfcorpus.tar.gz"
    archive_path.write_bytes(archive_path.read_bytes() + b"drift")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["inputs"]["official_nfcorpus_v1"]["bytes"] = archive_path.stat().st_size
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(NFCorpusVideoContractError, match="MD5 drift"):
        load_nfcorpus_video_query_representation(
            baseline_queries,
            representation="title",
            contract_path=contract_path,
            project_root=tmp_path,
            download_if_missing=False,
        )


def test_loader_rejects_beir_title_mismatch(tmp_path: Path) -> None:
    contract_path, baseline_queries = _make_fixture(tmp_path)
    baseline_queries["q2"] = "unrelated query"

    with pytest.raises(NFCorpusVideoContractError, match="do not align"):
        load_nfcorpus_video_query_representation(
            baseline_queries,
            representation="description",
            contract_path=contract_path,
            project_root=tmp_path,
            download_if_missing=False,
        )


def test_artifacts_and_title_reproduction_guard(tmp_path: Path) -> None:
    contract_path, baseline_queries = _make_fixture(tmp_path)
    bundle = load_nfcorpus_video_query_representation(
        baseline_queries,
        representation="title",
        contract_path=contract_path,
        project_root=tmp_path,
        download_if_missing=False,
    )

    summary, paths = write_nfcorpus_video_query_artifacts(
        bundle,
        tmp_path / "outputs" / "query_representation",
    )
    guard = validate_frozen_title_metrics(
        bundle,
        {
            "mrr@10": 0.5,
            "ndcg@10": 0.4,
            "recall@100": 0.3,
            "recall@1000": 0.6,
        },
        positive_score_metrics={
            "positive_score_recall@100": 0.3,
            "positive_score_recall@1000": 0.6,
        },
    )

    assert summary["representation"] == "title"
    assert len(paths) == 2
    assert guard["passed"] is True
    assert validate_frozen_title_reranker_metrics(
        bundle,
        {
            "mrr@10": 0.5,
            "ndcg@10": 0.4,
            "recall@100": 0.3,
        },
        {
            "mrr@10": 0.6,
            "ndcg@10": 0.5,
            "recall@100": 0.3,
        },
    )["passed"] is True
    with pytest.raises(NFCorpusVideoContractError, match="metric drift"):
        validate_frozen_title_metrics(
            bundle,
            {
                "mrr@10": 0.4,
                "ndcg@10": 0.4,
                "recall@100": 0.3,
                "recall@1000": 0.6,
            },
            positive_score_metrics={
                "positive_score_recall@100": 0.3,
                "positive_score_recall@1000": 0.6,
            },
        )


def test_bm25_deterministic_ties_score_then_doc_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def retrieve(self, _tokens, *, k, **_kwargs):
            assert k == 4
            return (
                np.asarray([[0, 1, 2, 3]]),
                np.asarray([[0.0, 1.0, 1.0, 0.0]]),
            )

    monkeypatch.setitem(
        sys.modules,
        "bm25s",
        SimpleNamespace(tokenize=lambda queries, **_kwargs: queries),
    )
    retriever = BM25Retriever(
        corpus_texts=[],
        doc_ids=["d2", "d1", "d4", "d3"],
    )
    retriever._bm25 = FakeBackend()

    scores, doc_ids = retriever.retrieve_batch(
        ["query"],
        k=3,
        deterministic_ties=True,
    )

    assert doc_ids == [["d1", "d4", "d2"]]
    assert scores.tolist() == [[1.0, 1.0, 0.0]]


def test_index_fingerprint_is_order_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "b.bin").write_bytes(b"two")
    (index_dir / "a.bin").write_bytes(b"one")
    monkeypatch.setattr(
        "experiments.run_retrieval.PROJECT_ROOT",
        tmp_path,
    )

    first = _index_fingerprint(index_dir)
    second = _index_fingerprint(index_dir)

    assert first == second
    assert first["file_count"] == 2
    assert first["bytes"] == 6
    assert first["path"] == "index"


def test_runner_requires_isolated_nfcorpus_output(tmp_path: Path) -> None:
    cfg = {"query_transform": {"method": "none"}}

    missing_output = parse_args(
        [
            "--dataset",
            "beir/nfcorpus/test",
            "--query-representation",
            "title",
        ]
    )
    with pytest.raises(SystemExit, match="explicit --output-dir"):
        _validate_query_representation_args(missing_output, cfg)

    wrong_dataset = parse_args(
        [
            "--dataset",
            "beir/scifact/test",
            "--query-representation",
            "title",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    with pytest.raises(SystemExit, match="restricted"):
        _validate_query_representation_args(wrong_dataset, cfg)

    reranker_missing_paths = parse_reranker_args(
        [
            "--dataset",
            "beir/nfcorpus/test",
            "--query-representation",
            "title",
        ]
    )
    with pytest.raises(SystemExit, match="explicit --input-run and --output-dir"):
        validate_reranker_representation_args(reranker_missing_paths, cfg)


def test_cohort_selection_happens_after_query_construction(tmp_path: Path) -> None:
    contract_path, baseline_queries = _make_fixture(tmp_path)
    bundle = load_nfcorpus_video_query_representation(
        baseline_queries,
        representation="description",
        contract_path=contract_path,
        project_root=tmp_path,
        download_if_missing=False,
    )
    benchmark = BenchmarkQueries(
        spec=get_benchmark_spec("beir/nfcorpus/test"),
        queries=baseline_queries,
        qrels={"q1": {"d1"}, "q2": {"d2"}, "other": {"d3"}},
        graded_qrels={
            "q1": {"d1": 1},
            "q2": {"d2": 2},
            "other": {"d3": 1},
        },
    )

    selected = _select_video_query_cohort(benchmark, bundle)

    assert selected.queries == bundle.queries
    assert selected.qrels == {"q1": {"d1"}, "q2": {"d2"}}
    assert selected.graded_qrels == {"q1": {"d1": 1}, "q2": {"d2": 2}}


def test_resume_rejects_mixed_query_representation(tmp_path: Path) -> None:
    contract_path, baseline_queries = _make_fixture(tmp_path)
    bundle = load_nfcorpus_video_query_representation(
        baseline_queries,
        representation="title",
        contract_path=contract_path,
        project_root=tmp_path,
        download_if_missing=False,
    )
    output_dir = tmp_path / "run"
    representation_dir = output_dir / "query_representation"
    representation_dir.mkdir(parents=True)
    (output_dir / "run.tsv").write_text("q1\tQ0\td1\t1\t1.0\tbm25\n", encoding="utf-8")
    stale = dict(bundle.summary)
    stale["representation"] = "description"
    (representation_dir / "summary.json").write_text(
        json.dumps(stale),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="representation differs"):
        _validate_representation_resume(output_dir, bundle, resume=True)


def test_reranker_requires_matching_upstream_representation(tmp_path: Path) -> None:
    contract_path, baseline_queries = _make_fixture(tmp_path)
    bundle = load_nfcorpus_video_query_representation(
        baseline_queries,
        representation="description",
        contract_path=contract_path,
        project_root=tmp_path,
        download_if_missing=False,
    )
    upstream = {"query_representation": dict(bundle.summary)}

    _validate_upstream_query_representation(bundle, upstream)
    upstream["query_representation"]["effective_queries_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="effective_queries_sha256"):
        _validate_upstream_query_representation(bundle, upstream)

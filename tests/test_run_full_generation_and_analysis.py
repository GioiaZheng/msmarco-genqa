"""Tests for ``scripts/run_full_generation_and_analysis.py``.

We cover the pure pieces — completeness checks, qid-pool mismatch
detection, and preflight summary computation. The phase functions
themselves call subprocesses, which we don't exercise here (the
``--dry-run`` mode covers wiring end-to-end).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The driver uses ``@dataclass``, which calls ``sys.modules[cls.__module__]``
# during class creation; this requires the module to be registered in
# ``sys.modules`` BEFORE ``exec_module`` runs. (Plain ``module_from_spec``
# alone doesn't register it.)
_module_name = "run_full_generation_and_analysis"
_spec = importlib.util.spec_from_file_location(
    _module_name,
    PROJECT_ROOT / "scripts" / "run_full_generation_and_analysis.py",
)
driver = importlib.util.module_from_spec(_spec)
sys.modules[_module_name] = driver
_spec.loader.exec_module(driver)


# --------------------------------------------------------------------------- #
# Output-completeness checks (idempotency basis)
# --------------------------------------------------------------------------- #


def _write_predictions(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "query_id": f"q{i}",
                "query": f"text {i}",
                "prediction": "p",
                "references": ["r"],
                "top_doc_ids": ["d1", "d2", "d3"],
                "passages": ["", "", ""],
            }) + "\n")


def _write_metrics(path: Path, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        path.write_text(json.dumps({"metrics": {"rouge-l": 0.5, "bleu": 0.3}}))
    else:
        path.write_text("{ not json")


class TestIsValidMetricsJson:
    def test_ok(self, tmp_path):
        p = tmp_path / "metrics.json"
        _write_metrics(p)
        assert driver._is_valid_metrics_json(p) is True

    def test_missing(self, tmp_path):
        assert driver._is_valid_metrics_json(tmp_path / "absent.json") is False

    def test_malformed_json(self, tmp_path):
        p = tmp_path / "metrics.json"
        _write_metrics(p, valid=False)
        assert driver._is_valid_metrics_json(p) is False

    def test_metrics_dict_missing_known_keys(self, tmp_path):
        p = tmp_path / "metrics.json"
        p.write_text(json.dumps({"metrics": {"random_unknown_key": 0.5}}))
        assert driver._is_valid_metrics_json(p) is False


class TestGenerationOutputsComplete:
    def test_complete_no_expected_qids(self, tmp_path):
        _write_predictions(tmp_path / "predictions.jsonl", 3)
        _write_metrics(tmp_path / "metrics.json")
        assert driver.generation_outputs_complete(tmp_path, expected_qids=None) is True

    def test_complete_with_expected_qids_match(self, tmp_path):
        _write_predictions(tmp_path / "predictions.jsonl", 5)
        _write_metrics(tmp_path / "metrics.json")
        assert driver.generation_outputs_complete(tmp_path, expected_qids=5) is True

    def test_incomplete_when_predictions_row_count_mismatches(self, tmp_path):
        _write_predictions(tmp_path / "predictions.jsonl", 3)
        _write_metrics(tmp_path / "metrics.json")
        # Asked for 5, got 3 — should treat as incomplete (re-run).
        assert driver.generation_outputs_complete(tmp_path, expected_qids=5) is False

    def test_missing_predictions(self, tmp_path):
        _write_metrics(tmp_path / "metrics.json")
        assert driver.generation_outputs_complete(tmp_path, expected_qids=None) is False

    def test_missing_metrics(self, tmp_path):
        _write_predictions(tmp_path / "predictions.jsonl", 1)
        assert driver.generation_outputs_complete(tmp_path, expected_qids=None) is False

    def test_malformed_metrics(self, tmp_path):
        _write_predictions(tmp_path / "predictions.jsonl", 1)
        _write_metrics(tmp_path / "metrics.json", valid=False)
        assert driver.generation_outputs_complete(tmp_path, expected_qids=None) is False


def test_analysis_outputs_complete(tmp_path):
    assert driver.analysis_outputs_complete(tmp_path) is False
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "report.md").write_text("# report")
    assert driver.analysis_outputs_complete(tmp_path) is True


# --------------------------------------------------------------------------- #
# qid-pool mismatch (fail-loudly)
# --------------------------------------------------------------------------- #


def _write_preds_with_qids(dir_: Path, qids: list[str]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    with open(dir_ / "predictions.jsonl", "w") as f:
        for q in qids:
            f.write(json.dumps({
                "query_id": q, "query": "x", "prediction": "p",
                "references": ["r"], "top_doc_ids": ["d"], "passages": [""],
            }) + "\n")


def test_assert_matched_qid_pools_ok(tmp_path):
    _write_preds_with_qids(tmp_path / "bm25", ["q1", "q2", "q3"])
    _write_preds_with_qids(tmp_path / "rerank", ["q3", "q1", "q2"])  # order shouldn't matter
    # No exception ⇒ pools match.
    driver.assert_matched_qid_pools(tmp_path / "bm25", tmp_path / "rerank")


def test_assert_matched_qid_pools_mismatch_raises(tmp_path):
    _write_preds_with_qids(tmp_path / "bm25", ["q1", "q2", "q3"])
    _write_preds_with_qids(tmp_path / "rerank", ["q1", "q2", "q4"])
    with pytest.raises(SystemExit) as exc:
        driver.assert_matched_qid_pools(tmp_path / "bm25", tmp_path / "rerank")
    msg = str(exc.value)
    assert "qid-pool mismatch" in msg
    assert "bm25=3" in msg and "rerank=3" in msg
    assert "shared=2" in msg


# --------------------------------------------------------------------------- #
# Preflight summary
# --------------------------------------------------------------------------- #


def _write_run_tsv(path: Path, qids: list[str], top_k: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in qids:
            for rank in range(1, top_k + 1):
                f.write(f"{q}\tQ0\td_{rank}\t{rank}\t{1.0 / rank:.4f}\tsys\n")


def test_count_qids_in_run(tmp_path):
    p = tmp_path / "run.tsv"
    _write_run_tsv(p, ["q1", "q2", "q3"])
    assert driver._count_qids_in_run(p) == 3


def test_count_qids_in_run_missing_file(tmp_path):
    assert driver._count_qids_in_run(tmp_path / "absent.tsv") == 0


def test_preflight_summary_full(tmp_path):
    rerank = tmp_path / "outputs/rerank/run.tsv"
    bm25 = tmp_path / "outputs/bm25/run.tsv"
    _write_run_tsv(rerank, [f"q{i}" for i in range(10)])
    _write_run_tsv(bm25, [f"q{i}" for i in range(20)])

    cfg = driver.DriverConfig(
        reranker_run=rerank,
        reranker_manifest=rerank.parent / "manifest.json",
        bm25_run=bm25,
        bm25_out=tmp_path / "out/bm25",
        rerank_out=tmp_path / "out/rerank",
        analysis_out=tmp_path / "out/analysis",
        log_dir=tmp_path / "logs",
        n_eval_queries=99999,
        expected_qids=10,
        rerank_top_k=100,
        retrieval_source_bm25="bm25",
        retrieval_source_reranked="reranked",
        seconds_per_query_estimate=0.30,
        require_full_manifest=True,
    )
    s = driver.preflight_summary(cfg)
    assert s["rerank_qids_on_disk"] == 10
    assert s["bm25_qids_on_disk"] == 20
    # Eligible pool is min of the two coverages.
    assert s["eligible_pool_estimate"] == 10
    # n_eval clamps to the pool size.
    assert s["n_eval_queries_after_clamp"] == 10
    # 2 generation passes × 10 queries × 0.3s = 6s = 0.1 min.
    assert s["expected_wall_clock_minutes"] == 0.1
    assert s["rerank_top_k"] == 100


def test_preflight_summary_with_eval_cap(tmp_path):
    rerank = tmp_path / "outputs/rerank/run.tsv"
    bm25 = tmp_path / "outputs/bm25/run.tsv"
    _write_run_tsv(rerank, [f"q{i}" for i in range(100)])
    _write_run_tsv(bm25, [f"q{i}" for i in range(100)])
    cfg = driver.DriverConfig(
        reranker_run=rerank,
        reranker_manifest=rerank.parent / "manifest.json",
        bm25_run=bm25,
        bm25_out=tmp_path / "out/bm25",
        rerank_out=tmp_path / "out/rerank",
        analysis_out=tmp_path / "out/analysis",
        log_dir=tmp_path / "logs",
        n_eval_queries=5,   # explicit cap
        expected_qids=100,
        rerank_top_k=100,
        retrieval_source_bm25="bm25",
        retrieval_source_reranked="reranked",
        seconds_per_query_estimate=0.30,
        require_full_manifest=True,
    )
    s = driver.preflight_summary(cfg)
    assert s["n_eval_queries_after_clamp"] == 5

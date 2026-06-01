from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


_CHECK_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_headline_metrics.py"
_spec = importlib.util.spec_from_file_location("check_headline_metrics", _CHECK_PATH)
check_headline_metrics = importlib.util.module_from_spec(_spec)
sys.modules["check_headline_metrics"] = check_headline_metrics
_spec.loader.exec_module(check_headline_metrics)  # type: ignore[union-attr]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_metadata_matches_results_summary():
    metadata = json.loads((PROJECT_ROOT / "metadata.json").read_text(encoding="utf-8"))
    results = (PROJECT_ROOT / "RESULTS.md").read_text(encoding="utf-8")

    parsed = check_headline_metrics.collect_results_metrics(results)

    assert check_headline_metrics.compare_headline_metrics(metadata, parsed) == []


def test_collect_results_metrics_extracts_expected_headlines():
    parsed = check_headline_metrics.collect_results_metrics(
        (PROJECT_ROOT / "RESULTS.md").read_text(encoding="utf-8")
    )

    assert parsed["bm25_to_reranked_t5_small"]["token_f1_delta"] == 0.1711
    assert parsed["bm25_to_reranked_t5_small"]["rouge_l_delta"] == 0.1742
    assert parsed["bm25_to_reranked_t5_small"]["paired_bootstrap_ci_token_f1"] == [
        0.1632,
        0.1789,
    ]
    assert parsed["dense_retrieval_sample"]["bm25_sample_mrr_at_10"] == 0.6948
    assert parsed["dense_retrieval_sample"]["dense_mrr_at_10"] == 0.8830
    assert parsed["dense_retrieval_sample"]["cross_encoder_mrr_at_10"] == 0.9304


def test_metric_comparison_reports_drift():
    metadata = json.loads((PROJECT_ROOT / "metadata.json").read_text(encoding="utf-8"))
    results = check_headline_metrics.collect_results_metrics(
        (PROJECT_ROOT / "RESULTS.md").read_text(encoding="utf-8")
    )
    drifted = copy.deepcopy(metadata)
    drifted["headline_metrics"]["bm25_to_reranked_t5_small"]["token_f1_delta"] = 0.0

    failures = check_headline_metrics.compare_headline_metrics(drifted, results)

    assert failures == [
        "bm25_to_reranked_t5_small.token_f1_delta: metadata=0.0000, RESULTS.md=0.1711"
    ]

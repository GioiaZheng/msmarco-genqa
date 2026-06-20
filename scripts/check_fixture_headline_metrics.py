"""Check deterministic fixture headline metrics against committed goldens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from msmarco_genqa.data.trec_dl import load_trec_dl_from_files
from msmarco_genqa.evaluation.generation import token_f1
from msmarco_genqa.evaluation.grounding import lexical_grounding
from msmarco_genqa.evaluation.retrieval import evaluate_retrieval
from msmarco_genqa.evaluation.retrieval_report import read_run_doc_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "tests/fixtures/headline_regression/config.json"
DEFAULT_GOLDEN = PROJECT_ROOT / "tests/fixtures/headline_regression/golden.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _mean(values: list[float], *, name: str) -> float:
    if not values:
        raise ValueError(f"{name} has no examples")
    return sum(values) / len(values)


def compute_observed(config: dict[str, Any]) -> dict[str, float]:
    resolved = config["resolved_config"]

    retrieval_cfg = resolved["retrieval"]
    retrieval_bundle = load_trec_dl_from_files(
        _resolve_project_path(retrieval_cfg["queries_path"]),
        _resolve_project_path(retrieval_cfg["qrels_path"]),
        year=2019,
        rel_threshold=int(retrieval_cfg["rel_threshold"]),
    )
    run_doc_ids = read_run_doc_ids(_resolve_project_path(retrieval_cfg["run_path"]))
    retrieval_metrics = evaluate_retrieval(
        run_doc_ids,
        retrieval_bundle.qrels,
        ks_mrr=tuple(int(k) for k in retrieval_cfg["ks_mrr"]),
        ks_ndcg=(),
        ks_recall=(),
    )

    examples = config["generation_examples"]
    token_f1_scores = [
        token_f1(example["prediction"], example["references"])
        for example in examples
    ]
    lexical_grounding_scores = [
        lexical_grounding(example["prediction"], example["passages"])
        for example in examples
    ]

    return {
        "retrieval.mrr@10": float(retrieval_metrics["mrr@10"]),
        "generation.mean_token_f1": _mean(token_f1_scores, name="generation_examples"),
        "grounding.mean_lexical_content_token_grounding": _mean(
            lexical_grounding_scores,
            name="generation_examples",
        ),
    }


def compare_to_golden(
    observed: dict[str, float],
    golden: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for name, spec in sorted(golden["metrics"].items()):
        if name not in observed:
            failures.append(f"{name}: missing observed value")
            continue
        expected = float(spec["expected"])
        tolerance = float(spec["tolerance"])
        actual = float(observed[name])
        delta = actual - expected
        if abs(delta) > tolerance:
            failures.append(
                f"{name}: expected {expected:.12g} +/- {tolerance:.3g}, "
                f"observed {actual:.12g}, delta {delta:+.12g}"
            )
    unexpected = sorted(set(observed) - set(golden["metrics"]))
    for name in unexpected:
        failures.append(f"{name}: observed metric has no golden entry")
    return failures


def observed_payload(config: dict[str, Any], observed: dict[str, float]) -> dict[str, Any]:
    return {
        "schema": "msmarco-genqa.headline-regression-observed.v1",
        "source_config_schema": config.get("schema"),
        "seed": config.get("seed"),
        "metrics": {
            name: {"expected": value, "tolerance": 1e-12}
            for name, value in sorted(observed.items())
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare deterministic fixture headline metrics against committed goldens.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument(
        "--dump-observed",
        action="store_true",
        help="Print observed metrics as JSON for review; do not overwrite goldens.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_json(args.config)
    observed = compute_observed(config)

    if args.dump_observed:
        print(json.dumps(observed_payload(config, observed), indent=2, sort_keys=True))
        return 0

    golden = _load_json(args.golden)
    failures = compare_to_golden(observed, golden)
    if failures:
        print("Fixture headline metric regression check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "Run `python scripts/check_fixture_headline_metrics.py --dump-observed` "
            "to inspect proposed values before editing the golden file.",
            file=sys.stderr,
        )
        return 1

    print("Fixture headline metric regression check passed")
    for name, value in sorted(observed.items()):
        print(f"  {name}: {value:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

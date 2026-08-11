"""Verify a frozen first-stage data and metric contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from msmarco_genqa.evaluation.retrieval_contract import (
    RetrievalContractError,
    verify_retrieval_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "nfcorpus_first_stage_contract.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "nfcorpus_first_stage"
    / "data_metric_contract.json"
)
DEFAULT_LABEL = "NFCorpus"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    args = parser.parse_args(argv)

    try:
        report = verify_retrieval_contract(
            _resolve(args.contract),
            project_root=PROJECT_ROOT,
        )
    except RetrievalContractError as exc:
        print(f"Contract verification failed: {exc}", file=sys.stderr)
        return 1

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    metrics = report["bm25_metrics"]
    print(
        f"{args.label} contract verified: "
        f"{report['scope']['test_queries']} queries, "
        f"BM25 {report['scope']['bm25']['row_count']} rows/depth "
        f"{report['scope']['bm25']['depth']}, "
        f"CE {report['scope']['ce']['row_count']} rows/depth "
        f"{report['scope']['ce']['depth']}"
    )
    print(
        "BM25 metrics: "
        f"MRR@10={metrics['mrr@10']:.10f}, "
        f"nDCG@10={metrics['ndcg@10']:.10f}, "
        f"Recall@100={metrics['recall@100']:.10f}, "
        f"Recall@1000={metrics['recall@1000']:.10f}"
    )
    print(f"Maximum absolute metric delta: {report['max_abs_delta']:.3g}")
    print(f"Wrote {output.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

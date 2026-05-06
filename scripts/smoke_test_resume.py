"""Smoke test for the chunked + checkpoint/resume path in run_retrieval.py.

Bypasses ``ir_datasets`` so no download is needed. Validates:

1. Fresh run: ``run.tsv`` is truncated, all qids written.
2. Simulated kill: drop the last few qids from ``run.tsv``, leave a partial
   chunk inside it, run with ``--resume``: only the missing qids are
   retrieved, the partial-chunk qid is retried, and the final ``run.tsv``
   covers every qid exactly once.
3. ``metrics.json`` is produced and reports the expected ``n_queries``.

Run::

    python scripts/smoke_test_resume.py
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.msmarco import MSMarcoPassage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("smoke_resume")

# ---------------------------------------------------------------------------
# Synthetic dataset (10 passages, 5 queries, deterministic qrels).
# ---------------------------------------------------------------------------

PASSAGES = [
    "The Eiffel Tower is in Paris, France.",
    "Canberra is the capital of Australia.",
    "William Shakespeare was an English playwright.",
    "Photosynthesis converts light energy to chemical energy in plants.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "DNA carries genetic information.",
    "Sydney is the largest city in Australia.",
    "Paris is the capital of France.",
    "Mount Everest is the highest mountain on Earth.",
    "The Great Wall of China was built for defense.",
]
DOC_IDS = [f"d{i}" for i in range(len(PASSAGES))]
QUERIES = {
    "q1": "what is the capital of australia",
    "q2": "where is the eiffel tower",
    "q3": "who was shakespeare",
    "q4": "what is photosynthesis",
    "q5": "what is the largest ocean",
}
QRELS = {
    "q1": {"d1"},
    "q2": {"d0"},
    "q3": {"d2"},
    "q4": {"d3"},
    "q5": {"d4"},
}


def _stub_loader(*_, **__):
    return MSMarcoPassage(
        corpus_doc_ids=list(DOC_IDS),
        corpus_texts=list(PASSAGES),
        queries=dict(QUERIES),
        qrels={k: set(v) for k, v in QRELS.items()},
        docs_store=None,
    )


# ---------------------------------------------------------------------------
# Test driver — invokes `run_retrieval.main()` with the loader stubbed.
# ---------------------------------------------------------------------------

def _run_main(extra_argv: list[str], output_dir: Path, index_dir: Path) -> None:
    import src.data.msmarco as _msmarco
    import experiments.run_retrieval as runner

    # ``run_retrieval`` does ``from src.data.msmarco import load_msmarco_passage``
    # at import time, so we have to patch the binding inside the runner module
    # *as well* as the source module. Patching only one is insufficient.
    _msmarco.load_msmarco_passage = _stub_loader
    runner.load_msmarco_passage = _stub_loader

    # Build a tiny config tailored to our synthetic data.
    cfg_path = output_dir.parent / "_smoke_resume_cfg.yaml"
    cfg_path.write_text(
        "seed: 42\n"
        "data:\n"
        "  cache_dir: data/raw\n"
        "  corpus_limit: null\n"
        "retrieval:\n"
        "  backend: bm25s\n"
        "  k1: 1.5\n"
        "  b: 0.75\n"
        "  stopwords: en\n"
        f"  top_k: {len(PASSAGES)}\n"
        f"  index_dir: {index_dir.relative_to(PROJECT_ROOT)}\n"
        "  chunk_size: 2\n"          # exercise multiple chunks: 5 queries / 2 = 3 chunks
        "  n_threads: 0\n"
        "  bm25s_chunksize: 50\n"
        "eval_retrieval:\n"
        f"  output_dir: {output_dir.relative_to(PROJECT_ROOT)}\n"
        "  ks_mrr: [10]\n"
        "  ks_recall: [10]\n"          # cap recall@k to corpus size
        "  n_examples: 3\n"
    )
    sys.argv = ["run_retrieval.py", "--config", str(cfg_path), *extra_argv]
    runner.main()


def _read_qids_from_tsv(path: Path) -> list[str]:
    qids = []
    with open(path) as f:
        for line in f:
            qids.append(line.split("\t", 1)[0])
    return qids


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "_smoke_resume"
    index_dir = PROJECT_ROOT / "data" / "processed" / "_smoke_resume_index"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    output_dir.mkdir(parents=True)

    # ---- Fresh run ----
    log.info("=== scenario A: fresh run ===")
    _run_main([], output_dir, index_dir)
    run_tsv = output_dir / "run.tsv"
    assert run_tsv.exists(), "run.tsv not created"
    assert (output_dir / "metrics.json").exists(), "metrics.json not created"

    fresh_lines = run_tsv.read_text().splitlines()
    fresh_qids_unique = set(_read_qids_from_tsv(run_tsv))
    assert fresh_qids_unique == set(QUERIES), (
        f"fresh run missed qids: expected {set(QUERIES)}, got {fresh_qids_unique}"
    )
    fresh_metrics = json.loads((output_dir / "metrics.json").read_text())
    fresh_mrr = fresh_metrics["metrics"]["mrr@10"]
    log.info("scenario A: %d lines, %d qids, mrr@10=%.4f",
             len(fresh_lines), len(fresh_qids_unique), fresh_mrr)

    # ---- Simulate a kill mid-run ----
    # Drop all entries for q4 and q5, plus 3 of the 10 lines for q3 to simulate
    # an incomplete final chunk that --resume must redo.
    log.info("=== scenario B: simulated kill — truncating run.tsv ===")
    kept = []
    q3_kept = 0
    for line in fresh_lines:
        qid = line.split("\t", 1)[0]
        if qid in {"q4", "q5"}:
            continue
        if qid == "q3":
            if q3_kept >= 7:
                continue
            q3_kept += 1
        kept.append(line)
    run_tsv.write_text("\n".join(kept) + "\n")
    pre_resume_qids = _read_qids_from_tsv(run_tsv)
    log.info("after truncation: %d lines, qid counts: %s",
             len(pre_resume_qids),
             {q: pre_resume_qids.count(q) for q in set(pre_resume_qids)})

    # ---- Resume ----
    log.info("=== scenario C: resume ===")
    _run_main(["--resume"], output_dir, index_dir)
    after_lines = run_tsv.read_text().splitlines()
    after_qids = _read_qids_from_tsv(run_tsv)
    after_counts = {q: after_qids.count(q) for q in set(after_qids)}
    expected_lines_per_qid = len(PASSAGES)  # top_k = corpus size in this test
    assert all(c == expected_lines_per_qid for c in after_counts.values()), (
        f"resume produced wrong per-qid line counts: {after_counts}"
    )
    assert set(after_qids) == set(QUERIES), (
        f"resume missed qids: {set(after_qids)} vs {set(QUERIES)}"
    )

    after_metrics = json.loads((output_dir / "metrics.json").read_text())
    after_mrr = after_metrics["metrics"]["mrr@10"]
    assert abs(after_mrr - fresh_mrr) < 1e-9, (
        f"mrr@10 differs: fresh={fresh_mrr}, resumed={after_mrr}"
    )
    log.info("scenario C: %d lines, %d qids, mrr@10=%.4f (matches fresh)",
             len(after_lines), len(set(after_qids)), after_mrr)
    assert after_metrics.get("resumed") is True, "resumed flag should be True"

    # ---- Cleanup ----
    shutil.rmtree(output_dir)
    shutil.rmtree(index_dir)
    log.info("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

"""Synthetic-data integration smoke test for the Week 4 dense pipeline.

Bypasses ``ir_datasets`` (no download), runs the *real*
``experiments/run_dense_retrieval.py`` orchestration end-to-end with:

- 200 synthetic passages (the relevant ones included by construction)
- 8 synthetic queries
- The real ``DenseRetriever`` (downloads ``all-MiniLM-L6-v2`` weights on
  first run; ~80 MB, cached after)
- The real ``BM25Retriever`` on the same sample

Asserts:

- run.tsv + run_bm25_sample.tsv + metrics.json + examples.jsonl created
- both retrievers find the synthetic relevant doc in top-1 (the synthetic
  signal is strong enough that any sensible retriever should ace it)
- metrics.json has the unified schema with ``metrics.dense`` and
  ``metrics.bm25_sample`` blocks

Usage::

    python scripts/smoke_test_week04.py

The script cleans up its outputs at the end so it doesn't pollute
the real ``outputs/week04_dense/`` directory.
"""

from __future__ import annotations

# macOS libomp workaround: faiss-cpu and torch each ship their own libomp,
# and loading both in the same process aborts with a duplicate-symbol error.
# Must be set BEFORE any import that pulls in faiss or torch.
import os  # noqa: E402

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.msmarco import MSMarcoPassage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("smoke_w4")


# ---------------------------------------------------------------------------
# Synthetic dataset: 8 queries, each with a clearly-matching gold passage,
# plus 192 random distractor passages.
# ---------------------------------------------------------------------------

GOLD = [
    ("q1", "what is the capital of australia",
     "d_q1", "Canberra is the capital city of Australia, located in the Australian Capital Territory."),
    ("q2", "where is the eiffel tower located",
     "d_q2", "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France."),
    ("q3", "who wrote hamlet",
     "d_q3", "Hamlet is a tragedy written by the English playwright William Shakespeare."),
    ("q4", "what is photosynthesis",
     "d_q4", "Photosynthesis is the biological process by which plants convert sunlight into chemical energy."),
    ("q5", "what is the largest ocean on earth",
     "d_q5", "The Pacific Ocean is the largest and deepest of the world's five oceanic divisions."),
    ("q6", "who invented the telephone",
     "d_q6", "Alexander Graham Bell is widely credited with inventing and patenting the first practical telephone."),
    ("q7", "what is dna",
     "d_q7", "DNA is a molecule that carries the genetic instructions for the development of all living organisms."),
    ("q8", "what is the highest mountain on earth",
     "d_q8", "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas."),
]

# 192 distractor passages drawn from a small mixed pool.
DISTRACTOR_TEXTS = [
    "Sydney is the largest city in Australia but it is not the capital.",
    "The Statue of Liberty was a gift from France to the United States.",
    "Charles Dickens wrote A Tale of Two Cities.",
    "Respiration is the process by which cells produce ATP from glucose.",
    "The Atlantic Ocean separates Europe and Africa from the Americas.",
    "Thomas Edison improved the design of the light bulb in the late 19th century.",
    "RNA is involved in protein synthesis inside cells.",
    "K2 is the second highest mountain in the world after Everest.",
] * 24  # 192


def _build_synthetic_data():
    queries = {q[0]: q[1] for q in GOLD}
    qrels = {q[0]: {q[2]} for q in GOLD}
    doc_ids = [g[2] for g in GOLD] + [f"distr{i}" for i in range(len(DISTRACTOR_TEXTS))]
    texts = [g[3] for g in GOLD] + DISTRACTOR_TEXTS
    return doc_ids, texts, queries, qrels


def main() -> int:
    out_dir = PROJECT_ROOT / "outputs" / "_smoke_week04"
    dense_idx = PROJECT_ROOT / "data" / "processed" / "_smoke_w4_dense"
    bm25_idx = PROJECT_ROOT / "data" / "processed" / "_smoke_w4_bm25"
    for p in (out_dir, dense_idx, bm25_idx):
        if p.exists():
            shutil.rmtree(p)

    doc_ids, texts, queries, qrels = _build_synthetic_data()
    log.info("Synthetic data: %d passages, %d queries", len(doc_ids), len(queries))

    # Patch the data loader and the pool-doc-ids reader inside the runner
    # module so we don't need ir_datasets or the W2 bm25 index.
    import src.data.msmarco as _msmarco
    import experiments.run_dense_retrieval as runner

    def _stub_loader(*args, **kwargs):
        return MSMarcoPassage(
            corpus_doc_ids=[],
            corpus_texts=[],
            queries=dict(queries),
            qrels={k: set(v) for k, v in qrels.items()},
            docs_store=None,
        )

    class _StubDocsStore:
        def __init__(self, mapping):
            self._m = mapping

        def get(self, doc_id):
            class _D:
                def __init__(self, text):
                    self.text = text
            if doc_id in self._m:
                return _D(self._m[doc_id])
            raise KeyError(doc_id)

    store = _StubDocsStore(dict(zip(doc_ids, texts)))

    _msmarco.load_msmarco_passage = _stub_loader
    runner.load_msmarco_passage = _stub_loader
    _msmarco.get_docs_store = lambda *a, **kw: store
    runner.get_docs_store = lambda *a, **kw: store
    runner._load_pool_doc_ids = lambda *a, **kw: list(doc_ids)

    # Write a tiny config tailored to the synthetic data.
    cfg_path = PROJECT_ROOT / "configs" / "_smoke_w4.yaml"
    cfg_path.write_text(
        "seed: 42\n"
        "data:\n"
        "  cache_dir: data/raw\n"
        "  corpus_limit: null\n"
        "retrieval:\n"
        "  k1: 1.5\n"
        "  b: 0.75\n"
        "  stopwords: en\n"
        "  top_k: 10\n"
        "  index_dir: data/processed/_smoke_w4_unused_bm25\n"
        "  chunk_size: 50\n"
        "  n_threads: 0\n"
        "  bm25s_chunksize: 50\n"
        "eval_retrieval:\n"
        "  ks_mrr: [10]\n"
        "  ks_ndcg: [10]\n"
        "  ks_recall: [10]\n"
        "  output_dir: outputs/_smoke_w4_unused\n"
        "dense:\n"
        "  model_name: sentence-transformers/all-MiniLM-L6-v2\n"
        f"  sample_size: {len(doc_ids)}\n"
        "  encode_batch_size: 32\n"
        "  device: cpu\n"
        f"  index_dir: {dense_idx.relative_to(PROJECT_ROOT)}\n"
        f"  bm25_sample_index_dir: {bm25_idx.relative_to(PROJECT_ROOT)}\n"
        f"  output_dir: {out_dir.relative_to(PROJECT_ROOT)}\n"
        "  top_k: 10\n"
        "  compare_bm25_on_sample: true\n"
        "  n_examples: 3\n"
    )

    sys.argv = ["run_dense_retrieval.py", "--config", str(cfg_path)]
    runner.main()

    # --- Assertions ---
    assert (out_dir / "run.tsv").exists(), "dense run.tsv missing"
    assert (out_dir / "run_bm25_sample.tsv").exists(), "bm25 sample run missing"
    assert (out_dir / "metrics.json").exists(), "metrics.json missing"
    assert (out_dir / "examples.jsonl").exists(), "examples.jsonl missing"
    assert (out_dir / "sample_doc_ids.json").exists(), "sample_doc_ids.json missing"

    payload = json.loads((out_dir / "metrics.json").read_text())
    assert payload["task"] == "retrieval", payload["task"]
    assert "dense" in payload["metrics"], "missing dense metrics block"
    assert "bm25_sample" in payload["metrics"], "missing bm25_sample metrics block"
    d_mrr = payload["metrics"]["dense"]["mrr@10"]
    b_mrr = payload["metrics"]["bm25_sample"]["mrr@10"]
    log.info("Synthetic dense MRR@10=%.4f, BM25 MRR@10=%.4f", d_mrr, b_mrr)
    assert d_mrr >= 0.7, f"dense MRR@10={d_mrr:.4f} too low for the synthetic signal"
    assert b_mrr >= 0.7, f"BM25 MRR@10={b_mrr:.4f} too low for the synthetic signal"

    # Cleanup
    for p in (out_dir, dense_idx, bm25_idx, cfg_path):
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    log.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    # ``os._exit`` skips the interpreter shutdown path. On macOS with both
    # faiss-cpu and torch loaded, that shutdown wedges in ``pthread_join``
    # trying to reap an OpenMP worker thread owned by the *other* libomp
    # instance (faiss ships libomp.dylib, torch ships libiomp5.dylib).
    # All real work has finished by the time we reach this line, so a hard
    # exit is safe and ~30 s faster than waiting on the broken join.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc or 0)

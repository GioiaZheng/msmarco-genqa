# Retrieval Lift Analysis

Aggregate retrieval metrics are useful for reporting, but they hide the
mechanism behind the gain. A reranker can improve MRR in several different
ways: it can move an already-retrieved relevant passage upward, recover a
relevant passage into the top-k window, or damage a query that the first stage
already handled well.

This project now exposes a query-level diagnostic layer for that question:

```bash
python scripts/analyze_retrieval_lift.py \
  --before-run outputs/dense_retrieval/run.tsv \
  --after-run outputs/cross_encoder_rerank_full/run.tsv \
  --output-dir outputs/retrieval_lift_analysis
```

If `--qrels` is omitted, the script loads MS MARCO passage dev/small qrels
through `ir_datasets`. A local qrels file can also be supplied in standard
4-column TREC format or compact 3-column `qid docid relevance` format.
For graded collections, pass the same binary relevance threshold used by the
headline MRR/recall evaluation. TREC-DL passage tracks use:

```bash
python scripts/analyze_retrieval_lift.py \
  --before-run outputs/trec_dl_2019/bm25/run.tsv \
  --after-run outputs/trec_dl_2019/cross_encoder_rerank/run.tsv \
  --qrels data/processed/trec-dl-2019-passage.qrels \
  --rel-threshold 2 \
  --output-dir outputs/trec_dl_2019/retrieval_lift_analysis
```

## Outputs

The analysis writes four artifacts:

- `retrieval_lift.json` -- headline counts and mean deltas.
- `retrieval_lift_by_bucket.csv` -- bucket-level summary table.
- `retrieval_lift_examples.jsonl` -- representative per-query examples.
- `retrieval_lift.md` -- a short report suitable for attaching to experiment
  notes.

## Buckets

Each query is assigned one movement bucket using the first qrels-relevant
document within `k_rank`:

| bucket | meaning |
|---|---|
| `promoted` | first relevant document stayed in the window and moved upward |
| `demoted` | first relevant document stayed in the window but moved downward |
| `new_hit` | after-run found a relevant document inside the window; before-run did not |
| `lost_hit` | before-run had a relevant document inside the window; after-run lost it |
| `unchanged_hit` | first relevant document stayed at the same rank |
| `unchanged_miss` | neither run found a relevant document inside the window |

The output keeps both ranking and recall signals:

- `rr_delta@k_rank` explains MRR movement.
- `rank_movement` explains how far the first relevant passage moved.
- `recall_delta@k_recall` catches candidate-set gains/losses at a deeper
  retrieval depth.

## Why this matters

This adds a failure-analysis view that is more useful than a single MRR row.
For example, a strong cross-encoder reranker should show most of its lift in
`promoted` and `new_hit` queries, while a high `demoted` or `lost_hit` count
would indicate overfitting, lexical mismatch, or candidate-set instability.
Those buckets can be sliced later by query form, answer type, generator
metric, or grounding score.

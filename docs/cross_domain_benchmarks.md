# NFCorpus / SciFact Cross-Domain Benchmarks

## Purpose

MS MARCO `dev/small` and TREC-DL 2019/2020 are useful retrieval evidence, but
they are still tied to the MS MARCO passage collection. NFCorpus and SciFact are
the first external-domain checks used here to test whether the retrieval and
reranking stack remains useful outside that collection.

The goal of this step is deliberately narrow:

- run the same BM25 first stage and cross-encoder reranker on two BEIR test
  collections;
- keep each dataset's own corpus, qrels, indexes, and outputs separate;
- report retrieval metrics before doing deeper error analysis; and
- avoid claiming cross-domain generalization until complete checked artifacts
  are committed.

## Dataset Contract

| Dataset id | Domain | Corpus | Qrels | Positive threshold |
|---|---|---|---|---:|
| `beir/nfcorpus/test` | Medical information retrieval | NFCorpus BEIR corpus | BEIR test qrels | `rel >= 1` |
| `beir/scifact/test` | Scientific claim retrieval | SciFact BEIR corpus | BEIR test qrels | `rel >= 1` |

Both datasets are loaded through `ir_datasets`. Unlike TREC-DL, they do not
reuse the MS MARCO passage corpus or the shared MS MARCO BM25 index. This keeps
the cross-domain benchmark from accidentally measuring retrieval against the
wrong document collection.

## Runner Commands

```bash
mgq-retrieve --dataset beir/nfcorpus/test --resume --require-clean-tree
mgq-rerank --dataset beir/nfcorpus/test --resume --require-clean-tree

mgq-retrieve --dataset beir/scifact/test --resume --require-clean-tree
mgq-rerank --dataset beir/scifact/test --resume --require-clean-tree
```

Default outputs are isolated by dataset:

- `outputs/beir_nfcorpus_test/bm25`
- `outputs/beir_nfcorpus_test/cross_encoder_rerank`
- `outputs/beir_scifact_test/bm25`
- `outputs/beir_scifact_test/cross_encoder_rerank`

Default BM25 indexes are also isolated:

- `data/processed/bm25_index_beir_nfcorpus_test`
- `data/processed/bm25_index_beir_scifact_test`

## Metrics

Runner `metrics.json` files use:

- MRR@10;
- nDCG@10 with the original graded labels;
- Recall@100 and Recall@1000; and
- coverage diagnostics for queries present in the run and qrels.

For thresholded metrics such as MRR and recall, BEIR qrels are binarized with
`rel >= 1`. nDCG retains the original relevance labels.

## Independent Cross-Check

Materialize the qrels and validate each run with the same TREC-compatible
evaluator used by the MS MARCO and TREC-DL runners:

```bash
ir_datasets export beir/nfcorpus/test qrels --format trec \
  > data/processed/beir-nfcorpus-test.qrels
ir_datasets export beir/scifact/test qrels --format trec \
  > data/processed/beir-scifact-test.qrels

mgq-trec-eval --backend ir-measures --qrels-format trec --rel-threshold 1 \
  --qrels data/processed/beir-nfcorpus-test.qrels \
  --run outputs/beir_nfcorpus_test/bm25/run.tsv \
  --output-dir outputs/beir_nfcorpus_test/bm25/trec_eval

mgq-trec-eval --backend ir-measures --qrels-format trec --rel-threshold 1 \
  --qrels data/processed/beir-scifact-test.qrels \
  --run outputs/beir_scifact_test/bm25/run.tsv \
  --output-dir outputs/beir_scifact_test/bm25/trec_eval
```

Repeat the same command for each dataset's
`cross_encoder_rerank/run.tsv`. Keep NFCorpus and SciFact separate in reports;
do not average them into one headline without reporting both dataset values.

## Current Status

The code path and offline tests are implemented. No NFCorpus or SciFact
benchmark score is claimed yet. A reportable result should include:

- the run commit;
- the resolved config and manifest;
- BM25 and reranked `metrics.json`;
- the TREC-compatible cross-check output; and
- notes on any query or document coverage gaps.

Only after those artifacts exist should the project move to deeper error
analysis on failures.

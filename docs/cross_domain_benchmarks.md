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
- make only the transfer claim supported by complete checked artifacts.

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

The BM25 retrieval `metrics.json` files use:

- MRR@10;
- nDCG@10 with the original graded labels;
- Recall@100 and Recall@1000; and
- coverage diagnostics for queries present in the run and qrels.

The cross-encoder only reranks the fixed BM25 top-100 candidate set. Its
`metrics.json` therefore reports MRR@10, nDCG@10, and Recall@100, but omits
Recall@1000. A top-100 reranked run cannot establish Recall@1000: evaluating
that cutoff would only repeat Recall@100 and could be mistaken for a recall
drop relative to the full BM25 top-1000 run. Recall@100 is expected to remain
unchanged because reranking changes order, not candidate membership.

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
The standalone evaluator still computes Recall@1000 for any supplied run. For
a top-100 reranked file, that value is candidate-set Recall@100 under a larger
cutoff and must not be presented as a comparable Recall@1000 result.

## Results

The full checked runs include every judged test query and use each dataset's
full corpus:

| Dataset | System | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 |
|---|---|---:|---:|---:|---:|
| NFCorpus (323 queries) | BM25 | 0.5186 | 0.3064 | 0.2378 | 0.4572 |
| NFCorpus (323 queries) | BM25 + CE | 0.5662 | 0.3411 | 0.2378 | n/a (top-100 run) |
| SciFact (300 queries) | BM25 | 0.6312 | 0.6617 | 0.8759 | 0.9606 |
| SciFact (300 queries) | BM25 + CE | 0.6517 | 0.6787 | 0.8759 | n/a (top-100 run) |

| Dataset | MRR@10 delta | Relative | nDCG@10 delta | Relative |
|---|---:|---:|---:|---:|
| NFCorpus | +0.0476 | +9.18% | +0.0347 | +11.33% |
| SciFact | +0.0205 | +3.25% | +0.0170 | +2.56% |

The reranker improves early ranking on both collections. Recall@100 is
unchanged because it receives the same BM25 top-100 document set. The large
difference in first-stage Recall@100 - 0.2378 on NFCorpus versus 0.8759 on
SciFact - is the main diagnostic result: NFCorpus is primarily candidate-set
limited, so a stronger first-stage retriever is a more plausible next change
than further tuning the fixed-candidate reranker.

## Provenance and Audit

| Evidence | NFCorpus | SciFact |
|---|---|---|
| Run commits | `e276d80fb4da` | BM25 `e276d80fb4da`; CE `f82caf651041` |
| BM25 coverage | 323/323 topics, depth 1000 | 300/300 topics, depth 1000 |
| CE coverage | 323/323 topics, 32,300 pairs | 300/300 topics, 30,000 pairs |
| BM25 runtime | 2.72 s index; 0.91 s search | 2.80 s index; 1.02 s search |
| CE runtime | 3,110.14 s; 10.39 pairs/s | 2,393.56 s; 12.53 pairs/s |
| CE device | CPU | CPU |

Both reranking runs use
`cross-encoder/ms-marco-MiniLM-L-6-v2`, batch size 64, maximum length 512,
and fixed depth 100. The machine has an NVIDIA GPU, but the recorded Python
environments used CPU-only PyTorch; the table therefore reports CPU runtimes.

The repository manifest verifier passed for every run. A separate structural
audit found no missing topics, malformed TREC rows, duplicate documents, rank
sequence errors, non-finite scores, or BM25/CE candidate-set mismatches. All
323 NFCorpus and all 300 SciFact rankings changed. Independent `ir-measures`
evaluation reproduced the reported metrics with maximum absolute difference
`4.45e-16`.

The SciFact CE scoring run completed before its launcher failed to record Git
metadata because Git was absent from that process's `PATH`. The manifest was
repaired without rescoring under strict checks: clean tree, exact commit,
300 queries, 30,000 pairs, fixed candidate sets, output hashes, and independent
metric agreement. The manifest records this repair explicitly.

## Public Evidence Bundle

The exact four ranked outputs behind the table are published in the immutable
GitHub Release `v2.2-beir-cross-domain-baselines`. The Git-tracked pointer
[`artifacts/beir_cross_domain_v1.json`](../artifacts/beir_cross_domain_v1.json)
pins the asset name, byte size, outer SHA-256 digest, source-record digest, and
the per-stage experiment commits.

From a configured clone, run:

```bash
make reproduce-beir-eval
```

This downloads the 6.0 MB archive, verifies the archive and every member,
recovers the public NFCorpus and SciFact test qrels through `ir_datasets`, and
recomputes the four metric rows with tolerance `1e-12`. It is an evidence
reproduction, not a new model run: it does not rebuild the two BM25 indexes or
rerun 62,300 cross-encoder pairs.

The bundle contains ranked document identifiers, ranks, scores, compact
metadata, and checksums only. It does not redistribute document/query text,
qrels mirrors, model weights, caches, or machine-local manifests.

## Interpretation Boundary

These results support a narrow conclusion: the unchanged MS-MARCO-trained
cross-encoder improves top-rank retrieval quality on two non-MS-MARCO corpora.
They do not show that the full retrieval-augmented generation pipeline
generalizes across domains, because neither dataset was run through generation
or grounded-answer evaluation. They also do not establish state of the art or
replace a broader benchmark suite.

The follow-up first-stage analyses are recorded in
[`nfcorpus_first_stage_error_analysis.md`](nfcorpus_first_stage_error_analysis.md)
and
[`scifact_first_stage_error_analysis.md`](scifact_first_stage_error_analysis.md).
They use the published fixed BM25 outputs and public qrels without rebuilding
indexes or rerunning retrieval. NFCorpus has 72/323 queries with no relevant
document in the top-100 candidate set; 24 of those first obtain a relevant hit
at ranks 101-1000, while 48 remain misses at depth 1000. SciFact has 35/300
queries with no relevant document in the top-100 candidate set, 24 first hits
at ranks 101-1000, and 11 misses at depth 1000. The comparison supports
treating the large NFCorpus gap as dataset- and representation-sensitive
rather than a general failure of the unchanged first stage.

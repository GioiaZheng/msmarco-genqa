# Results

This file summarizes the headline measurements and the interpretation limits
for the current MS MARCO GenQA experiment line. Detailed stage notes are kept
in `docs/experiments.md`.

## Headline Finding

Replacing BM25 top-3 passages with cross-encoder-reranked dense top-3
passages improves generation metrics under a paired setup:

| Retrieval source -> T5-small | ROUGE-L | BLEU | EM | Token-F1 |
|---|---:|---:|---:|---:|
| BM25 | 0.1859 | 0.0717 | 0.0135 | 0.1966 |
| Reranked | 0.3621 | 0.2922 | 0.0606 | 0.3677 |
| Delta | +0.1763 | +0.2206 | +0.0471 | +0.1711 |

The comparison uses the same generator, prompt format, top-k depth, and paired
query set.

## Statistical Reliability

Paired-bootstrap confidence intervals on 6,980 paired queries:

| Metric | Delta | 95% CI |
|---|---:|---:|
| ROUGE-L | +0.1742 | [+0.1663, +0.1820] |
| BLEU | +0.1330 | [+0.1265, +0.1395] |
| Exact Match | +0.0471 | [+0.0417, +0.0527] |
| Token-F1 | +0.1711 | [+0.1632, +0.1789] |

All intervals are above zero.

## Retrieval Stages

### Full-Corpus BM25

Measured on dev/small with the full passage corpus:

| Metric | Value |
|---|---:|
| MRR@10 | 0.1703 |
| Recall@100 | 0.6212 |
| Recall@1000 | 0.8154 |

### Dense Retrieval on 50k Qrels-Anchored Sample

The dense stage is evaluated on a controlled 50k-passage sample, not the full
8.8M-passage corpus.

| Metric | BM25-on-sample | Dense |
|---|---:|---:|
| MRR@10 | 0.6948 | 0.8830 |
| Recall@100 | 0.9338 | 0.9946 |

### Cross-Encoder Reranking

Reranking reorders dense top-100 results:

| Metric | Dense | Reranked | Delta |
|---|---:|---:|---:|
| MRR@10 | 0.8830 | 0.9304 | +0.0474 |
| nDCG@10 | 0.9041 | 0.9434 | +0.0393 |
| Recall@100 | 0.9946 | 0.9946 | +0.0000 |

Recall@100 is unchanged because reranking only changes order within the
retrieved top-100.

## External Retrieval Benchmark: TREC-DL 2019/2020

The full-corpus BM25 first stage and fixed top-100 cross-encoder reranker were
also evaluated on the deeply judged TREC-DL passage tracks:

| Track | Metric | BM25 | BM25 + CE | Delta |
|---|---|---:|---:|---:|
| 2019 (43 topics) | MRR@10, rel >= 2 | 0.5471 | 0.8787 | +0.3315 |
| 2019 (43 topics) | nDCG@10, graded | 0.4239 | 0.7210 | +0.2971 |
| 2020 (54 topics) | MRR@10, rel >= 2 | 0.6280 | 0.8256 | +0.1976 |
| 2020 (54 topics) | nDCG@10, graded | 0.4773 | 0.6801 | +0.2027 |

All judged topics remain in the denominator, and an independent `ir-measures`
cross-check reproduced the metrics with a maximum absolute delta of
`2.22e-16`. The complete protocol, runtime notes, query-level lift analysis,
and artifact hashes are in
[`docs/trec_dl_external_validity.md`](docs/trec_dl_external_validity.md).

These are validated retrieval results. They do not show that the retrieval
gain transfers to answer generation on TREC-DL.

## Cross-Domain Retrieval Benchmark: NFCorpus / SciFact

The same BM25 first stage and unchanged
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranker were evaluated on two BEIR test
collections with their own corpora and qrels. The reranker reorders the fixed
BM25 top-100 candidate set:

| Dataset | System | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 |
|---|---|---:|---:|---:|---:|
| NFCorpus (323 queries) | BM25 | 0.5186 | 0.3064 | 0.2378 | 0.4572 |
| NFCorpus (323 queries) | BM25 + CE | 0.5662 | 0.3411 | 0.2378 | n/a (top-100 run) |
| SciFact (300 queries) | BM25 | 0.6312 | 0.6617 | 0.8759 | 0.9606 |
| SciFact (300 queries) | BM25 + CE | 0.6517 | 0.6787 | 0.8759 | n/a (top-100 run) |

Cross-encoder reranking raises MRR@10 and nDCG@10 on both datasets. The larger
relative lift is on NFCorpus: +9.18% MRR@10 and +11.33% nDCG@10, compared with
+3.25% and +2.56% on SciFact. Recall@100 is unchanged by construction because
the candidate membership is fixed. The first-stage limitation is much stronger
on NFCorpus (Recall@100 0.2378) than on SciFact (0.8759), so reranking alone
cannot recover most NFCorpus relevant documents.

All 323 NFCorpus and 300 SciFact judged queries are included. Independent
`ir-measures` evaluation reproduced every reported metric to floating-point
precision (maximum absolute delta `4.45e-16`), and the run-file audit found no
missing topics, malformed rows, duplicate documents, or candidate-set changes.
The protocol, runtime evidence, provenance, and metric boundaries are recorded
in [`docs/cross_domain_benchmarks.md`](docs/cross_domain_benchmarks.md).

The exact four run files are recoverable without private credentials from the
checksummed
[`v2.2-beir-cross-domain-baselines`](https://github.com/GioiaZheng/msmarco-genqa/releases/tag/v2.2-beir-cross-domain-baselines)
release. `make reproduce-beir-eval` verifies the archive and member hashes,
recovers public qrels through `ir_datasets`, and recomputes all reported rows
without rebuilding indexes or rerunning the cross-encoder.

These results show that the ranking benefit transfers to two non-MS-MARCO
retrieval collections. They do not establish broad cross-domain RAG
generalization or downstream generation quality.

## Query-Type Slice

Token-F1 lift by query type:

| Query type | n | BM25 | Reranked | Delta |
|---|---:|---:|---:|---:|
| DESCRIPTION | 3,725 | 0.1889 | 0.3939 | +0.2050 |
| ENTITY | 631 | 0.1765 | 0.3186 | +0.1421 |
| LOCATION | 498 | 0.2495 | 0.3928 | +0.1433 |
| NUMERIC | 1,665 | 0.1997 | 0.3235 | +0.1238 |
| PERSON | 461 | 0.2186 | 0.3557 | +0.1371 |

DESCRIPTION queries benefit most; NUMERIC queries benefit least.

## Interpretation Limits

The main generation comparison is paired and stable, but the retrieval stages
have different evaluation boundaries:

| Comparison | Interpretation |
|---|---|
| Full-corpus BM25 | Realistic lexical baseline over the full corpus |
| Dense vs BM25-on-sample | Controlled comparison on the same qrels-anchored pool |
| Dense vs reranked | Valid ordering comparison over dense top-100 |
| Dense-sample vs full-corpus BM25 | Not a direct apples-to-apples comparison |

The dense sample includes all dev relevant documents by construction. This is
useful for isolating model behavior, but it is optimistic relative to full
corpus retrieval.

### Conclusion Boundary

| Status | Boundary |
|---|---|
| Validated | The paired T5-small generation comparison on MS MARCO `dev/small`; full-corpus BM25 plus cross-encoder retrieval on TREC-DL 2019/2020; full-corpus BM25 plus fixed top-100 cross-encoder reranking on BEIR NFCorpus and SciFact. |
| Implemented but not yet evaluated | The T5-base generator-capacity sweep and configurable alternative-generator paths. Their existence is not an empirical result. |
| Not supported by current evidence | Retrieval lift transfers to generation on TREC-DL or BEIR; a fair full-corpus dense-vs-BM25 conclusion; broad cross-domain RAG generalization beyond the two evaluated retrieval collections. |

## Grounding

The surface metrics show a clear lift from reranked retrieval. The grounding
audit should be read separately: stronger retrieval improves answer overlap,
but it does not by itself prove deeper reasoning or fully faithful generation.

The project therefore reports both answer-surface metrics and
grounding-oriented checks. The detailed analysis is in `docs/experiments.md`
and the paper-style report.

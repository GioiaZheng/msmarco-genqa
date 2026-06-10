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

## Grounding

The surface metrics show a clear lift from reranked retrieval. The grounding
audit should be read separately: stronger retrieval improves answer overlap,
but it does not by itself prove deeper reasoning or fully faithful generation.

The project therefore reports both answer-surface metrics and
grounding-oriented checks. The detailed analysis is in `docs/experiments.md`
and the paper-style report.

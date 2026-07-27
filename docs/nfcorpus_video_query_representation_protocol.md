# NFCorpus Video Query-Representation Protocol

## Decision

The next experiment will compare bounded query representations on the official
102-query NFCorpus video test subset. It will not replace the current BEIR
benchmark, change the architecture, or use the official full-page `all` field
as a headline condition.

The three predeclared conditions are:

1. the existing BEIR title;
2. the official video description;
3. title plus description.

The corpus, qrels, BM25 implementation, index, cross-encoder, candidate depth,
and evaluation rules remain fixed. Only the query text changes.

## Why this is the next controlled test

The 72-case first-stage review found that many NFCorpus failures depend on
source-page context omitted from the BEIR title. Before testing a new
retriever, the lower-risk question is whether a bounded source field recovers
candidate coverage.

The official dataset provides two relevant pieces of evidence:

- [`nfcorpus/test/video`](https://ir-datasets.com/nfcorpus.html) exposes video
  titles and descriptions for 102 test queries;
- the [dataset authors](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
  state that queries come from NutritionFacts page content and that relevance
  is constructed from direct citations, indirect internal links, and
  topic/tag links.

The video description is bounded: in the pinned v1 archive it has a median of
21 words and a maximum of 65. The full-page `all` field has a median of 192
words, exceeds 512 words for 136 of the 323 shared test queries, and reaches a
maximum of 42,966 words. That is a different retrieval condition rather than a
normal query expansion.

## Data alignment

The feasibility audit verified the following before any new retrieval run:

| Check | Result |
|---|---:|
| BEIR test qids found in the official test set | 323/323 |
| Normalized official titles matching BEIR query text | 323/323 |
| Official qrels matching BEIR after the verified filtering/remapping | 12,334/12,334 |
| Retained relevant docids present in the BEIR corpus | 3,128/3,128 |
| Official non-topic qids matching the URL-derived project slice | 144/144 |
| Official video qids matching the URL-derived project slice | 102/102 |

The two official test queries absent from BEIR contain only the lowest
relevance level, which BEIR excludes. Official relevance levels 3 and 2 map
exactly to BEIR levels 2 and 1 respectively.

The official NFCorpus v1 archive is pinned at:

- bytes: `31,039,523`;
- MD5: `49c061fbadc52ba4d35d0e42e2d742fd`;
- SHA-256:
  `e2325ace35d02185a22e96bb52e72e62f2caf45a4975757c81a1c4087d8c59e9`.

`ir_datasets` 0.6.1 currently points to an obsolete Heidelberg download path.
The experiment must use the current official download URL recorded in
[`configs/nfcorpus_video_query_representation.json`](../configs/nfcorpus_video_query_representation.json)
and must retain the published MD5 check. Integrity verification must not be
disabled.

## Leakage boundary

Query construction may read only:

- the frozen 102-qid membership;
- the BEIR query title;
- the official video title and description.

It must not read qrels, corpus documents, ranked outputs, metrics, or
case-specific annotations while constructing query text.

As a conservative audit proxy, no four-token-or-longer direct-qrel document
title occurred verbatim in the 91 video descriptions whose queries have a
direct judgment. This does not prove that the descriptions are independent of
the cited evidence. It only rules out one obvious form of literal leakage.
The description condition should therefore be interpreted as a richer source
representation, not as an independently authored user query.

The following are excluded from the primary comparison:

- full-page `all` text;
- live scraping of current NutritionFacts pages;
- manual rewriting;
- qrel-conditioned or retrieved-document-conditioned expansion.

## Frozen baseline

The existing title-only fixed run gives the following 102-query video slice:

| System | MRR@10 | nDCG@10 | Recall@100 | Recall@1000 |
|---|---:|---:|---:|---:|
| BM25 | 0.478019 | 0.249165 | 0.282065 | 0.532002 |
| BM25 + cross-encoder | 0.517110 | 0.297206 | 0.282065 | n/a |

Eleven video queries have no positive qrel in the BM25 top 100. Five first
recover a positive at ranks 101–1000, and six still have no positive at depth
1000.

The new title condition must reproduce this slice before either treatment is
accepted.

## Evaluation

Recall@100 is the primary metric because the experiment targets the candidate
set. MRR@10, nDCG@10, Recall@1000, and the number of no-hit queries at depths
100 and 1000 are secondary diagnostics.

All aggregate comparisons use the same 102 qids. Query-level metric
differences will be summarized with 10,000 paired-bootstrap resamples, a fixed
seed, and 95% confidence intervals. Recall@1000 is not reported for reranked
top-100 runs.

The cross-encoder may be run only after each BM25 condition passes its input,
run-depth, score, and qid checks. It must rerank exactly that condition's
top-100 candidates; it cannot add documents.

## Acceptance and stopping rules

The experiment is accepted only if:

- all three conditions contain exactly the frozen 102 qids;
- the title condition reproduces the fixed-run slice within `1e-12`;
- corpus, qrels, index fingerprint, and model configuration are identical;
- query construction runs without access to outcome data;
- run files have finite scores, contiguous ranks, and no duplicate documents;
- an independent metric cross-check agrees with the project evaluator.

The result will be interpreted through the paired Recall@100 change and the
change from 11 no-hit-at-100 queries. A better MRR alone is not sufficient to
claim that missing context repaired the candidate ceiling.

Hybrid or dense retrieval should be considered only if a material first-stage
limitation remains after the best bounded representation. README headline
claims will not change until the complete result, review, and CI are finished.

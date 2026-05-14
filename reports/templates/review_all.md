# MS MARCO Generative QA — Progress Report

*Generated {{generated_at}}. For per-week depth see `reports/generated/weekNN_*.pdf`.*

## Project goal

Build an end-to-end open-domain QA system on MS MARCO that goes
**query → retrieve passages → generate an answer**, and measure each
stage in isolation so it's clear where errors come from. The aim is not
SOTA; the aim is a clean, reproducible pipeline whose retrieval,
reranking, and generation stages can each be ablated.

## Methods (by week)

| Week | Stage                     | Data                                                  | Method                                                                  |
|------|---------------------------|-------------------------------------------------------|-------------------------------------------------------------------------|
| W1   | EDA                       | MS MARCO QA v2.1 validation, 5k-row sample            | Length / type / answer-type distributions; sizing decisions for W2–W5   |
| W2   | Sparse retrieval (BM25)   | MS MARCO Passage (8.8M docs), dev/small (6 980 q)     | `bm25s` backend, `k1=1.5`, `b=0.75`, top-1000 per query                 |
| W3   | RAG generation            | BM25 top-3 from W2 ∩ QA references, 200 q             | T5-small (no fine-tuning), prompt = `question: <q> context: <p1 p2 p3>` |
| W4   | Dense retrieval           | Qrels-anchored 50k sample of Passage, dev/small (6 980 q) | `all-MiniLM-L6-v2` + FAISS `IndexFlatIP`; same sample re-indexed with BM25 for apples-to-apples comparison |
| W5   | Cross-encoder reranking   | W4 dense top-100, full dev/small (6 980 q)            | `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank, depth K=100              |

All runs share a single config (`configs/baseline.yaml`); the four
entrypoints in `experiments/run_*.py` produce the metrics each report
cites.

## Key results

**W1 (EDA)** — Queries are short (5–10 words median); passages ~10× longer
(median ~50 words). `DESCRIPTION` and `NUMERIC` dominate the query-type
buckets; a non-trivial fraction of queries has no answer and must be
filtered for any SFT supervision. Most queries have 0 or 1 marked-relevant
passage (sparse, binary qrels).

**W2 (BM25 baseline, full 8.8M corpus, 6 980 dev/small queries)**

| Metric      | Value  |
|-------------|--------|
| MRR@10      | 0.1703 |
| Recall@100  | 0.6212 |
| Recall@1000 | 0.8154 |

Published Anserini/Lucene reference is ~0.184 MRR@10; the ~0.014 gap is
attributable to tokenizer differences between Lucene and `bm25s`. No
`k1`/`b` tuning. Search cost: 586 ms/query.

**W3 (RAG, T5-small + BM25 top-3, 200 queries)**

| Metric    | Value  |
|-----------|--------|
| ROUGE-L   | 0.1619 |
| BLEU      | 0.0573 |
| EM        | 0.0050 |
| Token F1  | 0.1756 |

Honest end-to-end check, not a leaderboard number. T5-small is pretrained-only
(no SFT) and produces verbose, extractive-style outputs that are penalised
by surface-form metrics. Retrieval errors propagate directly into
generation — a reranking step (W5) is expected to lift these numbers
when wired in.

**W4 (Dense vs BM25 on the same 50k qrels-anchored sample, 6 980 queries)**

| Metric      | BM25 (sample) | Dense (sample) | Δ          |
|-------------|---------------|----------------|------------|
| MRR@10      | 0.6948        | 0.8830         | **+0.1882** |
| nDCG@10     | 0.7270        | 0.9041         | +0.1771    |
| Recall@100  | 0.9338        | 0.9946         | +0.0608    |
| Recall@1000 | 0.9686        | 0.9991         | +0.0305    |

Dense wins on every metric. Caveat: every dev/small relevant doc is
unconditionally included in the 50k pool, so absolute numbers are
upper-bounded; the *delta* between dense and BM25 on the same pool is
what's meaningful. Recall@100 = 0.9946 means dense has effectively hit
the ceiling on this sampled setting → from W5 onward, recall is no
longer the bottleneck.

**W5 (Cross-encoder rerank over W4 dense top-100, full 6 980-query dev/small)**

| Metric      | Dense (top-100) | Dense + CE rerank | Δ          |
|-------------|-----------------|-------------------|------------|
| MRR@10      | 0.8830          | 0.9304            | **+0.0474** |
| nDCG@10     | 0.9041          | 0.9434            | +0.0393    |
| Recall@100  | 0.9946          | 0.9946            | +0.0000    |

Recall@100 is unchanged by construction (the reranker only re-orders the
top-100). The interesting columns are MRR@10 and nDCG@10: even with
recall saturated, the cross-encoder still buys roughly +0.04 by tightening
*local ordering* — pushing the relevant passage from rank 2–3 to rank 1.
Cost: 4 h 37 min wall-clock for 538 k pairs (~32 pairs/s), peak RSS
3.3 GiB; the CE is the slowest stage by ~2 orders of magnitude. The
earlier 1 000-query subsample of dev/small produced consistent deltas
(MRR Δ +0.0435, nDCG Δ +0.0398), confirming the gain is not an artefact
of the subsample.

## Limitations

- **Sampled corpus from W4 onward.** Dense + rerank numbers are on a
  50k qrels-anchored sample where every relevant doc is present by
  construction. Comparable *within* W4/W5; the W4/W5 absolute MRR is
  *not* directly comparable to the W2 full-corpus number.
- **No fine-tuning anywhere.** Generic encoders/generator used as-is
  (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`, `t5-small`). A
  MS-MARCO-tuned encoder (`msmarco-MiniLM-L6-cos-v5`) or SFT generator
  would close most of the gap to leaderboard baselines.
- **Dev/small only, single seed.** No held-out test split; distractor
  sample is deterministic but not averaged across seeds.
- **W3 was evaluated on 200 queries.** A subsample of dev/small for
  compute reasons; metric variance not quantified. W5 now covers full
  dev/small (6 980 queries).
- **Surface-form generation metrics only.** ROUGE/BLEU/EM penalise valid
  paraphrases — a semantic metric (BERTScore) would give a fairer signal.

## Next steps

- Wire W5's reranked top-K into the W3 RAG generator and measure whether
  better ordering improves answer quality (not just retrieval metrics).
- Rerank BM25 top-100 (W2 run) and compare the delta to the dense delta
  — does the reranker recover more from a weaker first stage?
- Hybrid first-stage (RRF over BM25 + dense) → rerank.
- Scale the dense sample from 50k to 200k–500k passages to track how the
  BM25↔dense gap shifts as relevant-doc density drops.
- Add a held-out test split and BERTScore for generation, so improvements
  aren't overfit to dev/small or to surface-form metrics.

## How this maps to the repo

- Conclusions per week: `reports/generated/weekNN_*.pdf`
- Process logs per week: `notebooks/weekNN_*.ipynb`
- Reproducible pipeline: `experiments/run_*.py` + `src/` + `configs/baseline.yaml`
- Reproduce locally: `make` targets (see `Makefile`); outputs land in
  `outputs/` (not committed).

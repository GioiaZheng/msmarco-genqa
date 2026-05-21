# Week 5: Cross-Encoder Reranking

*Auto-generated {{generated_at}} from `outputs/week05_reranker/`. Do not edit
by hand. Re-run `python -m src.reporting.build_report --week week05` to refresh.*

## 1. Objective

W4 showed that, on the qrels-anchored sampled corpus, dense retrieval
already pushes Recall@100 to {{dense_recall_at_100_w4}}. Recall is no
longer where the gain comes from. W5 asks the next question:

> Once recall has saturated, can a cross-encoder reranker still improve
> the *local ordering* — i.e. push the relevant passage from rank 3 to
> rank 1?

Pipeline:

    dense top-{{rerank_top_k}}  →  cross-encoder rerank  →  evaluate

We deliberately keep the first stage unchanged from W4 (no re-encoding,
no new index) so the delta is attributable to the reranker alone.

## 2. Setup

- **First-stage retriever**: W4 dense run (`{{input_run}}`),
  truncated to top-{{rerank_top_k}} per query.
- **Reranker**: `{{model_name}}` — the standard MS MARCO cross-encoder
  baseline (small enough to run on CPU).
- **Eval set**: {{n_queries}} dev/small queries with ≥1 qrel landing in
  the W4 sampled pool.
- **Evaluation**: same sample-restricted qrels as W4, so the dense
  numbers reported in column 1 below are directly comparable to the W4
  dense numbers.

## 3. Why top-{{rerank_top_k}}?

A cross-encoder is O(K) forward passes per query — every (query, passage)
pair goes through the model. K=1000 would be ~10× the cost of K=100
with diminishing returns once recall is already near ceiling. The
standard depth in the MS MARCO reranking literature is 100, which is
what we use here.

## 4. Results

| Metric        | Dense (top-{{rerank_top_k}}) | Dense + CE rerank | Δ (rerank − dense) |
|---------------|------------------------------|-------------------|--------------------|
| MRR@10        | {{dense_mrr10}}              | {{rerank_mrr10}}  | {{delta_mrr10}}    |
| nDCG@10       | {{dense_ndcg10}}             | {{rerank_ndcg10}} | {{delta_ndcg10}}   |
| Recall@100    | {{dense_r100}}               | {{rerank_r100}}   | {{delta_r100}}     |

Recall@100 is unchanged by construction: the reranker only re-orders
the top-{{rerank_top_k}} candidates the dense retriever already
returned. The interesting columns are MRR@10 and nDCG@10 — both measure
where in the ranked list the relevant passage actually lands.

## 5. Runtime

| Quantity              | Value                          |
|-----------------------|--------------------------------|
| Rerank depth (K)      | {{rerank_top_k}}               |
| Pairs scored          | {{n_pairs}}                    |
| Wall-clock (rerank)   | {{rerank_seconds}} s           |
| Throughput            | {{queries_per_sec}} q/s, {{pairs_per_sec}} pairs/s |
| Peak RSS              | {{peak_memory_mib}} MiB        |
| Text resolution       | {{resolve_seconds}} s          |
| Batch size            | {{batch_size}}                 |
| Max length            | {{max_length}}                 |

The cost is **linear in K**: doubling the rerank depth roughly doubles
the wall-clock. Queries-per-second scales inversely.

## 6. Qualitative examples (before vs after)

Sampled queries where the cross-encoder moved the relevant passage:

{{case_studies}}

## 7. Discussion

{{discussion_bullets}}

## 8. Limitations

- **No fine-tuning**: `{{model_name}}` is the published MS MARCO baseline
  weights, used as-is. A larger or domain-tuned cross-encoder would
  shift the numbers further.
- **Sampled corpus**: the first-stage W4 dense run is on the qrels-
  anchored 50k sample, not the full 8.8M corpus. The reranker delta
  ({{delta_mrr10}} on MRR@10) should transfer qualitatively to the full
  corpus, but absolute numbers will differ.
- **Single first stage**: we only rerank the dense top-{{rerank_top_k}}.
  Reranking BM25 top-{{rerank_top_k}}, or a hybrid BM25+dense fusion,
  is the natural next step.
- **Latency**: the cross-encoder is the slowest component in the pipeline
  by ~2 orders of magnitude. Production-realistic deployments would
  use a smaller distilled reranker or restrict rerank to top-50.

## 9. Next

- Rerank BM25 top-100 (W2 run) and compare the delta to the dense
  delta — does the reranker recover more from a weaker first stage?
  **Delivered in W5-A; see §9.1.**
- Hybrid first-stage (RRF over BM25 + dense) → rerank.
- Connect the reranked top-K to the W3 RAG generator and measure
  whether better ordering improves answer quality.

### 9.1 W5 follow-ups

- *W5-A* — **done.** Cross-encoder rerank applied to the W2 BM25
  full-corpus top-100; head-to-head with the existing W5 dense+rerank.
  Numbers are not embedded here to keep the W5 snapshot stable; see
  the README §1 Week 5 follow-ups bullet for the live writeup.
  Headline: the naive recovery rate Δ / (1 − first_stage) makes Dense
  look better (40.5 % vs 22.3 % on MRR@10), but the *constrained*
  recovery Δ / (Recall@100 − first_stage) — which accounts for the
  fact that BM25's full-corpus Recall@100 is only 0.62, so 38 % of
  queries have no relevant doc for the reranker to promote at all —
  comes in essentially tied (BM25 41.1 %, Dense 42.4 %). The reranker
  does the same job on both first stages once the headroom is
  matched. Driver: `scripts/compare_rerank_first_stages.py`. Output:
  `outputs/week05_rerank_first_stage_compare/`.
- *W5-B* — **queued (post-deadline).** Perf–latency Pareto over
  K ∈ {50, 100, 200} on both first stages (1 000-query subsample for
  K=50 and K=200, K=100 reuses the W5-A / W5 full-dev runs). Driver:
  `scripts/run_w5b_k_sweep.py` with `--k-values`.

# Week 2: BM25 Retrieval Baseline

*Auto-generated {{generated_at}} from `outputs/week02_bm25/`. Do not edit by hand. Re-run
`python -m src.reporting.build_report --week week02` to refresh.*

## 1. Objective

Reproduce the standard MS MARCO Passage Ranking BM25 baseline using the
`bm25s` pure-Python backend, against the official corpus and dev/small
qrels.

## 2. Dataset

- **Corpus**: MS MARCO Passage Ranking collection ({{n_corpus}} passages)
- **Queries**: dev/small ({{n_queries_total}} queries; {{n_queries_eval}} have ≥1 positive qrel)
- **Relevance**: `qrels.dev.small`, binary judgments
- **Source**: loaded via `ir_datasets` (`msmarco-passage` and `msmarco-passage/dev/small`)

## 3. Method

- Tokenizer: `bm25s` default tokenizer with stopwords = `{{stopwords}}`
- Parameters: `k1 = {{k1}}`, `b = {{b}}`
- Top-{{top_k}} retrieved per query

Wall-clock:

- Indexing: {{indexing_seconds}} s
- Search: {{search_seconds}} s ({{search_ms_per_query}} ms / query)

## 4. Evaluation Metrics

- MRR@10
- nDCG@10
- Recall@100
- Recall@1000

## 5. Results

| Metric        | Value                |
|---------------|----------------------|
| MRR@10        | {{mrr_at_10}}        |
| nDCG@10       | {{ndcg_at_10}}       |
| Recall@100    | {{recall_at_100}}    |
| Recall@1000   | {{recall_at_1000}}   |
| # queries     | {{n_queries_eval}}   |

The published Anserini/Lucene BM25 baseline on this split is approximately
**MRR@10 ≈ 0.184**. The number above should be in the same ballpark; small
deviations are expected from tokenizer differences between Lucene and
`bm25s`.

## 6. Case Studies

Sampled queries where BM25 places a relevant passage in the top-10:

{{case_studies}}

## 7. Error Analysis

Sampled queries where BM25 fails to surface any relevant passage in the
top-10:

{{error_analysis}}

## 8. Limitations

- Pure lexical matching: BM25 cannot bridge surface vocabulary gaps. Failure
  cases concentrate on queries with paraphrasing, synonyms, or implicit
  reasoning.
- The dev/small qrels contain at most one positive judgment per query and
  are sparse, so true recall is likely underestimated.
- No hyperparameter tuning of `k1` / `b`.

## 9. Next Steps

- Add a Sentence-BERT bi-encoder retriever and a hybrid BM25 + dense
  fusion.
- Add a cross-encoder reranker over the BM25 top-100.
- Compare against the Anserini reference implementation to quantify the
  tokenizer gap.

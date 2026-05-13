# Week 4: Dense Retrieval Baseline (sampled corpus)

*Auto-generated {{generated_at}} from `outputs/week04_dense/`. Do not edit by hand.
Re-run `python -m src.reporting.build_report --week week04` to refresh.*

## 1. Objective

Build a Sentence-Transformers + FAISS dense retrieval baseline on a
**sampled** sub-corpus of MS MARCO Passage, and compare it head-to-head
against BM25 rebuilt on the **same** sample. We deliberately skip
encoding the full 8.8M corpus on a CPU laptop; the trade-off is spelled
out below.

## 2. Setup

- **Encoder**: `{{model_name}}` (no fine-tuning, generic semantic similarity).
- **Sample size**: {{sample_size}} passages, qrels-anchored.
- **Index**: `faiss.IndexFlatIP` over L2-normalised embeddings — exact
  inner-product search ≡ cosine similarity. No ANN approximation, so the
  numbers reflect the encoder, not the index.
- **Comparison BM25**: rebuilt on the **same** {{sample_size}}-passage
  sample with `bm25s` (k1={{k1}}, b={{b}}). This is the apples-to-apples
  comparison; the W2 full-corpus BM25 baseline (MRR@10 = 0.1703) is
  *not* directly comparable to either column below.
- **Eval queries**: dev/small ({{n_eval_queries}} of {{n_total_queries}}
  queries have ≥1 relevant doc landing in the sample).

## 3. Sampling note (important caveat)

A uniform random 50k sample from 8.8M passages would put almost no
relevant doc in the pool, so both retrievers would score near zero and
the comparison would be meaningless. We use **qrels-anchored sampling**
instead:

- Include every dev/small relevant doc_id ({{n_qrels_doc_ids_in_sample}}
  unique docs).
- Fill the rest with random distractors from the full corpus's doc_id
  pool (seed = {{seed}}).

The relevant doc is therefore *always* present in the pool, which makes
absolute MRR / Recall / nDCG higher than any "real" retrieval setting.
**The valid comparison is BM25-on-sample vs dense-on-sample**, not
either column vs the W2 full-corpus number.

## 4. Results

| Metric        | BM25 (sample) | Dense (sample) | Δ (dense − BM25) |
|---------------|---------------|----------------|------------------|
| MRR@10        | {{bm25_mrr10}}  | {{dense_mrr10}}  | {{delta_mrr10}}    |
| nDCG@10       | {{bm25_ndcg10}} | {{dense_ndcg10}} | {{delta_ndcg10}}   |
| Recall@100    | {{bm25_r100}}   | {{dense_r100}}   | {{delta_r100}}     |
| Recall@1000   | {{bm25_r1000}}  | {{dense_r1000}}  | {{delta_r1000}}    |

Wall-clock:

- Encode corpus: {{encode_seconds}} s ({{encode_per_doc_ms}} ms / passage)
- Dense search ({{n_total_queries}} queries × top-{{top_k}}): {{dense_search_seconds}} s
- BM25-sample build: {{bm25_build_seconds}} s
- BM25-sample search: {{bm25_search_seconds}} s

## 5. Qualitative examples

{{case_studies}}

## 6. Discussion

Concrete observations from this run (sampled, generic encoder, no fine-tune):

{{discussion_bullets}}

## 7. Limitations

- **Sampled corpus**: every dev/small relevant doc is unconditionally
  included. Numbers here are systematically higher than they would be on
  the full 8.8M corpus.
- **No fine-tuning**: `{{model_name}}` is pretrained for generic
  semantic similarity, not MS MARCO retrieval specifically. A
  fine-tuned encoder (e.g. `msmarco-MiniLM-L6-cos-v5`) would close most
  of the gap with leaderboard dense baselines.
- **No ANN approximation**: `IndexFlatIP` is exact and slow on big
  corpora. For the full corpus we'd want IVF-PQ / HNSW.
- **Single seed**: distractor sample is deterministic but not averaged.

## 8. Next

- Hybrid retrieval (BM25 + dense score fusion, e.g. RRF or weighted sum).
- Cross-encoder reranking over the BM25 top-100 (`ms-marco-MiniLM-L-6-v2`).
- Switch the encoder to a MS MARCO-tuned variant and re-run on the same sample.
- Scale the sample to 200k–500k passages to track how the BM25 ↔ dense
  gap shifts as the pool grows (relevant doc density drops).

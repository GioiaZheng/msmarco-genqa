---
title: "Week 3: RAG Generation Baseline"
date: "{{generated_at}}"
geometry: margin=1in
---

# Week 3: RAG Generation Baseline

*Auto-generated from `outputs/week03_generation/`. Do not edit by hand. Re-run
`python -m src.reporting.build_report --week week03` to refresh.*

## 1. Objective

Build an honest end-to-end RAG generation baseline on top of the Week 2 BM25
results. The goal is to measure how well a vanilla pretrained Seq2Seq model
answers MS MARCO QA questions when conditioned on the *real* top-k passages
returned by BM25 (rather than the toy 3-passage corpus used in the
prototype).

## 2. Pipeline

```
query
  ↓
BM25 retrieval (Week 2 run.tsv)
  ↓
top-{{top_k_passages}} passages
  ↓
prompt:  "question: <q> context: <p1> <p2> ..."
  ↓
{{model_name}} (no fine-tuning)
  ↓
answer
```

## 3. Dataset

- Queries: MS MARCO Passage Ranking dev/small (intersected with the QA v2.1
  validation split so that human-written reference answers are available).
- Reference answers: MS MARCO QA v2.1 `validation`, `answers` field (with
  `"No Answer Present."` filtered out).
- Number of evaluated queries: **{{n_eval}}**.

## 4. Model

- **{{model_name}}** loaded from HuggingFace, used in pretrained form (no
  fine-tuning).
- Max input tokens: {{max_input_length}}; max new tokens: {{max_new_tokens}}.

## 5. Evaluation Metrics

- **ROUGE-L** — longest-common-subsequence overlap (HF `rouge`, vs. the
  first reference).
- **BLEU** — corpus-level n-gram precision (HF `bleu`).
- **Exact Match (EM)** — SQuAD-style normalisation, best-of references.
- **Token F1** — SQuAD-style token overlap, best-of references.

## 6. Results

| Metric         | Value           |
|----------------|-----------------|
| ROUGE-L        | {{rouge_l}}     |
| BLEU           | {{bleu}}        |
| Exact Match    | {{exact_match}} |
| Token F1       | {{token_f1}}    |
| # predictions  | {{n_eval}}      |

These numbers are not directly comparable to leaderboard MS MARCO QA results
because (a) the model is pretrained-only, no SFT, and (b) the conditioning
context is the BM25 top-k, not the gold passage.

## 7. Qualitative Examples

{{examples}}

## 8. Error Analysis

Predictions with token-F1 = 0 against every reference (a strong signal that
the answer is wrong rather than just paraphrased):

{{error_analysis}}

## 9. Limitations

- The generation model is not fine-tuned on MS MARCO QA. Reference answers
  are short while pretrained T5/BART tend to produce longer, extractive-style
  outputs, which depresses overlap-based scores.
- Conditioning on BM25 top-k means retrieval errors propagate directly into
  generation. A reranking step (week 5) is expected to lift these numbers.
- Surface-form metrics (ROUGE/BLEU/EM) penalise valid paraphrases. Adding a
  semantic similarity metric (e.g. BERTScore) would give a fairer signal.

## 10. Next Steps

- Add a cross-encoder reranker between retrieval and generation.
- Add semantic generation metrics (BERTScore, optional NLI-based
  faithfulness).
- Supervised fine-tuning of the generation model on `(question, gold passage,
  answer)` triples drawn from MS MARCO QA v2.1.

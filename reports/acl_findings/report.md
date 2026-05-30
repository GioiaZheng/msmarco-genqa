# MS MARCO Generative QA with Dense Retrieval, Reranking, and Reproducible Evaluation

## Abstract

This report presents a reproducible retrieval-augmented question-answering
pipeline on MS MARCO dev/small. The system compares BM25 with
Sentence-BERT + FAISS dense retrieval, adds cross-encoder reranking, and
measures downstream generation with paired-bootstrap confidence intervals.
Replacing BM25 top-3 passages with cross-encoder-reranked dense passages
raises T5-small Token-F1 from 0.197 to 0.368 and ROUGE-L from 0.193 to
0.368 on 6,980 paired queries. The Token-F1 lift is +0.171 with a 95%
paired-bootstrap CI of [+0.163, +0.179].

## 1 Introduction

Retrieval-augmented generation can fail either because retrieval misses the
answer-bearing passage or because the generator fails to use retrieved
evidence. This project separates those effects by holding the generator and
prompt fixed while changing the retrieval source: BM25, dense retrieval, and
cross-encoder-reranked dense retrieval.

## 2 Method

The pipeline has six stages: BM25 retrieval, dense retrieval, cross-encoder
reranking, generation, paired-bootstrap evaluation, and grounding analysis.
Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2` with L2-normalised
embeddings in a FAISS inner-product index. Reranking uses
`cross-encoder/ms-marco-MiniLM-L-6-v2` over the dense top-100.

Generation uses a concatenation-style RAG prompt:

`question: <query> context: <passage_1> <passage_2> ...`

The default generator is T5-small. The repository includes a capacity-sweep
driver for T5-base and can be pointed at FLAN-T5 by setting
`--model-name google/flan-t5-base`.

## 3 Experiments

The full evaluation uses 6,980 MS MARCO dev/small queries. BM25 retrieval runs
against the 8.8M-passage corpus. Dense retrieval is evaluated on a qrels-anchored
50k-passage sample that contains every dev relevant passage plus sampled
distractors, making dense-vs-BM25 comparisons controlled within that sample.

Downstream generation is evaluated on the same paired query set for BM25 and
reranked retrieval sources. Metrics are ROUGE-L, BLEU, exact match, Token-F1,
and a BERTScore proxy.

## 4 Results

On the qrels-anchored 50k sample, dense retrieval improves MRR@10 from 0.695
for BM25-on-sample to 0.883. Cross-encoder reranking further improves MRR@10 to
0.930 while preserving Recall@100.

For generation, reranked retrieval roughly doubles surface-form metrics under
the same T5-small generator:

| Retrieval source | ROUGE-L | BLEU | EM | Token-F1 |
|---|---:|---:|---:|---:|
| BM25 | 0.1859 | 0.0717 | 0.0135 | 0.1966 |
| Reranked | 0.3621 | 0.2922 | 0.0606 | 0.3677 |
| Delta | +0.1763 | +0.2206 | +0.0471 | +0.1711 |

Paired-bootstrap intervals on 6,980 query pairs are strictly above zero for all
reported metrics. Token-F1 has CI [+0.1632, +0.1789], and ROUGE-L has CI
[+0.1663, +0.1820].

## 5 Analysis

Most generation regressions occur when reranking retrieves richer passages but
T5-small emits short or truncated fragments. Increasing `max_new_tokens` from
64 to 128 does not remove the effect, suggesting a model/prompt limitation
rather than a decoding-budget cap.

## 6 Reproducibility

Every major runner writes `manifest.json`, `resolved_config.yaml`, metrics, and
hashes for output artefacts. `configs/pipeline.yaml` defines the full command
sequence, and `scripts/run_pipeline.py --dry-run` prints the executable plan.
Experiment tracking defaults to local JSONL and optionally supports MLflow or
Weights & Biases.

## 7 Limitations

The dense retrieval result is sample-controlled rather than full-corpus. T5-base
and FLAN-T5 experiments are implemented as a sweep driver but should be rerun on
the final hardware before reporting as headline numbers.

## 8 Ethics Statement

The work uses public benchmark data and does not train on private user data.
Generated answers may still reproduce benchmark artefacts or hallucinate when
retrieval evidence is weak, so outputs should not be used as authoritative
answers without evidence inspection.

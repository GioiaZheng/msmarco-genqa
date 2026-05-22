# Experiment line

This document is the structural reference that the `experiments/` and
`scripts/` directories implement. It stitches the four pipeline stages —
BM25 retrieval, dense retrieval, cross-encoder reranking, RAG generation —
together with the evaluation and grounding layers that read their
outputs.

Reference numbers cited below are canonical; ongoing work that may
update some downstream numbers is flagged inline as **In flight**.
Provenance for each number (commit hash, manifest) is recorded under
*Reproducing the experiment line* at the bottom.

## Pipeline overview

```
                 ┌────────────────────────────────────────────────────────┐
                 │  MS MARCO dev/small  (6,980 queries, ~8.8M passages)   │
                 └────────────────────────────────────────────────────────┘
                                            │
       ┌────────────────────────────────────┼────────────────────────────────────┐
       ▼                                    ▼                                    │
  ╔════════════╗                      ╔════════════╗                             │
  ║   Stage 1  ║                      ║   Stage 2  ║                             │
  ║   BM25     ║   (full corpus)      ║   Dense    ║   (qrels-anchored 50k       │
  ║ retrieval  ║                      ║ retrieval  ║    sample, sentence-trans-  │
  ║            ║                      ║            ║    formers + FAISS)         │
  ╚═════╤══════╝                      ╚═════╤══════╝                             │
        │                                   │                                    │
        │     top-1000 / query              │ top-1000 / query                   │
        │                                   ▼                                    │
        │                            ╔════════════╗                              │
        │                            ║   Stage 3  ║                              │
        │                            ║   Cross-   ║   (MS MARCO MiniLM-L6        │
        │                            ║  encoder   ║    cross-encoder, top-100    │
        │                            ║  rerank    ║    rerank depth)             │
        │                            ╚═════╤══════╝                              │
        │                                  │                                     │
        │       top-3 / query              │ top-3 / query                       │
        ▼                                  ▼                                     │
  ╔════════════════════════════════════════════════════════════╗                 │
  ║                  Stage 4 — RAG generation                  ║                 │
  ║  T5-small (frozen), "question: ... context: ..." prompt    ║                 │
  ║  max_new_tokens=64; paired comparison on identical qid set ║                 │
  ╚════════════════════════════╤═══════════════════════════════╝                 │
                               │                                                 │
                               ▼                                                 │
  ╔═══════════════════════════════════════════════════════════════════════════╗  │
  ║    Evaluation: paired bootstrap CI (Token-F1 / ROUGE-L / BLEU / EM)       ║  │
  ║                BERTScore semantic proxy, regression taxonomy             ║  │
  ╠═══════════════════════════════════════════════════════════════════════════╣  │
  ║    Grounding: lexical content-token, n-gram, NLI entailment              ║◀─┘
  ╚═══════════════════════════════════════════════════════════════════════════╝
```

The headline finding is the **calibration** between the retrieval-side
gains (Stages 2/3 push MRR@10 well past Stage 1) and the
generation-side gains: ranks improve sharply, generated answers improve
on surface-form metrics, but the **grounding audit** shows the
generator is doing almost entirely extractive copying rather than
neural reasoning — and the **NLI entailment** signal moves the *other
way* compared to every other metric. See *Grounding audit* below.

---

## Stage 1 — BM25 first-stage retrieval

| What | Where |
|---|---|
| Driver | [`experiments/run_retrieval.py`](../experiments/run_retrieval.py) |
| Library | [`src/retrieval/bm25.py`](../src/retrieval/bm25.py) — `bm25s` wrapper with save/load + chunked retrieve |
| Config | [`configs/baseline.yaml`](../configs/baseline.yaml) — `retrieval.*` and `eval_retrieval.*` blocks |
| Index location | `data/processed/bm25_index_msmarco/` (~2.1 GB, gitignored) |
| Output | `outputs/week02_bm25/{run.tsv, metrics.json, examples.jsonl, manifest.json}` |

**Reference numbers** (6,980 dev/small queries, full 8.8M-passage corpus):

| Metric | Value |
|---|---:|
| MRR@10 | 0.1703 |
| Recall@100 | 0.6212 |
| Recall@1000 | 0.8154 |

Anserini/Lucene's published BM25 on the same split is MRR@10 ≈ 0.184;
the gap is consistent with tokenizer differences between `bm25s` and
Lucene's analyzer.

**Resume:** retrieval flushes `run.tsv` every `retrieval.chunk_size`
queries (default 200). Passing `--resume` picks up at the next chunk
boundary instead of restarting. See [`scripts/smoke_test_resume.py`](../scripts/smoke_test_resume.py)
for the end-to-end resume integration smoke.

---

## Stage 2 — Dense first-stage retrieval (sampled corpus)

| What | Where |
|---|---|
| Driver | [`experiments/run_dense_retrieval.py`](../experiments/run_dense_retrieval.py) |
| Library | [`src/retrieval/dense.py`](../src/retrieval/dense.py) (FAISS `IndexFlatIP` over L2-normalised embeddings); [`src/retrieval/sampling.py`](../src/retrieval/sampling.py) (qrels-anchored sub-corpus sampler) |
| Config | `configs/baseline.yaml`, `dense.*` block |
| Index location | `data/processed/dense_index_minilm_50k/` (gitignored) |
| Output | `outputs/week04_dense/{run.tsv, metrics.json, examples.jsonl, manifest.json}` |

The dense stage uses a **qrels-anchored 50k-passage sample** of the
full corpus rather than the full 8.8M-passage corpus. Every dev/small
relevant document is included by construction; random distractors fill
the remainder. This makes the comparison **dense-on-sample vs BM25-on-sample**
the apples-to-apples one — neither number is comparable to Stage 1's
full-corpus MRR@10.

**Reference numbers** (50k sample, `all-MiniLM-L6-v2`):

| Metric | Dense | BM25-on-sample | Δ |
|---|---:|---:|---:|
| MRR@10 | 0.8830 | 0.6948 | +0.1882 |
| nDCG@10 | 0.9041 | — | — |
| Recall@100 | 0.9946 | 0.9338 | +0.0608 |

Both numbers are **upper-bounded** by qrels-anchoring. The takeaway is
the gap, not the absolute level.

### Encoder horizontal ablation

[`scripts/run_w4b_encoder_horizontal.py`](../scripts/run_w4b_encoder_horizontal.py)
swaps the encoder on the identical 50k sample. (Slated for rename to
`scripts/run_encoder_comparison.py` in a follow-up cleanup commit.)

| Encoder | MRR@10 | nDCG@10 | Recall@100 | ms/passage |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` (baseline) | 0.8830 | 0.9041 | 0.9946 | 16.2 |
| `all-MiniLM-L12-v2` | 0.8933 | 0.9131 | 0.9955 | 31.3 |
| **`BAAI/bge-small-en-v1.5`** | **0.9021** | **0.9196** | **0.9967** | 34.4 |

bge-small lifts MRR@10 by +0.019 over the baseline at roughly 2× the
CPU encoding cost. ANCE / MS MARCO-tuned encoders are deferred.

### Density sweep ablation

[`scripts/run_w4a_density_sweep.py`](../scripts/run_w4a_density_sweep.py)
runs bge-small on three sample sizes (15k / 30k / 50k passages). As
the sample grows, the relevant-doc density drops and dense's advantage
over BM25-on-sample grows monotonically:

| Sample size | Density of relevants | Δ MRR@10 (dense − BM25) |
|---:|---:|---:|
| 15,000 | 49.6 % | +0.166 |
| 30,000 | 24.8 % | +0.188 |
| 50,000 | 14.9 % | +0.207 |

True 1 % / 5 % / 10 % density cells would need 70k–700k samples
(dev/small has ~7,437 unique relevants) and are deferred.

---

## Stage 3 — Cross-encoder reranking

| What | Where |
|---|---|
| Driver | [`experiments/run_reranker.py`](../experiments/run_reranker.py) |
| Library | [`src/reranking/cross_encoder.py`](../src/reranking/cross_encoder.py), [`src/reranking/io.py`](../src/reranking/io.py) |
| Config | `configs/baseline.yaml`, `reranker.*` block |
| Output | `outputs/week05_reranker_full/{run.tsv, metrics.json, examples.jsonl, manifest.json}` |

Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` over the **W4
dense top-100** per query. Reranking is order-only, so Recall@100 is
unchanged by construction.

**Reference numbers** (6,980 dev/small queries, full pipeline):

| Metric | Dense | + CE rerank | Δ |
|---|---:|---:|---:|
| MRR@10 | 0.8830 | 0.9304 | **+0.0474** |
| nDCG@10 | 0.9041 | 0.9434 | +0.0393 |
| Recall@100 | 0.9946 | 0.9946 | 0 |

Runtime on a 6-core MacBook CPU: ~4 h 37 min for the full dev/small
(538k (query, passage) pairs at ~32 pairs/s; peak RSS ~3.3 GiB).

### k-sweep ablation

[`scripts/run_w5b_k_sweep.py`](../scripts/run_w5b_k_sweep.py) varies
`rerank_top_k` to characterise the depth/quality trade-off. (Slated
for rename to `scripts/run_topk_sweep.py`.)

---

## Stage 4 — RAG generation

| What | Where |
|---|---|
| Driver | [`experiments/run_generation_baseline.py`](../experiments/run_generation_baseline.py) |
| Library | [`src/generation/rag_generator.py`](../src/generation/rag_generator.py) |
| Config | `configs/baseline.yaml`, `generation.*` block |
| Output | `outputs/week03_generation_{bm25,reranked}_full/{predictions.jsonl, metrics.json, manifest.json}` |

T5-small (frozen, no fine-tuning), `question: ... context: ...`
prompt, top-3 passages from the upstream retrieval source folded into
the prompt, `max_new_tokens=64`. The runner is **retrieval-source
agnostic** — pass any TREC-format `run.tsv` (BM25 / dense / reranked)
via `--input-run`. Use `--restrict-to-run` to force two generation
runs to evaluate on the SAME qid set when their upstream retrievers
cover different sets.

The headline paired comparison is **BM25-top-3 vs Reranked-top-3**,
same T5-small, mutually restricted to the **identical 6,980-query
set** (verified by qid set-diff = ∅). See *Evaluation* below for the
numbers.

---

## Evaluation

| What | Where |
|---|---|
| Paired bootstrap CI (per-metric) | [`scripts/bootstrap_generation_comparison.py`](../scripts/bootstrap_generation_comparison.py) → [`src/evaluation/bootstrap.py`](../src/evaluation/bootstrap.py) |
| BERTScore semantic proxy | [`scripts/bertscore_paired_eval.py`](../scripts/bertscore_paired_eval.py) → [`src/evaluation/bertscore.py`](../src/evaluation/bertscore.py) |
| Surface metrics | [`src/evaluation/generation.py`](../src/evaluation/generation.py) — ROUGE-L / BLEU / Exact-Match / Token-F1 |
| Regression taxonomy | [`scripts/regression_failure_taxonomy.py`](../scripts/regression_failure_taxonomy.py) (40-query seeded triage) |
| Query-form analysis | [`scripts/tag_query_forms.py`](../scripts/tag_query_forms.py), [`scripts/analyze_rerank_by_query_form.py`](../scripts/analyze_rerank_by_query_form.py) |
| Retrieval metrics | [`src/evaluation/retrieval.py`](../src/evaluation/retrieval.py) — MRR / Recall / nDCG |

**Reference paired-bootstrap numbers** (6,980 paired qids, BM25 → Reranked):

| Metric | BM25 | Rerank | Δ | 95 % CI |
|---|---:|---:|---:|---|
| Token-F1 | 0.197 | 0.368 | **+0.171** | [+0.163, +0.178] |
| ROUGE-L | 0.193 | 0.368 | **+0.174** | [+0.166, +0.181] |
| BLEU    | — | — | +0.1325 | (strictly > 0) |
| EM      | — | — | +0.0471 | (strictly > 0) |

**BERTScore semantic proxy** (3,000-paired-qid subsample, DistilBERT,
same paired-bootstrap convention): Δ +0.173 with 95 % CI strictly
above zero. The semantic proxy moves in lockstep with the surface
metrics — at least under this DistilBERT-based scorer.

**Decoding-budget falsification** (`max_new_tokens` 64 → 128, identical
upstream): rerank Δ is statistically and practically unchanged.
Token-F1 Δ +0.1706 (CI [+0.163, +0.178]), ROUGE-L Δ +0.1736. The
truncation share in the regression bucket only drops 90.0 % → 87.5 %.
T5-small is hitting EOS naturally on this prompt, not running out of
decode budget; the mid-word-ending output style is intrinsic to the
model.

---

## Grounding audit

| What | Where |
|---|---|
| Driver | [`scripts/grounding_audit.py`](../scripts/grounding_audit.py) |
| Lexical + n-gram | [`src/evaluation/grounding.py`](../src/evaluation/grounding.py) |
| NLI entailment | [`src/evaluation/nli_grounding.py`](../src/evaluation/nli_grounding.py) — `cross-encoder/nli-deberta-v3-small` |
| Correlation analysis | [`scripts/grounding_correlation.py`](../scripts/grounding_correlation.py) |
| Case study | [`scripts/low_grounding_case_study.py`](../scripts/low_grounding_case_study.py) |

A cheap, deterministic CPU pass over the same paired predictions used
in Evaluation. No new generation. The question: is T5-small extracting
from the prompt, or generating from parametric memory?

**Three grounding signals** (6,980 paired qids for lex / n-gram; 3,000
for NLI):

| Signal | BM25 | Rerank | Δ | 95 % CI |
|---|---:|---:|---:|---|
| Lexical content-token | 0.9972 | 0.9977 | +0.0005 | [−0.0003, +0.0014] |
| 3-gram                | 0.9873 | 0.9905 | +0.0032 | [+0.0015, +0.0050] |
| **NLI entailment**    | 0.2270 | 0.0821 | **−0.1448** | [−0.1597, −0.1297] |

Two read-outs:

1. **Both arms sit at the lexical/3-gram ceiling.** Almost every
   content word T5-small emits is already in the prompt, and 99 % of
   its 3-grams appear as contiguous spans in some single passage.
   T5-small on this prompt is effectively performing extractive QA;
   the Stage 4 rerank gain on surface-form metrics is downstream of
   retrieval putting the right tokens into the prompt, not of any
   neural reasoning improvement.

2. **NLI entailment reverses sign.** This is the only metric whose Δ
   crosses zero in the negative direction. The likely mechanism:
   T5-small on reranked top-3 emits more fragmentary / mid-word-cut
   snippets that inflate word and 3-gram overlap (positive lex /
   n-gram / Token-F1) but score low on a sentence-level NLI
   cross-encoder which cannot entail a fragmentary hypothesis.

The capacity-swap experiment that disambiguates whether this is a
tiny-generator artefact or a real retrieval phenomenon is **In flight**;
see *Phase A — generator capacity sweep* below.

---

## Ablations and follow-ups

### Phase A — generator capacity sweep (**In flight**)

[`scripts/run_w7b_generator_comparison.py`](../scripts/run_w7b_generator_comparison.py)
(slated for rename to `scripts/run_generator_capacity_sweep.py`) runs
T5-base on the identical 6,980 paired-qid set with the identical
prompt, then recomputes the full metric suite (surface metrics +
BERTScore + lex / n-gram / NLI grounding).

The smoke (n=50) showed NLI Δ = **−0.221**, more negative than
T5-small's −0.145. If this persists at full sample, the sign-reversal
is not a tiny-generator artefact: the paper framing then shifts from a
capacity narrative toward a metric-methodology calibration. The
full-sample outcome determines the direction.

### Other ablations

| Ablation | Script | Status |
|---|---|---|
| Density sweep on dense retrieval (15k/30k/50k) | [`scripts/run_w4a_density_sweep.py`](../scripts/run_w4a_density_sweep.py) | Done; see Stage 2 |
| Dense encoder horizontal | [`scripts/run_w4b_encoder_horizontal.py`](../scripts/run_w4b_encoder_horizontal.py) | Done; see Stage 2 |
| Rerank depth (k-sweep) | [`scripts/run_w5b_k_sweep.py`](../scripts/run_w5b_k_sweep.py) | Done |
| Regression vs non-regression query profile | [`scripts/regression_query_profile.py`](../scripts/regression_query_profile.py) | Done |
| Generation × retrieval source first-stage comparison | [`scripts/compare_rerank_first_stages.py`](../scripts/compare_rerank_first_stages.py) | Done |
| Validation of full reranker run | [`scripts/validate_full_rerank.py`](../scripts/validate_full_rerank.py) | Done |
| Generator capacity (T5-small → T5-base) | [`scripts/run_w7b_generator_comparison.py`](../scripts/run_w7b_generator_comparison.py) | **In flight (Phase A)** |

---

## Reproducing the experiment line

The full pipeline is driven from `experiments/run_*.py` with config in
[`configs/baseline.yaml`](../configs/baseline.yaml). End-to-end on a
6-core MacBook CPU is roughly:

| Stage | Wall-clock (cold) | Wall-clock (warm) |
|---|---:|---:|
| BM25 retrieve (full corpus, 6,980 queries) | ~1 h 40 min (incl. download + index) | ~70 min |
| Dense encode (50k sample) + FAISS search | ~14 min | ~20 s |
| Cross-encoder rerank (top-100 over 6,980 queries) | ~4 h 37 min | ~4 h 37 min |
| Generation × 2 arms (T5-small, 6,980 queries each) | ~2 h | ~2 h |
| Evaluation + grounding audit | ~5 min | ~5 min |

Every runner writes a `manifest.json` next to its outputs capturing
git commit, command line, config hash, and dependency hashes —
sufficient to re-identify the run six months later. See
[`src/util/manifest.py`](../src/util/manifest.py) and
[`src/util/environment.py`](../src/util/environment.py).

Three canonical runs predate the manifest plumbing and have no runtime
`manifest.json`: the W2 BM25 full-corpus retrieval, the W4 dense
baseline (50k sample), and the W5 reranker over the full BM25 run.
Each of those output directories carries a `provenance.backfill.json`
instead — a separate file with a deliberately distinct schema string
(`msmarco-genqa.backfilled-provenance.v1`) and an explicit `unknown`
block enumerating what cannot be recovered from outside-of-runtime
information: the exact production commit, the CLI argv, the wall-clock
timestamp, whether the tree was dirty, the installed package versions,
the input-file byte identities, and the effective coverage of the
seed-42 promise. See
[`scripts/backfill_provenance.py`](../scripts/backfill_provenance.py).

Deterministic seed (`seed: 42`) is set in `configs/baseline.yaml` and
threaded through the samplers, the dense encoder batching, and the
evaluation bootstrap. For an environment that reproduces the numbers
below, install [`requirements-lock.txt`](../requirements-lock.txt)
rather than the loose [`requirements.txt`](../requirements.txt).

The frozen reference snapshot is
[`reports/internship_report/report.pdf`](../reports/internship_report/report.pdf)
at tag `v1.0-internship-final`; numbers in this document supersede that
snapshot only where annotated **In flight**.

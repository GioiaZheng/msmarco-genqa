# MS MARCO GenQA

## TL;DR

A reproducible, single-machine end-to-end MS MARCO retrieval-augmented QA
pipeline built up across six experimental stages:

> **W1 EDA → W2 BM25 retrieval → W3 RAG generation (T5-small) → W4 dense
> retrieval (SBERT + FAISS) → W5 cross-encoder reranking → W6 semantic-proxy
> evaluation + regression-failure taxonomy.**

**Headline result.** Swapping the first-stage retriever from BM25 to a
cross-encoder-reranked dense top-3 — same T5-small generator, same 6 980
dev/small queries — roughly doubles every generation metric: Token-F1
0.197 → 0.368 (Δ +0.171), ROUGE-L 0.193 → 0.368 (Δ +0.174), with 95 %
paired-bootstrap CIs strictly above 0 on all four surface-form metrics.
A DistilBERT-based BERTScore proxy on a 3 000-pair subsample recovers
Δ +0.173, so the gain is not a surface-form artefact. A
`max_new_tokens=64→128` sweep on full dev/small leaves all four deltas
within <0.005, ruling out the decode-budget reading.

**How to read this repo.** Per-stage write-up with numbers in
[§1 Status](#1-status); reproduction commands in
[§4 Run the official baselines](#4-run-the-official-baselines); frozen
final PDF at `reports/internship_report/report.pdf`. The repo is
batch-eval oriented — no one-shot `--question` CLI; see
[§4.5](#45-single-query-demo) for the minimal in-Python composition.

## Project layout

- **`experiments/`** — four pipeline-stage runners (BM25, dense, rerank, generation). Source of the benchmark numbers.
- **`scripts/`** — analyses, ablations, validation, integration smokes that read `experiments/` outputs.
- **`src/`** — importable library code backing both.
- **`reports/internship_report/`** — frozen v1.0 PDF + sources.
- **`docs/experiments.md`** — pipeline narrative stitching the stages together.

See [§2 Directory layout](#2-directory-layout) for the full breakdown.

## 1. Status

### Stage 1 — EDA

Dataset statistics + query/passage/answer-type distributions covered in §1 of
[`reports/internship_report/report.pdf`](reports/internship_report/report.pdf).
Source figures: `figures/{query_length,passage_length,query_type,answer_type_by_query_type}_distribution.png`.

### Stage 2 — BM25 retrieval

[`experiments/run_retrieval.py`](experiments/run_retrieval.py). On `dev/small`
(6 980 queries, full 8.8 M-passage corpus):
**MRR@10 = 0.1703**, Recall@100 = 0.6212, Recall@1000 = 0.8154.

### Stage 3 — RAG generation (200-query sample)

[`experiments/run_generation_baseline.py`](experiments/run_generation_baseline.py).
200-query sample of dev/small (seed 42), T5-small (no fine-tuning),
top-3 BM25 passages:
ROUGE-L = 0.1626, BLEU = 0.0574, EM = 0.0050, Token-F1 = 0.1756.
The full-dev comparison against a reranked upstream lives in
*Generation × retrieval source* below.

### Stage 4 — Dense retrieval (qrels-anchored 50 k sample)

[`experiments/run_dense_retrieval.py`](experiments/run_dense_retrieval.py).
`all-MiniLM-L6-v2`, FAISS `IndexFlatIP` over L2-normalised embeddings.

Dense MRR@10 = 0.8830, nDCG@10 = 0.9041, Recall@100 = 0.9946 — vs
BM25-on-sample MRR@10 = 0.6948, Recall@100 = 0.9338 (same 50 k pool).
Numbers are upper-bounded by the sampling — every dev relevant doc is
in the pool. Read the comparison as dense-vs-BM25 on the same sample,
not against the W2 full-corpus number.

**W4 follow-ups:**

- *Same-tier encoder horizontal* on the identical 50 k sample
  ([`scripts/run_encoder_horizontal.py`](scripts/run_encoder_horizontal.py)):

  | Encoder | MRR@10 | nDCG@10 | Recall@100 | ms/passage |
  |---|---:|---:|---:|---:|
  | `all-MiniLM-L6-v2` (baseline) | 0.8830 | 0.9041 | 0.9946 | 16.2 |
  | `all-MiniLM-L12-v2` | 0.8933 | 0.9131 | 0.9955 | 31.3 |
  | **`BAAI/bge-small-en-v1.5`** | **0.9021** | **0.9196** | **0.9967** | 34.4 |

  `bge-small` lifts MRR@10 by +0.019 at ~2× the CPU encoding cost.
- *Density sensitivity* with `bge-small` at three sample sizes
  ([`scripts/run_density_sweep.py`](scripts/run_density_sweep.py)).
  Δ MRR@10 over BM25-on-sample grows monotonically with sample size:
  +0.166 (49.6 % density, 15 k) → +0.188 (24.8 %, 30 k) → +0.207 (14.9 %, 50 k).

### Stage 5 — Cross-encoder reranking

[`experiments/run_reranker.py`](experiments/run_reranker.py).
`cross-encoder/ms-marco-MiniLM-L-6-v2` over the W4 dense top-100.
Full dev/small (6 980 queries):

| Metric      | Dense (W4) | + CE rerank | Δ          |
|-------------|-----------:|------------:|-----------:|
| MRR@10      | 0.8830     | 0.9304      | **+0.0474** |
| nDCG@10     | 0.9041     | 0.9434      | +0.0393    |
| Recall@100  | 0.9946     | 0.9946      | +0.0000    |

Runtime ~4h37m on a 6-core MacBook (538 000 query-passage pairs at
~32 pairs/s, peak RSS ~3.3 GiB). Recall@100 is unchanged because
reranking only reorders the top-100. A 1 000-query pilot produced
MRR Δ +0.0435, nDCG Δ +0.0398 — close to the full-dev deltas, so the
gain is not specific to the subsample.

### Generation × retrieval source

Cross-stage comparison on **full dev/small (6 980 queries)**: same
T5-small (no fine-tuning), same top-3 passages, only the retrieval
source changes. Mutually restricted via `--restrict-to-run` so both
runs cover the same 6 980 qids (qid set-diff = ∅).

| Retrieval source → T5-small | ROUGE-L | BLEU | EM | Token-F1 |
|---|---:|---:|---:|---:|
| BM25     | 0.1859 | 0.0717 | 0.0135 | 0.1966 |
| Reranked | **0.3621** | **0.2922** | **0.0606** | **0.3677** |
| **Δ (rerank − BM25)** | **+0.1763** | **+0.2206** | **+0.0471** | **+0.1711** |

Reranking roughly doubles every generation metric. The retrieval-flag
rate (at least one relevance-judged passage in top-3) jumps from
20.8 % (BM25) to 96.9 % (reranked). Per-query token-F1 splits:
4 015 strict improvements / 1 766 regressions / 1 199 ties — net
+2 249 / 6 980 queries (32 %) strictly improved.

#### Paired-bootstrap 95 % CI on Δ (rerank − BM25)

Full 6 980 paired qids, 10 000 bootstrap resamples, seed 42 (per-query
ROUGE-L and BLEU from `rouge_score.RougeScorer` and NLTK `sentence_bleu`
smoothing-1):

| Metric          | Δ       | 95 % CI on Δ          | p₂      |
|-----------------|--------:|----------------------:|--------:|
| ROUGE-L         | +0.1742 | [+0.1663, +0.1820]    | < 0.001 |
| BLEU (sentence) | +0.1330 | [+0.1265, +0.1395]    | < 0.001 |
| Exact-Match     | +0.0471 | [+0.0417, +0.0527]    | < 0.001 |
| Token-F1        | +0.1711 | [+0.1632, +0.1789]    | < 0.001 |

#### Bucket analysis by query type (token-F1)

| query_type     |    n |   BM25 |  Reranked |        Δ |
|----------------|-----:|-------:|----------:|---------:|
| DESCRIPTION    | 3725 | 0.1889 |    0.3939 | **+0.2050** |
| ENTITY         |  631 | 0.1765 |    0.3186 | +0.1421  |
| LOCATION       |  498 | 0.2495 |    0.3928 | +0.1433  |
| NUMERIC        | 1665 | 0.1997 |    0.3235 | +0.1238  |
| PERSON         |  461 | 0.2186 |    0.3557 | +0.1371  |

`DESCRIPTION` (53 % of eval) benefits most; `NUMERIC` benefits least.

#### Semantic-proxy BERTScore (3 000-paired subsample)

DistilBERT-based BERTScore (rescaled, paired-bootstrap CI; seed 42):
BM25 0.2192 → Reranked 0.3920, **Δ +0.1728, 95 % CI [+0.1608, +0.1850],
p < 0.001**. Per-query: rerank strictly better 64.8 %, tie 5.7 %,
BM25 strictly better 29.5 %.

The semantic-proxy Δ sits within a hair of the surface-form Token-F1 Δ
(+0.1711) and ROUGE-L Δ (+0.1742). The rerank gain shows up in a
semantic-similarity scorer at the same magnitude, so it is not a
surface-form artefact. DistilBERT is a proxy for the conventional
`roberta-large` BERTScore — for cross-paper comparison re-run with
`--model-type microsoft/deberta-xlarge-mnli` or
`--model-type roberta-large`.

#### Regression failure taxonomy (40-query triage)

233 of 6 980 paired qids land in the `regression` bucket (reranker
brings a relevant passage into top-3 but token-F1 drops vs BM25). A
40-query seeded triage with deterministic heuristic labels
([`scripts/regression_failure_taxonomy.py`](scripts/regression_failure_taxonomy.py),
seed 42):

| label                    |  n | share |
|--------------------------|---:|------:|
| `truncation_midword`     | 22 | 55 %  |
| `truncation_short`       | 14 | 35 %  |
| `topic_drift`            |  2 |  5 %  |
| `extractive_passage_bias`|  2 |  5 %  |
| `semantic_mismatch`      |  0 |  0 %  |

~90 % of regressions are generation-side output style — the reranker
brings in richer passages, but T5-small extracts short or mid-sentence
fragments. Example (qid 49802, "belizean cuisine"): BM25 produces a
24-token description; reranked produces "Belizean cuisine" (the
passage title).

### Stage 6 — Evaluation layer follow-ups

- **Decoding-budget closure** (`max_new_tokens=64→128`, full dev/small).
  Same generator, prompts, retrieval inputs, seed; only `max_new_tokens`
  changes. Token-F1 Δ +0.1706, ROUGE-L Δ +0.1736, BLEU Δ +0.1325,
  EM Δ +0.0471 — all four CIs strictly above 0. Regression bucket
  shrinks 233 → 231; truncation share 90.0 % → 87.5 %. The
  budget-cap reading is falsified: T5-small hits EOS naturally, not a
  decode-budget wall. The mid-word output style is intrinsic to the
  model on this prompt format.
- **W6-A/B/C question-form analyses** (no new generation; offline
  analyses over the existing per-query metrics):
  [`scripts/tag_query_forms.py`](scripts/tag_query_forms.py) tags all
  6 980 queries; [`scripts/analyze_rerank_by_query_form.py`](scripts/analyze_rerank_by_query_form.py)
  reports rerank Δ per form. `which` (n = 120) is the only form whose
  95 % CI on ΔToken-F1 includes zero (Δ = +0.0250, CI [−0.028, +0.077],
  p = 0.33) despite a retrieval-side ΔMRR@10 of +0.71 — the reranker
  improves retrieval but the gain does not transfer downstream on
  selection-style queries. Factual wh-forms (what / where / why /
  how / who) all see Δ in +0.12 to +0.23 with CIs strictly above 0.
  [`scripts/regression_query_profile.py`](scripts/regression_query_profile.py):
  Mann-Whitney on five structural features finds no meaningful
  query-side difference between regressions and the rest.

### Stage 7 — Grounding audit

Deterministic CPU pass over the existing full-dev paired predictions,
asking what T5-small is actually doing on this prompt format
([`scripts/grounding_audit.py`](scripts/grounding_audit.py); ~2 s
scoring + bootstrap, no model load):

| Metric                          | BM25    | Reranked | Δ        | 95 % CI on Δ         |
|---------------------------------|--------:|---------:|---------:|---------------------:|
| Lexical content-token grounding | 0.9972  | 0.9977   | +0.0005  | [−0.0003, +0.0014]   |
| 3-gram grounding                | 0.9873  | 0.9905   | +0.0032  | [+0.0015, +0.0050]   |
| NLI entailment (3 000-pair)     | 0.2270  | 0.0821   | **−0.1448** | [−0.1597, −0.1297] |

Both arms sit at the lexical ceiling — T5-small on
`question: ... context: ...` is doing extractive QA. The surface-form
rerank gain (ΔToken-F1 +0.171) is downstream of retrieval: better
passages → better extractive output.

The NLI Δ is the one metric in this project whose sign reverses vs
the surface-form story. Likely mechanism (to be confirmed by a
generator swap): reranked top-3 produces more fragmentary / mid-word
snippets that inflate word + 3-gram overlap but the sentence-level
NLI cross-encoder cannot entail a fragmentary hypothesis.
[`src/msmarco_genqa/evaluation/nli_grounding.py`](src/msmarco_genqa/evaluation/nli_grounding.py)
implements the score; driver is `grounding_audit.py --nli-n-pairs 3000`.

A 30-query seeded triage of low-grounding rerank outputs
([`scripts/low_grounding_case_study.py`](scripts/low_grounding_case_study.py)):
77 % `paraphrase_reorder` (content words present, order/phrasing
differs), 23 % `partial_external` (mostly tokeniser artefacts:
`350oF` vs `350°F`; `competed` vs `compete`), 0 %
`parametric_or_external`. No genuine hallucinations in the sample.

### Reference points

Published Anserini/Lucene BM25 on MS MARCO `dev/small`: MRR@10 ≈ 0.184.
Our `bm25s`-based 0.1703 is in the same ballpark; the gap is mostly
tokenizer-induced.

## 2. Directory layout

```
configs/             baseline.yaml — paths + retrieval/generation/reranker/eval knobs
experiments/         four pipeline-stage runners
scripts/             analysis, drivers, ablations, validation, smokes
src/                 importable library code backing experiments/ and scripts/
  data/                msmarco.py — ir_datasets loader for the official corpus
  retrieval/           bm25.py, dense.py, sampling.py
  reranking/           cross_encoder.py, io.py
  generation/          rag_generator.py — T5/BART RAG generator
  evaluation/          retrieval.py, generation.py, grounding.py, nli_grounding.py,
                       bertscore.py, bootstrap.py, query_form.py
  util/                manifest.py, environment.py — per-run provenance
tests/               pytest suite (no network, no models)
docs/                experiments.md — pipeline narrative (BM25 → dense → rerank → gen)
reports/
  internship_report/   report.tex + report.pdf + figures/ (committed, frozen at v1.0)
figures/             plots used in the report (committed)
outputs/             run.tsv, metrics.json, examples.jsonl, manifest.json per stage (gitignored)
data/                raw/, processed/, cache/ — all gitignored, .gitkeep tracked
```

### `experiments/` vs `scripts/`

- **`experiments/`** — the four pipeline-stage runners. Each consumes the
  official MS MARCO corpus (via `ir_datasets`) and produces a structured
  output directory under `outputs/`:

  | Runner | Stage |
  |---|---|
  | [`experiments/run_retrieval.py`](experiments/run_retrieval.py) | BM25 first-stage retrieval |
  | [`experiments/run_dense_retrieval.py`](experiments/run_dense_retrieval.py) | Dense first-stage retrieval (sampled) |
  | [`experiments/run_reranker.py`](experiments/run_reranker.py) | Cross-encoder reranking |
  | [`experiments/run_generation_baseline.py`](experiments/run_generation_baseline.py) | RAG generation |

  These runners produce the numbers cited in
  [`reports/internship_report/report.pdf`](reports/internship_report/report.pdf)
  and in [`docs/experiments.md`](docs/experiments.md).

- **`scripts/`** — everything that reads or analyses outputs of
  `experiments/`, plus ablation drivers, validation, and integration
  smokes. Examples:

  | Kind | Examples |
  |---|---|
  | Evaluation drivers | `bootstrap_generation_comparison.py`, `bertscore_paired_eval.py`, `grounding_audit.py`, `grounding_correlation.py` |
  | Failure / case analysis | `regression_failure_taxonomy.py`, `regression_query_profile.py`, `low_grounding_case_study.py` |
  | Slicing / tagging | `tag_query_forms.py`, `analyze_rerank_by_query_form.py`, `analyze_generation_rerank.py` |
  | Ablation drivers | `run_density_sweep.py`, `run_encoder_horizontal.py`, `run_k_sweep.py`, `run_generator_capacity_sweep.py` |
  | End-to-end driver | `run_full_generation_and_analysis.py` |
  | Validation / smoke | `validate_full_rerank.py`, `smoke_test_resume.py` |

  Scripts may change shape as analyses evolve; only the `experiments/`
  output schema is held fixed.

Everything runs from the project root. Scripts add `PROJECT_ROOT` to
`sys.path` themselves; no `PYTHONPATH` needed.

## 3. Setup

Python 3.10+ recommended (3.9 also works).

```bash
pip install -r requirements.txt
pip install -e .                       # register `src` as a real package
```

To reproduce the numbers in the reports, pin to the lockfile instead:

```bash
pip install -r requirements-lock.txt
pip install -e .
```

Or, equivalently:

```bash
make install
```

Optional, only for PDF report generation:

```bash
brew install pandoc                    # macOS
brew install --cask basictex           # macOS LaTeX
# Linux: sudo apt-get install pandoc texlive-xetex
```

## 4. Run the official baselines

Each runner is exposed both as a Python script and as a console script
(see `pyproject.toml [project.scripts]`):

```bash
python experiments/run_retrieval.py    # script form
mgq-retrieve                            # console form
```

Console names: `mgq-retrieve`, `mgq-dense`, `mgq-rerank`, `mgq-generate`.
The examples below use the script form.

### Stage 2 — BM25

```bash
python experiments/run_retrieval.py
```

First run: ~5 min download (~1 GB), ~15 min `ir_datasets` encoding fix,
~10 min `bm25s` index build, ~70 min retrieve. Total ≈ 1h40m on a
16 GB MacBook. Subsequent runs reuse
`data/processed/bm25_index_msmarco/` (2.1 GB) and only retrieve.

```bash
python experiments/run_retrieval.py --resume         # picks up at next chunk boundary
python experiments/run_retrieval.py --rebuild-index  # force fresh index
```

Outputs: `outputs/week02_bm25/{metrics.json, run.tsv, examples.jsonl, manifest.json}`.

### Stage 3 — RAG generation

Requires Stage 2 output `outputs/week02_bm25/run.tsv`.

```bash
python experiments/run_generation_baseline.py
```

Default: 200 dev queries, T5-small, top-3 BM25 passages. CPU runtime
~5–15 min. Tunable knobs in `configs/baseline.yaml`:
- `generation.model_name` (`t5-small`, `t5-base`, `facebook/bart-base`, …)
- `generation.num_eval_queries`
- `generation.top_k_passages`

The runner is retrieval-source agnostic — feed it any TREC-format
`run.tsv` via CLI flags:

```bash
python experiments/run_generation_baseline.py \
    --input-run outputs/week05_reranker/run.tsv \
    --output-dir outputs/week03_generation_reranked \
    --retrieval-source reranked
```

Use `--restrict-to-run <other_run.tsv>` to force two runs to evaluate on
the same query subsample even when their upstream retrievers cover
different sets — used in *Generation × retrieval source* above.

### Stage 4 — Dense retrieval

Requires Stage 2 to have produced
`data/processed/bm25_index_msmarco/doc_ids.json` (sampler input).

```bash
python experiments/run_dense_retrieval.py
```

First run: ~13.5 min to encode 50 k passages on a 6-core CPU
(`all-MiniLM-L12-v2` takes ~26 min); ~20 s FAISS search. Subsequent
runs reuse the cached index. Tunable knobs:
- `dense.model_name` (e.g. `sentence-transformers/msmarco-MiniLM-L6-cos-v5`)
- `dense.sample_size` (default 50 000)
- `dense.compare_bm25_on_sample`

### Stage 5 — Cross-encoder reranking

Requires Stage 4 output `outputs/week04_dense/run.tsv`.

```bash
python experiments/run_reranker.py
```

Reranks W4 dense top-100 per query. CPU runtime scales linearly with
the number of queries — full 6 980 × top-100 is ~6 h on a 6-core
laptop. Use `--num-eval-queries` for a deterministic subsample.

```bash
python experiments/run_reranker.py --num-eval-queries 50 --rerank-top-k 100   # smoke (~1 min)
OMP_NUM_THREADS=12 python experiments/run_reranker.py --num-eval-queries 1000  # ~50 min
```

Tunable knobs under `reranker:`:
- `reranker.model_name` (any HF cross-encoder)
- `reranker.rerank_top_k` (depth; cost is O(K))
- `reranker.batch_size`, `reranker.max_length`

### Full-dev BM25-vs-reranked comparison

The block reproduced in *Generation × retrieval source*:

```bash
# Generation on full dev/small (~1 h each on a 6-core CPU; mutually restricted):
python experiments/run_generation_baseline.py \
    --input-run outputs/week02_bm25/run.tsv \
    --output-dir outputs/week03_generation_bm25_full \
    --retrieval-source bm25 \
    --restrict-to-run outputs/week05_reranker_full/run.tsv \
    --num-eval-queries 9999

python experiments/run_generation_baseline.py \
    --input-run outputs/week05_reranker_full/run.tsv \
    --output-dir outputs/week03_generation_reranked_full \
    --retrieval-source reranked \
    --restrict-to-run outputs/week02_bm25/run.tsv \
    --num-eval-queries 9999

# Paired bootstrap + bucket analysis + grounding:
python scripts/bootstrap_generation_comparison.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week03_generation_bootstrap_full
python scripts/analyze_generation_rerank.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week06_analysis
python scripts/bertscore_paired_eval.py --n-pairs 3000
python scripts/regression_failure_taxonomy.py
```

## 4.5. Single-query demo

The pipeline is batch-eval oriented: every entrypoint in `experiments/`
consumes a retrieval run (`run.tsv`) and emits metrics + paired
predictions. There is no `--question "..."` CLI. The blocks below are
the smallest in-Python composition for a single-query sanity check.

### A. Generator-only

```python
from msmarco_genqa.generation.rag_generator import RAGGenerationConfig, RAGGenerator

gen = RAGGenerator(RAGGenerationConfig())
answer = gen.generate(
    query="what is bm25",
    passages=[
        "BM25 is a probabilistic ranking function used by search engines "
        "to estimate the relevance of documents to a given search query.",
        "BM25 was developed in the 1970s and 1980s as part of the Okapi "
        "information retrieval system at City University, London.",
        "Unlike plain TF-IDF, BM25 saturates term frequency and normalises "
        "for document length.",
    ],
)
print(answer)
```

Passages are hand-written here, so the output reflects generator
behaviour, not the end-to-end pipeline.

### B. Retrieve + generate (sketch)

Loading a real index, then mapping returned `doc_ids` back to passage
text. Not a runnable copy-paste — the `doc_id → passage` map is the
missing piece, currently built inside the batch runners:

```python
from msmarco_genqa.retrieval.bm25 import BM25Retriever

bm25 = BM25Retriever.load("data/processed/bm25_index_msmarco")
scores, doc_ids = bm25.retrieve("what is bm25", k=3)
passages = [passage_by_id[d] for d in doc_ids]  # supply your own map
answer = gen.generate(query="what is bm25", passages=passages)
```

A thin `python -m msmarco_genqa.demo.ask "<question>"` wrapper that
bundles the mapping is listed under §8 *Next*.

## 6. Configuration

All knobs live in [`configs/baseline.yaml`](configs/baseline.yaml). Key ones:

| Key | Effect |
|-----|--------|
| `retrieval.k1`, `retrieval.b` | BM25 hyperparameters |
| `retrieval.top_k` | Depth of saved run (default 1000) |
| `retrieval.chunk_size` | Checkpoint cadence in queries (default 200). Smaller = more durable, more I/O. |
| `retrieval.n_threads` | `0` = sequential (matches the 0.1703 baseline). `-1` = all CPUs. |
| `data.corpus_limit` | Set to a small int for a smoke test; does **not** reproduce official numbers. Leave `null` for real runs. |
| `generation.num_eval_queries` | Size of the W3 eval subset |

## 6.5. Reproducibility status

| Area | Status | Notes |
|---|---|---|
| Unit tests | works | `make test` / `pytest -q` — no network, no heavy deps. Slow tests excluded by `pytest.ini_options`. |
| Slow tests | works (skips gracefully) | `make test-slow` includes `@pytest.mark.slow`. HF metric scripts skip if unavailable. |
| Lockfile | basic | `requirements-lock.txt` is pip-freeze-style; sub-dep transitive closure + hash pinning are TODO. |
| Installable package | basic | `pip install -e .` registers `src` via `pyproject.toml`. Existing `sys.path.insert` shims in `experiments/` and `scripts/` are kept for now. |
| CI | basic | `.github/workflows/ci.yml`: pytest + ruff on push/PR to main. No slow tests, no data download. |
| Lint | minimal | `ruff` with `F` + `W` (pyflakes + whitespace). `E` / `I` / `UP` are off on the first pass. |
| Artifact manifest | wired | `src/msmarco_genqa/util/manifest.py` writes `outputs/<stage>/manifest.json` alongside `metrics.json`. Captures git commit + dirty flag, command, config hash, dependency-file hashes, per-output sha256 (truncated). |
| Numbers in `reports/internship_report/report.pdf` | historical | Reflect the dev environment at tag `v1.0-internship-final`. Re-running with `requirements-lock.txt` is the closest we get today. |

**Historical output-path naming.** `outputs/week02_bm25/`,
`outputs/week04_dense/`, `outputs/week05_reranker/` retain
`outputs/weekNN_<topic>/` names even though the rest of the repo
speaks in *stages*. These are the snapshot anchors referenced by the
`provenance.backfill.json` files committed alongside tag
`v1.0-internship-final` (`5a35de9c18ea`); renaming them would
invalidate those provenance records.

Limitations to be aware of:

- The lockfile reflects a macOS CPU-only dev environment. Linux / CUDA may resolve different versions; install `torch` from the appropriate PyTorch index first.
- Corpus, encoder, and reranker checkpoints are downloaded by `ir_datasets` / HuggingFace at first run and are not checksummed by the project.
- `experiments/run_*.py` still rely on `sys.path.insert(0, PROJECT_ROOT)` at the top of the file. `pip install -e .` makes this unnecessary; the shim is kept until a later pass removes it.

## 7. Known limitations

- **Tokenizer mismatch with Anserini.** Our 0.1703 vs reference 0.184 is mostly tokenizer-induced (`bm25s` default tokenizer ≠ Lucene `EnglishAnalyzer`).
- **CPU-only retrieve is slow at 8.8 M docs.** ~70 min for 6 980 queries. `n_threads=-1` may help; not yet benchmarked on this corpus.
- **Generation: pretrained T5-small, no fine-tuning.** Numbers will be low on overlap-based metrics. Fine-tuning is in scope for a later iteration.
- **NumPy 2.x runtime warning.** Some compiled deps (torch) were built against NumPy 1.x. Cosmetic; downgrade to `numpy<2` if it ever causes a real failure.

## 8. Next

- **W5-B — K-sweep Pareto.** K ∈ {50, 100, 200} perf-latency Pareto on
  both first stages (1 000-q subsample for K=50/200; K=100 reuses W5
  full-dev). Queued.
- **W7-B — generator capacity, not decode budget.** The W6 closure
  (`max_new_tokens=64→128`) plus the W7 grounding ceiling (~99 %
  extractiveness on both arms) together imply generator-side work
  should target richer prompt formats (multi-passage synthesis,
  citation-aware decoding) or a different model — not the decode
  budget. Driver
  [`scripts/run_generator_capacity_sweep.py`](scripts/run_generator_capacity_sweep.py)
  runs T5-base on both BM25 and reranked top-3 and re-scores every
  W7 metric, so the open question — does the W7-A NLI sign flip
  (Δ = −0.145) hold at higher capacity? — is one command away.
- **Try a MS-MARCO-tuned dense encoder.** Swap
  `sentence-transformers/all-MiniLM-L6-v2` for
  `sentence-transformers/msmarco-MiniLM-L6-cos-v5` and re-run W4 on
  the same 50 k sample. Measures how much of the current dense-vs-BM25
  gap is attributable to generic-encoder choice vs the retrieval setup.
- **Single-query demo CLI.** A thin
  `python -m msmarco_genqa.demo.ask "<question>"` wrapper around the
  composition shown in §4.5.

## 9. License

See [LICENSE](LICENSE).

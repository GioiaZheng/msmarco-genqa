# Architecture

This repository is built around one research question: how much of a
retrieval-augmented QA gain is caused by better passage selection, and how
much is caused by the generator's own behaviour?

The codebase is intentionally split into three layers:

```text
configs/                   experiment choices and output locations
        |
        v
experiments/ + rag-eval     stage runners and orchestration
        |
        v
src/msmarco_genqa/          reusable retrieval, reranking, generation, and evaluation code
        |
        v
outputs/                   run.tsv, predictions.jsonl, metrics.json, manifests
        |
        v
scripts/ + docs/ + reports/ analysis, audit, written interpretation
```

The separation matters. The runners are allowed to be operational and
command-line friendly; the library modules should stay importable,
deterministic where possible, and easy to unit test without downloading MS
MARCO or model weights.

## Core Boundaries

### Data Loading

`src/msmarco_genqa/data/msmarco.py` owns access to the official MS MARCO
Passage corpus and dev/small query/qrel structures. Downstream code should
not invent local dataset formats unless it writes a clear adapter.

Large data lives under `data/` and is gitignored. The repository records
configuration, code, metrics, and provenance, not the corpus itself.

### Retrieval

`src/msmarco_genqa/retrieval/query_transform.py` owns optional
pre-retrieval query normalization, lexical expansion, and de-contextualization
baselines. The default method is `none`; transformed runs must preserve the
original query text and record the transformation config hash.

Retrieval modules produce TREC-style `run.tsv` files. A retrieval run is the
contract between the ranking layer and every downstream experiment.

Important outputs:

- `outputs/W2_bm25/run.tsv`
- `outputs/W4_dense/run.tsv`
- `outputs/W5_reranker/run.tsv`

The dense stage uses a qrels-anchored sample. Absolute dense metrics should
not be compared against full-corpus BM25; only same-sample comparisons are
valid.

### Reranking

Reranking is order-only over a fixed candidate set. Recall at the rerank
depth should not change by construction. If a future reranker changes
coverage, it should be treated as a new first-stage retrieval condition, not
as the same rerank experiment.

### Generation

Generation consumes a `run.tsv`, retrieves the top-k passages, writes
`predictions.jsonl`, and records both output text and the passages shown to
the model. That persisted prompt context is what makes later grounding audits
possible without re-running generation.

The current generator is frozen T5-small. Current evidence suggests it mostly
extracts from the prompt. Future generator work should therefore treat prompt
format, passage structure, citation behaviour, and model capacity as
experimental variables.

### Evaluation

Evaluation is paired wherever the claim is comparative. For BM25 vs reranked
generation, both arms must cover the same query ids in the same order before
bootstrap confidence intervals are meaningful.

Evaluation scripts read committed code and gitignored outputs. They should
write machine-readable summaries first, then console tables as a convenience.

## Orchestration

There are two orchestration surfaces:

- `mgq-pipeline` reads `configs/pipeline.yaml`, a command-oriented plan.
- `rag-eval run --config configs/baseline.yaml` builds the research
  evaluation plan from the baseline config and the `rag_eval` section.

Use `rag-eval` when the goal is to reproduce or extend the current research
claim. Use individual `mgq-*` commands when iterating on one stage.

Recommended first check:

```bash
rag-eval run --config configs/baseline.yaml --dry-run
```

That command prints the exact stage sequence and expected artifacts without
loading the corpus or model weights.

## Artifact Contract

Every headline-producing stage should leave enough state to answer four
questions later:

1. Which code produced this run?
2. Which config and CLI overrides were used?
3. Which data inputs and upstream run files were consumed?
4. Which metrics and output files were produced?

The manifest contract in `docs/reproducibility_protocol.md` covers this for
stage runners. Analysis scripts should follow the same spirit even when they
do not need the full manifest machinery.

## Extension Rules

Add a new module under `src/` when logic is reused, unit-testable, or part of
the research contract.

Add a new script under `scripts/` when it reads existing artifacts and answers
a bounded analysis question.

Add a new experiment runner under `experiments/` when it produces primary
artifacts that later analyses depend on.

Add a new config key only when it changes a real experimental degree of
freedom. If the key affects a headline number, record it in the resolved
config and mention it in the evaluation protocol.

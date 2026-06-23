# Roadmap

This roadmap keeps long-horizon research and engineering directions visible
without turning every idea into an open issue. Open issues should stay reserved
for work that is specific enough to start, review, or close.

## Active Queue

These are the near-term threads worth keeping as open issues:

- Generator capacity and grounding disambiguation: #4
- MS MARCO-tuned dense encoder comparison: #15
- Artifact registry and lockfile hardening: #18
- Repeatable error-analysis workflow: #19
- End-to-end inference API packaging: #25
- Ablation runners and report tables: #27
- Retrieval and generation profiling: #29
- Static typing and stricter linting: #50
- External versioning for large data and index artifacts: #123
- RAG input hardening and prompt-injection guardrails: #124
- Statistical regression gates for metric deltas: #125
- Experiment-to-report artifact automation: #126

## Research Backlog

Future research directions that should become issues only when the experiment
contract, dataset slice, and acceptance criteria are clear:

- Citation-aware generation and citation-aware decoding.
- NLI and alternative faithfulness metric stress tests.
- Full-sample semantic metrics and judge-based factual consistency protocols.
- Query robustness, unanswerable-query handling, and calibrated abstention.
- Alternative retrieval families such as SPLADE, BGE, ColBERT, and semantic or
  parent-child chunking.
- Adaptive retrieval sufficiency checks and richer reranker error slicing.
- Larger generator comparisons under matched RAG conditions.
- Optional adapter layers for RAGAS, TruLens, or similar metric suites.
- Full-corpus retrieval leaderboard matrices once storage and runtime are
  better managed.

## Serving And Productization

The serving direction should grow from the current package and API surface
rather than becoming a separate application too early:

- Keep the Python API as the canonical first serving surface.
- Add streaming only after request validation, error payloads, and artifact
  lookup are stable.
- Treat vector-store backends, Docker Compose, and a Go gateway as later
  integration work, not immediate project scope.
- A lightweight demo UI is useful after the API path is stable, but it should
  not replace the batch-evaluation focus of the repository.

## Infrastructure Backlog

Infrastructure work should reduce reproduction friction or review time:

- Separate offline indexing from online retrieval once artifact versioning is
  settled.
- Publish generated API documentation after the public package interfaces stop
  moving quickly.
- Extend report generation only where it removes manual copy/paste between
  metrics artifacts, tables, and written reports.

## Issue Hygiene

- Keep open issues actionable and closeable.
- Move broad research directions here until they have a concrete experiment
  contract.
- Prefer one issue per reproducible comparison, runner, report, or validation
  gate.
- Close roadmap-only issues with a short note pointing back to this file.

# rag-observatory Trace Exports

This repository owns the pipeline side of the boundary. `rag-observatory`
owns failure taxonomy, trace validation, and report rendering. The export
format below keeps the two projects separate while allowing a selected
MS MARCO GenQA row to be inspected as a RAG trace.

## Format

The exporter writes one JSON object with:

- `format`: fixed value `msmarco-genqa.trace-export.v1`
- `run`: run id, timestamp, dataset, config hash, code version, stage flags,
  and optional model labels
- `query`: query id, query text, first reference answer, and all references
  under `query.extra.references`
- `retrieved_documents`: ranked candidate passages with optional scores and
  relevance flags when qrels are supplied
- `reranked_documents`: reranked candidates when present in the source row,
  otherwise `null`
- `selected_context`: passages exposed to the answer model
- `prompt`: prompt content when present in the source row, otherwise `null`
- `answer`: model answer plus citation spans when present
- `metrics`: deterministic query-level context relevance, groundedness, answer
  relevance, and related signals
- `failures`: empty by default; failure labeling remains the observability
  layer's responsibility
- `diagnostic_notes`: compact notes about unavailable source fields
- `extra`: source file references and low-score dimensions

The current exporter intentionally works from explicit prediction JSONL rows.
It does not import `rag-observatory`, run retrieval, load models, or move large
outputs into Git.

## Configuration Sweeps

`mgq-export-rag-observatory-sweep` writes a small sweep bundle for comparing
the same query across a few RAG configurations. The bundle contains:

- `rag_observatory_sweep.json`: sweep metadata, stable configuration ids,
  supported/deferred dimensions, and a metric comparison matrix
- `traces/<config_id>/<query_id>.json`: one standard
  `msmarco-genqa.trace-export.v1` file per configuration arm

The sweep manifest is a local comparison index. The trace files remain the
interop boundary consumed by `rag-observatory`, so the observability layer does
not need custom per-sweep parsing.

Example:

```bash
mgq-export-rag-observatory-sweep \
  --arm bm25=outputs/generation_bm25_full/predictions.jsonl \
  --arm dense-rerank=outputs/generation_reranked_full/predictions.jsonl \
  --qrels path/to/qrels.tsv \
  --query-id 12345 \
  --sweep-id retrieval-source-smoke-001 \
  --timestamp 2026-07-01T00:00:00Z \
  --dataset msmarco-passage/dev/small \
  --generator t5-small \
  --output-dir outputs/trace_exports/retrieval-source-smoke-001
```

Current supported dimensions are intentionally narrow:

- stable `config_id`
- retriever label
- reranker presence
- generator label
- selected context depth (`top_k`)

Query rewriting and context compression are recorded as deferred sweep
dimensions until their source rows include a stable, comparable contract.

## Small Reproduction

Run the public-safe fixture export:

```bash
make reproduce-small
```

The command writes:

```text
outputs/reproduce_small/rag_observatory_export.json
outputs/reproduce_small/rag_observatory_sweep/rag_observatory_sweep.json
outputs/reproduce_small/rag_observatory_sweep/traces/bm25/msmarco-synthetic-q001.json
outputs/reproduce_small/rag_observatory_sweep/traces/dense-rerank/msmarco-synthetic-q001.json
```

The fixture uses one synthetic query and small synthetic candidate sets under
`tests/fixtures/rag_observatory_export/`. It is only a schema and interop smoke
test; it is not a benchmark result.

## CLI

Export one query from a prediction file:

```bash
mgq-export-rag-observatory \
  --predictions outputs/generation_bm25_full/predictions.jsonl \
  --qrels path/to/qrels.tsv \
  --query-id 12345 \
  --run-id bm25-full-12345 \
  --dataset msmarco-passage/dev/small \
  --retriever bm25 \
  --generator t5-small \
  --output outputs/trace_exports/bm25-full-12345.json
```

The `rag-observatory` side can ingest the JSON with:

```bash
rag-observe ingest-msmarco-genqa outputs/trace_exports/bm25-full-12345.json \
  --output outputs/traces/bm25-full-12345.trace.json
```

## Limits

- A prediction row must contain matching `top_doc_ids` and `passages`.
- Relevance flags are only populated when qrels are supplied.
- Reranking and prompt fields are exported only when the source row includes
  them.
- Reranked document ids are namespaced as `reranked:<doc_id>` so
  `rag-observatory` can keep retrieved and reranked document records globally
  unique. The original MS MARCO document id is preserved as
  `extra.original_doc_id`.
- No large corpus text, indexes, or model weights are committed by this path.

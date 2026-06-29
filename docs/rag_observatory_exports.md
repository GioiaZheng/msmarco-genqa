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

## Small Reproduction

Run the public-safe fixture export:

```bash
make reproduce-small
```

The command writes:

```text
outputs/reproduce_small/rag_observatory_export.json
```

The fixture uses one synthetic query and a two-document qrels file under
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
- No large corpus text, indexes, or model weights are committed by this path.


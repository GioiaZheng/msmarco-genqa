# Input Validation Contract

The research pipeline assumes MS MARCO-style data, but every boundary that
accepts external artifacts should fail early on malformed input. The shared
helpers in `msmarco_genqa.util.input_validation` define that behavior.

## Text Fields

Query and passage text is normalized by collapsing whitespace and replacing
Unicode replacement characters with spaces. If replacement characters dominate
the field, the input is treated as corrupted and raises `InputValidationError`.

Generation prompt assembly uses deterministic character budgets before model
tokenization:

- query text: normalized and truncated at a word boundary;
- passage text: normalized, empty optional passages dropped, and long passages
  truncated at a word boundary;
- empty queries are rejected.

The tokenizer still enforces `max_input_length`; the character limits are an
auditable pre-tokenization guard for runners and serving.

## Run Files

TREC-style `run.tsv` files must have six tab-separated fields:

```text
qid    Q0    doc_id    rank    score    system
```

The parser rejects empty query ids, empty document ids, empty system names,
non-positive or non-integer ranks, non-finite scores, duplicate ranks,
duplicate document ids within a query, replacement characters in identifiers,
and non-UTF-8 bytes. Entries are returned in ascending rank order.

Batch runners use this strict parser instead of skipping malformed lines, so a
bad run file fails before expensive generation or analysis starts.

## JSONL Records

Serving passage JSONL files are UTF-8 JSONL with one object per line. Blank
lines are ignored. Records must include non-empty `id` and `text` fields, and
passage ids must be unique. Invalid JSON, non-object records, duplicate ids,
empty fields, and replacement-character-heavy text raise `InputValidationError`
with file and line context when available.

## Serving Errors

The optional FastAPI wrapper maps `InputValidationError` to a structured 422
payload:

```json
{
  "error": {
    "type": "input_validation",
    "message": "must not be empty",
    "field": "query"
  }
}
```

This keeps production-facing errors explicit while preserving the research
pipeline's deterministic batch behavior.

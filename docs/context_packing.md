# Context Packing and Prompt Compression

Context packing sits between retrieval/reranking and generation. It decides
which passage text is allowed into the final prompt while preserving the source
document mapping needed by grounding and triad reports.

## Packing Strategy

The default historical generation baseline is unchanged: `context_packing.enabled`
is `false` in `configs/baseline.yaml`.

When enabled, the runner applies a deterministic CPU-friendly packer:

- `max_context_chars`: total context budget, used as a stable token-cost proxy.
- `max_passage_chars`: per-passage cap before a passage enters the prompt.
- `sentence_selection`: `query_overlap` ranks sentences by query-token overlap;
  `head` keeps leading text.
- `deduplicate`: drops repeated normalized passage text.
- `ordering`: `rank` preserves retrieval order; `shorter_first` is available for
  budget stress tests.

The packer writes per-query metadata into `predictions.jsonl` under
`context_packing`. Each row records the original document ids, retained document
ids, dropped ids, character counts, compression ratio, and span offsets back to
the retained source document ids.

## Reproduction

Run the regular reranked generation baseline first, then run the packed variant
into a separate output directory:

```bash
mgq-generate \
  --config configs/baseline.yaml \
  --input-run outputs/W5_reranker_full/run.tsv \
  --output-dir outputs/W3_generation_reranked_packed \
  --retrieval-source reranked_packed \
  --restrict-to-run outputs/W2_bm25/run.tsv \
  --num-eval-queries 9999 \
  --context-packing \
  --context-max-chars 900 \
  --context-max-passage-chars 320 \
  --context-sentence-selection query_overlap \
  --context-ordering rank
```

Compare the compressed run against the uncompressed reranked predictions:

```bash
mgq-context-packing-report \
  --baseline-predictions outputs/W3_generation_reranked_full/predictions.jsonl \
  --compressed-predictions outputs/W3_generation_reranked_packed/predictions.jsonl \
  --baseline-name reranked \
  --compressed-name reranked_packed \
  --output-dir outputs/W9_context_packing
```

Artifacts:

- `comparison.json`: aggregate matched-qid metrics and context-cost deltas.
- `per_query.jsonl`: query-level predictions, references, context sizes, and
  per-query metric deltas.
- `report.md`: compact Markdown tables for experiment notes.

`rag-eval run --config configs/baseline.yaml --dry-run` also prints the packed
generation and comparison stages.

## Limitations

- Character count is a deterministic proxy for tokenizer cost. It is suitable
  for CI and CPU-only smoke tests, but model-specific tokenizer counts should be
  added before claiming exact serving cost.
- Query-overlap sentence selection is lexical. It is intentionally transparent,
  but it will miss paraphrased evidence that does not share surface tokens with
  the query.
- Compression can improve cost without improving answer quality. Interpret
  `context_packing_report` together with grounding and RAG triad outputs.
- Span offsets are relative to the packed context string, not to the original
  MS MARCO passage text.

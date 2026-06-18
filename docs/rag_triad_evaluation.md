# RAG Triad Evaluation

`mgq-rag-triad` builds a deterministic triad report over one or more
generation `predictions.jsonl` files. It is designed as a regression and
diagnostic layer rather than a leaderboard metric.

## Dimensions

| Dimension | Deterministic baseline | Output field |
|---|---|---|
| Context relevance | Qrels hit in the shown context when `--qrels` is provided; otherwise lexical query-context overlap. | `scores.context_relevance` |
| Groundedness | Fraction of generated content tokens supported by the shown passages. | `scores.groundedness` |
| Answer relevance | Token-F1 against the reference answers stored in `predictions.jsonl`. | `scores.answer_relevance` |

The aggregate `scores.triad` is the mean of the three dimensions. The JSONL
keeps the individual scores so readers can distinguish retrieval failures,
unsupported generations, and answer-mismatch failures.

## Reproduce

```bash
mgq-rag-triad \
  --predictions bm25=outputs/generation_bm25_full/predictions.jsonl \
  --predictions reranked=outputs/generation_reranked_full/predictions.jsonl \
  --baseline-config bm25 \
  --output-dir outputs/rag_triad
```

If a local qrels TSV is available, pass it to make context relevance
answer-evidence based:

```bash
mgq-rag-triad \
  --predictions bm25=outputs/generation_bm25_full/predictions.jsonl \
  --predictions reranked=outputs/generation_reranked_full/predictions.jsonl \
  --qrels data/qrels.dev.small.tsv \
  --baseline-config bm25 \
  --context-top-k 3 \
  --output-dir outputs/rag_triad
```

Outputs:

- `metrics.json`: aggregate means, low-score counts, and settings.
- `per_query_triad.jsonl`: one row per query/config pair with query metadata,
  retrieval candidates, generation text, references, scores, flags, and
  movement diagnostics when qrels are available.
- `low_score_cases.jsonl`: compact inspection rows for low triad or low
  dimension scores.
- `report.md`: a short Markdown summary for experiment notes.

## Limitations

The deterministic evaluator is intentionally conservative:

- Qrels-based context relevance is binary at the shown-context level. It says
  whether a judged relevant passage appeared, not whether the passage is enough
  for faithful synthesis.
- The qrels-free fallback is lexical query-context overlap. It is useful in CI
  and smoke runs, but it should not be used as a headline evidence metric.
- Lexical groundedness misses faithful paraphrases and can over-reward copied
  fragments.
- Token-F1 answer relevance follows the MS MARCO short-answer convention, but
  it is not a semantic adequacy judge.

Model-assisted RAGAS/TruLens-style scoring can be layered behind an explicit
evaluator later. The current CLI rejects unsupported evaluator names so runs do
not silently mix deterministic and model-graded numbers.

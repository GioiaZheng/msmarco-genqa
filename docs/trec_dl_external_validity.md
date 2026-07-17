# TREC-DL 2019 / 2020 external-validity coverage

The primary evaluation split for this project is MS MARCO Passage
`dev/small` (6,980 queries). Its qrels are **sparse and binary**: roughly
one relevant passage is marked per query, and everything else is treated as
non-relevant whether or not it was ever judged. That is fine for a stable
leaderboard signal but it is a single dataset with shallow judgments.

The TREC-DL 2019 and 2020 passage tracks reuse the *same* MS MARCO passage
collection but ship **deep, graded** relevance judgments produced by NIST
assessors over a pooled, small query set. This gives a second,
independently-judged dataset on which the existing retrieval and grounding
metrics can be cited.

## What the loader provides

`msmarco_genqa.data.trec_dl` exposes:

- `load_trec_dl(year, cache_dir=None, rel_threshold=2)` — load a track via
  `ir_datasets` (the `msmarco-passage/trec-dl-20XX/judged` subsets).
- `load_trec_dl_from_files(queries_path, qrels_path, *, year, ...)` — load
  from local query/qrels files, no network (used by the fixture tests and
  for offline runs).
- `TrecDlPassages` — the returned bundle.

The bundle's `queries` (`{qid: text}`) and `qrels` (`{qid: set[doc_id]}`)
use the **same schema as the MS MARCO loader**, so the existing
`evaluate_retrieval` / `mrr@k` / `ndcg@k` / `recall@k` functions in
`msmarco_genqa.evaluation.retrieval` run on them unchanged. The full graded
labels are additionally preserved on `graded_qrels`
(`{qid: {doc_id: label}}`) for the graded evaluation work tracked in #153.

## Full-corpus runner commands

Both tracks are first-class options on the existing BM25 and cross-encoder
runners. They reuse the same full MS MARCO passage index; only the selected
queries and qrels change.

```bash
# TREC-DL 2019
mgq-retrieve --dataset msmarco-passage/trec-dl-2019/judged --resume
mgq-rerank --dataset msmarco-passage/trec-dl-2019/judged --resume

# TREC-DL 2020
mgq-retrieve --dataset msmarco-passage/trec-dl-2020/judged --resume
mgq-rerank --dataset msmarco-passage/trec-dl-2020/judged --resume
```

Default outputs are isolated by track:

- `outputs/trec_dl_2019/bm25` and
  `outputs/trec_dl_2019/cross_encoder_rerank`
- `outputs/trec_dl_2020/bm25` and
  `outputs/trec_dl_2020/cross_encoder_rerank`

Each `metrics.json` and manifest records the dataset id, track year, judged
topic count, corpus scope, and upstream run. The existing dev/small defaults
remain unchanged when `--dataset` is omitted.

## Judgment depth and binarization

| Dataset | Judgments | Label scale | Judged queries |
|---|---|---|---|
| MS MARCO `dev/small` | sparse | binary (0/1) | 6,980 (~1 positive each) |
| TREC-DL 2019 passage | deep, pooled | graded 0–3 | 43 |
| TREC-DL 2020 passage | deep, pooled | graded 0–3 | 54 |

The TREC-DL label scale is 0 (irrelevant) … 3 (perfectly relevant). The
binary metrics (MRR, Recall) need a positive set, so labels are binarized at
`rel_threshold = 2` (i.e. labels **2 and 3 are relevant**, 0 and 1 are not).
This is the standard convention for the MS MARCO TREC-DL passage tracks. The
threshold is a parameter; lower it to 1 to include the "related" tier.

A judged query whose passages are all below threshold is kept with an empty
positive set, so coverage diagnostics can tell "no relevant passage above
threshold" apart from "query never judged" — mirroring how the MS MARCO
evaluation path skips empty-qrels queries.

## Determinism and seeds

The loaders are pure functions of the dataset files: they iterate the
`ir_datasets` queries/qrels and bucket them into dicts. There is no
sampling, shuffling, or randomness, so no seed is required — the bundle is
byte-for-byte reproducible given a fixed `ir_datasets` cache. The judged
query counts above (43 / 54) are the full evaluable sets, not a subsample.

## Tests and fixtures

`tests/test_trec_dl.py` is fully offline. It parses miniature, synthetic
fixtures under `tests/fixtures/trec_dl/` (a handful of queries with graded
0–3 qrels per track, plus one run file) and asserts that:

- binarization at `rel>=2` drops the 0/1-labelled passages,
- the bundle schema matches what the metric code consumes, and
- `evaluate_retrieval` computes end-to-end on the binarized qrels.

The fixtures are intentionally small and not the real NIST topics, so the
suite never depends on a multi-gigabyte download.

## Scope note

The runner integration deliberately keeps the BM25 tokenizer, full-corpus
index, cross-encoder weights, candidate depth, and prompt/generation code
unchanged. At this stage, the runners still report the existing binary metric
view at `rel_threshold = 2`; reportable graded nDCG and the independent
evaluator cross-check are tracked separately in #153. Dense retrieval and
multi-encoder calibration are not part of this benchmark step.

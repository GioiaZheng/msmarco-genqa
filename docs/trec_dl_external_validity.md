# TREC-DL 2019 / 2020 external-validity benchmark

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

## Full-corpus results

Both tracks were run on 18 July 2026 from commit
`0362a48bb4d000a93f5af5c26a4de934db37e74f`. The first stage is `bm25s`
0.3.9 over all 8,841,823 MS MARCO passages (`k1=1.5`, `b=0.75`, English
stopwords, top-1,000). The second stage uses
`cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`, reranking the fixed BM25
top-100 with batch size 64 and maximum length 512.

### TREC-DL 2019 passage

| Metric | BM25 | BM25 + CE | Delta |
|---|---:|---:|---:|
| MRR@10, rel >= 2 | 0.5471 | 0.8787 | **+0.3315** |
| nDCG@10, graded | 0.4239 | 0.7210 | **+0.2971** |
| Recall@100, rel >= 2 | 0.4469 | 0.4469 | 0.0000 |
| Recall@1000, rel >= 2 | 0.6983 | n/a (top-100 output) | n/a |

Scope: all 43 judged topics, 9,260 qrels rows, and 43/43 run-topic
coverage. `ir-measures` 0.4.3 reproduced every metric within
`2.22e-16`, below the `1e-12` acceptance tolerance.

### TREC-DL 2020 passage

| Metric | BM25 | BM25 + CE | Delta |
|---|---:|---:|---:|
| MRR@10, rel >= 2 | 0.6280 | 0.8256 | **+0.1976** |
| nDCG@10, graded | 0.4773 | 0.6801 | **+0.2027** |
| Recall@100, rel >= 2 | 0.5105 | 0.5105 | 0.0000 |
| Recall@1000, rel >= 2 | 0.7521 | n/a (top-100 output) | n/a |

Scope: all 54 judged topics, 11,386 qrels rows, and 54/54 run-topic
coverage. The independent cross-check again passed with maximum absolute
delta `2.22e-16`.

The CE run files contain 100 documents per topic. Their computed
`Recall@1000` therefore equals `Recall@100`; it is an output-depth property,
not evidence that reranking reduced BM25 top-1,000 recall. Only MRR@10,
nDCG@10, and Recall@100 are paired BM25-vs-CE comparisons.

## Runtime and resource note

The cold 2019 run loaded and indexed the full collection in 602.3 seconds;
the index calculation itself reported 302.7 seconds, followed by index
serialization. Search took 18.5 seconds for 43 topics. Reusing the same index,
2020 search took 20.9 seconds for 54 topics.

CPU reranking took 98.9 seconds for 4,300 pairs in 2019 (43 pairs/s) and
91.8 seconds for 5,400 pairs in 2020 (59 pairs/s). The 2019 timing includes
the first model load and warm-up, so this one-run comparison is not a stable
throughput benchmark. During the cold index build on the 16 GiB Windows host,
the observed process peak was about 8.1 GiB working set and 18.3 GiB private
allocation, so pagefile headroom mattered. Cached-index runs stayed near
3.1 GiB working set.

## Query-level lift and error review

The review uses the same threshold-2 qrels and deterministic top examples
from `scripts/analyze_retrieval_lift.py`; seed 42 is recorded in both run
manifests. No query is sampled out.

| Track | Promoted | New hit@10 | Demoted | Lost hit@10 | Unchanged hit | Unchanged miss |
|---|---:|---:|---:|---:|---:|---:|
| 2019 | 21 | 2 | 1 | 0 | 18 | 1 |
| 2020 | 18 | 3 | 3 | 0 | 26 | 4 |

The improvement is primarily local ordering. In 2019, topic `1115776`
(`what is an aml surveillance analyst`) moves its first threshold-relevant
passage from rank 8 to rank 1, and `1063750` (US entry into WWI) becomes a
new hit at rank 1. In 2020, `1051399` (`who sings monk theme song`) moves
from rank 9 to rank 1. The unchanged-miss topics show the limit of an
order-only reranker: if BM25 does not put a threshold-relevant passage in its
top-100, CE cannot recover it.

The few demotions are not all clean model failures. For 2019 topic `490595`
(`rsa definition key`), CE puts a broad RSA definition labelled 1 above an
RSA-algorithm passage labelled 2, moving the first threshold hit from rank 1
to 3. For 2020 topic `405163` (`is caffeine an narcotic`), CE ranks an
explicit "No" answer first, but that passage is labelled 0 while a passage
about a caffeine/codeine combination drug is labelled 3. Topic `1064670`
similarly puts a fuller shotgun-patterning explanation labelled 1 above a
short multiple-choice answer labelled 2. These cases justify retaining
graded scores and inspecting judgment semantics instead of treating every
threshold demotion as an obvious relevance error.

The checked source for the tables, exact unrounded metrics, runtime fields,
lift counts, commit/config identifiers, and SHA-256 digests of every local
run, manifest, cross-check, and lift artifact is
[`reports/generated/artifacts/trec_dl_bm25_ce.json`](../reports/generated/artifacts/trec_dl_bm25_ce.json).

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
(`{qid: {doc_id: label}}`). TREC-DL runners pass that mapping to
`evaluate_trec_retrieval`, which keeps graded nDCG separate from thresholded
MRR and recall.

## Full-corpus runner commands

Both tracks are first-class options on the existing BM25 and cross-encoder
runners. They reuse the same full MS MARCO passage index; only the selected
queries and qrels change.

```bash
# TREC-DL 2019
mgq-retrieve --dataset msmarco-passage/trec-dl-2019/judged \
  --output-dir outputs/trec_dl_2019/bm25 --require-clean-tree
mgq-rerank --dataset msmarco-passage/trec-dl-2019/judged \
  --input-run outputs/trec_dl_2019/bm25/run.tsv \
  --output-dir outputs/trec_dl_2019/cross_encoder_rerank \
  --rerank-chunk-size 10 --resume --require-clean-tree

# TREC-DL 2020
mgq-retrieve --dataset msmarco-passage/trec-dl-2020/judged \
  --output-dir outputs/trec_dl_2020/bm25 --require-clean-tree
mgq-rerank --dataset msmarco-passage/trec-dl-2020/judged \
  --input-run outputs/trec_dl_2020/bm25/run.tsv \
  --output-dir outputs/trec_dl_2020/cross_encoder_rerank \
  --rerank-chunk-size 10 --resume --require-clean-tree
```

On Windows, run Python in UTF-8 mode (`PYTHONUTF8=1`) when `ir_datasets`
reads the collection. POSIX environments already default to UTF-8 in the
supported setup.

Default outputs are isolated by track:

- `outputs/trec_dl_2019/bm25` and
  `outputs/trec_dl_2019/cross_encoder_rerank`
- `outputs/trec_dl_2020/bm25` and
  `outputs/trec_dl_2020/cross_encoder_rerank`

Each `metrics.json` and manifest records the dataset id, track year, judged
topic count, corpus scope, and upstream run. The existing dev/small defaults
remain unchanged when `--dataset` is omitted.

## Independent metric cross-check

Materialize each track's official qrels and cross-check its run separately.
The threshold affects MRR and recall only; nDCG retains the original labels.

```bash
ir_datasets export msmarco-passage/trec-dl-2019/judged qrels --format trec \
  > data/processed/trec-dl-2019-passage.qrels
ir_datasets export msmarco-passage/trec-dl-2020/judged qrels --format trec \
  > data/processed/trec-dl-2020-passage.qrels

mgq-trec-eval --backend ir-measures --qrels-format trec --rel-threshold 2 \
  --qrels data/processed/trec-dl-2019-passage.qrels \
  --run outputs/trec_dl_2019/bm25/run.tsv \
  --output-dir outputs/trec_dl_2019/bm25/trec_eval

mgq-trec-eval --backend ir-measures --qrels-format trec --rel-threshold 2 \
  --qrels data/processed/trec-dl-2020-passage.qrels \
  --run outputs/trec_dl_2020/bm25/run.tsv \
  --output-dir outputs/trec_dl_2020/bm25/trec_eval
```

Repeat the same command with each track's
`cross_encoder_rerank/run.tsv`. Keep the 2019 and 2020 reports separate; do
not average them into one headline without also reporting both track values.

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
positive set. It contributes zero to thresholded MRR and recall, while its
original labels still contribute to graded nDCG. All qrels topics remain in
the TREC-DL denominator, so coverage diagnostics distinguish "no relevant
passage above threshold", "missing from the run", and "never judged".

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

`tests/test_trec_cross_check.py` adds a hand-verifiable graded ranking with a
missing topic, tied source scores, and below-threshold judgments. CI compares
nDCG@10, MRR@10, Recall@100, and Recall@1000 against `ir-measures`.

The fixtures are intentionally small and not the real NIST topics, so the
suite never depends on a multi-gigabyte download.

## Scope note

The runner integration deliberately keeps the BM25 tokenizer, full-corpus
index, cross-encoder weights, candidate depth, and prompt/generation code
unchanged. Runner metrics use graded nDCG plus the documented threshold-2
binary view; reportable artifacts are independently cross-checked with
`mgq-trec-eval`. Dense retrieval and multi-encoder calibration are not part of
this benchmark step.

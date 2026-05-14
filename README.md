# MS MARCO GenQA

Retrieval-augmented question answering on MS MARCO. The repo has two parallel tracks:

- **Script pipeline** (`experiments/`, `src/`) — reproducible, official-corpus, structured outputs + auto-generated reports. **This is the source of truth for benchmark numbers.**
- **Notebooks** (`notebooks/`) — small prototype experiments on hand-written toy corpora or sampled data. They exist for narrative + visualization + smoke-checking the API. **Numbers in notebooks are illustrative only — do not cite them as benchmarks.** Every notebook has a "Limitations" section that says so explicitly, and points at the equivalent script for the honest result.

## 1. Status

### Week 1 — EDA &nbsp;&nbsp;✅ done (notebook only)

- Notebook: [`notebooks/week01_eda.ipynb`](notebooks/week01_eda.ipynb) — runs end-to-end
- Report: `python -m src.reporting.build_report --week week01` →
  [`reports/generated/week01_eda.md`](reports/generated/) *(gitignored)*

### Week 2 — BM25 retrieval &nbsp;&nbsp;✅ done

- Script: [`experiments/run_retrieval.py`](experiments/run_retrieval.py)
- **MRR@10 = 0.1703** &nbsp;·&nbsp; **Recall@100 = 0.6212** &nbsp;·&nbsp; **Recall@1000 = 0.8154**
  on `dev/small` (6,980 queries, full 8.8M-passage corpus)
- Prototype notebook: [`notebooks/week02_retrieval.ipynb`](notebooks/week02_retrieval.ipynb)
  — sampled closed-set (MRR@10 = 0.1956, n=30, structurally optimistic; not a benchmark)

### Week 3 — RAG generation &nbsp;&nbsp;✅ done

- Script: [`experiments/run_generation_baseline.py`](experiments/run_generation_baseline.py)
- **Sampled baseline** — 200-query sample of dev/small (seed 42), T5-small (no fine-tuning),
  top-3 BM25 passages from the W2 run, best-of-N reference scoring:
  - **ROUGE-L = 0.1626** &nbsp;·&nbsp; **BLEU = 0.0574** &nbsp;·&nbsp; **EM = 0.0050** &nbsp;·&nbsp; **Token-F1 = 0.1756**
  - Run config: [`configs/baseline.yaml`](configs/baseline.yaml) · Command: `python experiments/run_generation_baseline.py` ·
    Manifest: `outputs/week03_generation/manifest.json` *(gitignored; regenerated each run)*
  - **Not** a full dev/small benchmark — 200 / 6,980 queries, CPU-friendly.
- Prototype notebook: [`notebooks/week03_generation.ipynb`](notebooks/week03_generation.ipynb)
  — 3-passage toy demo with T5-small (smoke test, not a benchmark)
- Historical note: a re-run on 2026-05-13 produced essentially the same numbers
  (ROUGE-L 0.1626 vs prior 0.1619, BLEU 0.0574 vs prior 0.0573). The earlier
  numbers therefore do not need to be retracted — they were produced with
  effectively the same scoring as today, and the `R2` best-of-N fix had only
  marginal impact on T5-small at this sample size.

### Week 4 — Dense retrieval (sampled) &nbsp;&nbsp;✅ done

- Script: [`experiments/run_dense_retrieval.py`](experiments/run_dense_retrieval.py)
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, FAISS `IndexFlatIP` over
  L2-normalised embeddings, qrels-anchored 50k-passage sample.
- **Dense MRR@10 = 0.8830** &nbsp;·&nbsp; **nDCG@10 = 0.9041** &nbsp;·&nbsp; **Recall@100 = 0.9946**
  vs **BM25-on-sample MRR@10 = 0.6948** &nbsp;·&nbsp; **Recall@100 = 0.9338** (same 50k pool).
- Numbers are **upper-bounded** by qrels-anchoring (every dev relevant doc is
  in the pool by construction). The valid comparison is *dense vs BM25 on the
  same sample*, not against the W2 full-corpus number.

### Week 5 — Cross-encoder reranking &nbsp;&nbsp;✅ done

- Script: [`experiments/run_reranker.py`](experiments/run_reranker.py)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over the W4 dense top-100.
- **Full dev/small (6 980 queries)** — `outputs/week05_reranker_full/` (gitignored):

  | Metric      | Dense (W4) | + CE rerank | Δ          |
  |-------------|-----------:|------------:|-----------:|
  | MRR@10      | 0.8830     | 0.9304      | **+0.0474** |
  | nDCG@10     | 0.9041     | 0.9434      | +0.0393    |
  | Recall@100  | 0.9946     | 0.9946      | +0.0000    |

  Runtime ~4 h 37 min on a 6-core MacBook (538 000 (query, passage) pairs
  at ~32 pairs/s, peak RSS ~3.3 GiB). Recall@100 is unchanged by
  construction — reranking is order-only.
- **Sanity check vs the 1 000-query CPU subsample run.** An earlier
  pilot on a 1 000-query subsample of dev/small produced
  MRR Δ +0.0435, nDCG Δ +0.0398 (level: 0.8846 → 0.9282 / 0.9014 → 0.9412).
  The full-dev deltas (+0.0474 / +0.0393) sit on top of the subsample
  deltas with a similar magnitude, so the subsample was not biased — the
  reranker gain is a property of dev/small, not of the 1 000-query slice.
- Prototype notebook: [`notebooks/week05_reranker.ipynb`](notebooks/week05_reranker.ipynb)
  — 8-passage toy demo showing the score-margin sharpening (bi-encoder gap
  ~0.08 → cross-encoder gap ~7).
- Narrative: W4 closed the recall gap (semantic matching). W5 closes the
  *local-ordering* gap — once the relevant passage is somewhere in the top-100,
  the cross-encoder is what pushes it to top-1 / top-3.

### Generation × retrieval source &nbsp;&nbsp;✅ done

Does feeding reranked top-K back into the T5-small generator actually
improve answer quality? Apples-to-apples comparison on **full dev/small
(6 980 queries)**: same T5-small (no fine-tuning), same top-3 passages
— only the upstream retrieval source changes. Both runs are mutually
restricted via `--restrict-to-run` so the two predictions cover the
**exact same 6 980 query ids** (verified by qid set-diff = ∅).

| Retrieval source &rarr; T5-small | ROUGE-L | BLEU | EM | Token-F1 |
|---|---:|---:|---:|---:|
| BM25 &nbsp; *(`outputs/week02_bm25/run.tsv`)*                              | 0.1859 | 0.0717 | 0.0135 | 0.1966 |
| Reranked &nbsp; *(`outputs/week05_reranker_full/run.tsv`)*                 | **0.3621** | **0.2922** | **0.0606** | **0.3677** |
| **Δ (rerank − BM25)** | **+0.1763** | **+0.2206** | **+0.0471** | **+0.1711** |

**Reranking the first stage roughly doubles every generation metric on
full dev/small, and the 95% paired-bootstrap CI on the per-query Δ
excludes 0 for all four metrics** (see the bootstrap table below).
Cross-encoder reordering of top-100 → top-3 delivers materially better
passages into the generator's context window, and surface-form metrics
pick that up even with a frozen pretrained T5-small. The structural
mechanism is visible in retrieval flags: the rate of having at least
one relevance-judged passage in the top-3 jumps from **20.8 %** (BM25)
to **96.9 %** (reranked) over the 6 980 paired queries.

Per-query token-F1 splits 4 015 strict improvements / 1 766
regressions / 1 199 ties — i.e. **net +2 249 / 6 980 queries (32 %)**
strictly improved by feeding reranked passages, with the regressions
mostly clustered around already-easy queries where BM25 happened to
surface a usable passage.

Qualitative example (qid 1043064, *"what is the chemical formula for
oxygen tetrafluoride?"*): BM25 → *"Li 2 O"*; reranked → *"N2F4"*
(matches the reference).

#### Paired-bootstrap 95% CI on Δ (rerank − BM25)

Full 6 980 paired qids, 10 000 bootstrap resamples, seed 42.
ROUGE-L and BLEU per-query scores here come from `rouge_score.RougeScorer`
and NLTK `sentence_bleu` (smoothing method 1) so the per-query means
differ slightly from the HF corpus-level numbers above; what matters is
that all four CIs lie strictly above 0 with effectively zero overlap.

| Metric            | BM25 (per-query mean) | Reranked | Δ        | 95% CI on Δ           | p₂ (10k resamples) |
|-------------------|----------------------:|---------:|---------:|----------------------:|-------------------:|
| ROUGE-L           | 0.1934                | 0.3677   | +0.1742  | [+0.1663, +0.1820]    | < 0.001            |
| BLEU (sentence)   | 0.0423                | 0.1754   | +0.1330  | [+0.1265, +0.1395]    | < 0.001            |
| Exact-Match       | 0.0135                | 0.0606   | +0.0471  | [+0.0417, +0.0527]    | < 0.001            |
| Token-F1          | 0.1966                | 0.3677   | +0.1711  | [+0.1632, +0.1789]    | < 0.001            |

CIs are ~6× tighter than the earlier 200-query subsample (n=200
vs. n=6 980), and the full-dev Δ point estimates all sit inside the
200-query CIs — the original direction-and-magnitude claim survives
the move to full dev/small intact, with the level numbers naturally
deflating to the harder full benchmark.

#### Bucket analysis by query type (token-F1)

| query_type     |    n |   BM25 |  Reranked |        Δ |
|----------------|-----:|-------:|----------:|---------:|
| DESCRIPTION    | 3725 | 0.1889 |    0.3939 | **+0.2050** |
| ENTITY         |  631 | 0.1765 |    0.3186 | +0.1421  |
| LOCATION       |  498 | 0.2495 |    0.3928 | +0.1433  |
| NUMERIC        | 1665 | 0.1997 |    0.3235 | +0.1238  |
| PERSON         |  461 | 0.2186 |    0.3557 | +0.1371  |

`DESCRIPTION` queries (53 % of the eval set) benefit most from
reranking; `NUMERIC` benefits least — consistent with the intuition
that short numeric answers depend more on lexical surface match in
the passage than on which-of-the-near-duplicates the reranker picks.
Full bucket counts (`rerank_fixed_generation_improved` = 2 184,
`rerank_fixed_generation_still_failing` = 2 684, `regression` = 233,
…) in [`outputs/week06_analysis/summary.json`](outputs/week06_analysis/) (gitignored).

Reproduce:

```bash
# 1. Generation on full dev/small (~1 h each on a 6-core CPU; mutually
# restricted to the same qid set):
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

# 2. Four-metric paired bootstrap CI:
python scripts/bootstrap_generation_comparison.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week03_generation_bootstrap_full

# 3. Bucket + retrieval-flag analysis (with token-F1 / EM CIs inline):
python scripts/analyze_generation_rerank.py \
    --bm25-dir outputs/week03_generation_bm25_full \
    --reranked-dir outputs/week03_generation_reranked_full \
    --output-dir outputs/week06_analysis
```

The earlier 200-query subsample comparison (BM25 0.2131 / Rerank
0.4006 ROUGE-L, Δ +0.1875 with CI [+0.1344, +0.2389]) lives in
`outputs/week03_generation_{bm25,reranked}/` (gitignored). The Δ point
estimates and conclusions are consistent across the two scales; the
full-dev numbers above are the version to cite.

### Reference points

- Published Anserini/Lucene BM25 baseline on MS MARCO `dev/small`: MRR@10 ≈ 0.184. Our `bm25s`-based **0.1703** is in the same ballpark; the gap is consistent with tokenizer differences.

## 2. Directory layout

```
configs/        baseline.yaml — paths + retrieval/generation/reranker/eval knobs
experiments/    run_retrieval.py, run_dense_retrieval.py,
                run_generation_baseline.py, run_reranker.py
src/
  data/         msmarco.py — ir_datasets loader for the official corpus
  retrieval/    bm25.py    — bm25s wrapper with save/load + chunked retrieve
                dense.py   — Sentence-Transformers + FAISS dense retriever
                sampling.py — qrels-anchored sub-corpus sampling
  reranking/    cross_encoder.py — Cross-encoder reranker wrapper (W5)
                io.py            — TREC run.tsv read/truncate/write helpers
  generation/   rag_generator.py — T5/BART RAG generator
  evaluation/   retrieval.py (MRR/Recall/nDCG), generation.py (ROUGE/BLEU/EM/F1)
  reporting/    build_report.py — fills markdown templates from outputs/
  bm25_retriever.py — legacy rank_bm25 wrapper, used only by week03 notebook
notebooks/      prototype, narrative, plots
reports/
  templates/    week02_bm25.md, week03_generation.md  (committed)
  generated/    filled-in markdown + optional PDF     (gitignored)
outputs/        run.tsv, metrics.json, examples.jsonl per week  (gitignored)
data/           raw/, processed/, cache/ — all gitignored, .gitkeep tracked
figures/        plots from notebooks (committed)
scripts/        smoke tests + the notebook regenerator
```

Everything runs from the project root. Scripts add `PROJECT_ROOT` to `sys.path` themselves; no `PYTHONPATH` needed.

## 3. Setup

Python 3.10+ recommended (3.9 also works).

```bash
# Fast development install — loose lower bounds, latest compatible versions.
pip install -r requirements.txt
pip install -e .                       # register `src` as a real package
```

For an environment that reproduces the numbers checked into the reports,
pin to the lockfile instead:

```bash
pip install -r requirements-lock.txt   # exact versions, no upgrades
pip install -e .
```

Or, equivalently:

```bash
make install                           # uses requirements.txt + editable install
```

Optional, only needed for PDF report generation (markdown reports work without):

```bash
brew install pandoc                    # macOS
brew install --cask basictex           # macOS LaTeX engine
# Linux: sudo apt-get install pandoc texlive-xetex
```

## 4. Run the official baselines

### Week 2 — BM25 retrieval

```bash
python experiments/run_retrieval.py
python -m src.reporting.build_report --week week02
```

First run: ~5 min download (~1 GB), ~15 min `ir_datasets` encoding fix pass, ~10 min `bm25s` index build, ~70 min retrieve. Total ≈ 1h40m on a 16 GB MacBook.

Subsequent runs reuse the cached index (`data/processed/bm25_index_msmarco/`, 2.1 GB) and skip download/index — only retrieve runs (~70 min).

If a run is killed mid-retrieve:

```bash
python experiments/run_retrieval.py --resume     # picks up at next chunk boundary
```

To force a fresh index:

```bash
python experiments/run_retrieval.py --rebuild-index
```

Outputs:
- `outputs/week02_bm25/metrics.json`
- `outputs/week02_bm25/run.tsv` (TREC-format top-1000, ~250 MB)
- `outputs/week02_bm25/examples.jsonl`
- `reports/generated/week02_bm25.md` (+ `.pdf` if pandoc installed)

### Week 3 — RAG generation baseline

Requires Week 2 to have produced `outputs/week02_bm25/run.tsv`.

```bash
python experiments/run_generation_baseline.py
python -m src.reporting.build_report --week week03
```

Default: 200 dev queries, T5-small, top-3 passages from BM25. ROUGE-L / BLEU / EM / Token-F1 against MS MARCO QA v2.1 reference answers. CPU runtime ~5–15 min.

Tunable knobs in `configs/baseline.yaml`:
- `generation.model_name` (`t5-small`, `t5-base`, `facebook/bart-base`, …)
- `generation.num_eval_queries`
- `generation.top_k_passages`

The runner is **retrieval-source agnostic** — feed it any TREC-format
`run.tsv` (BM25 / dense / reranked) via the CLI flags. Defaults preserve
the W3 BM25 baseline behaviour exactly; explicit flags pick a different
upstream:

```bash
# Reranked → T5-small, on the queries the reranker actually covers
python experiments/run_generation_baseline.py \
    --input-run outputs/week05_reranker/run.tsv \
    --output-dir outputs/week03_generation_reranked \
    --retrieval-source reranked
```

Use `--restrict-to-run <other_run.tsv>` to force two generation runs to
evaluate on the SAME 200-query subsample even when their upstream
retrievers cover different query sets — see the "Generation × retrieval
source" section above for the BM25-vs-reranked comparison this enables.

### Week 4 — Dense retrieval (sampled corpus)

Requires Week 2 to have produced `data/processed/bm25_index_msmarco/doc_ids.json`
(the doc_id pool the qrels-anchored sampler draws from).

```bash
python experiments/run_dense_retrieval.py
python -m src.reporting.build_report --week week04
```

First run: ~15 min to encode the 50k sampled passages, ~20 s for FAISS search.
Subsequent runs reuse the cached FAISS index. Tunable knobs:
- `dense.model_name` (e.g. `sentence-transformers/all-MiniLM-L6-v2`,
  `sentence-transformers/msmarco-MiniLM-L6-cos-v5`)
- `dense.sample_size` (default 50000; grows the pool, shrinks recall)
- `dense.compare_bm25_on_sample` (head-to-head on the same sample)

### Week 5 — Cross-encoder reranking

Requires Week 4 to have produced `outputs/week04_dense/run.tsv`.

```bash
python experiments/run_reranker.py
python -m src.reporting.build_report --week week05
```

Default: rerank the top-100 dense candidates per query with
`cross-encoder/ms-marco-MiniLM-L-6-v2`. CPU-only runtime scales linearly with
the number of queries — full 6,980 queries × top-100 is ~6 hours on a 6-core
laptop. Use `--num-eval-queries` for a deterministic subsample.

```bash
# fast smoke test (~1 min)
python experiments/run_reranker.py --num-eval-queries 50 --rerank-top-k 100
# canonical baseline (~50 min on CPU with batch 128, OMP=12)
OMP_NUM_THREADS=12 python experiments/run_reranker.py --num-eval-queries 1000
```

Tunable knobs in `configs/baseline.yaml` under `reranker:`:
- `reranker.model_name` (any HF cross-encoder)
- `reranker.rerank_top_k` (depth; cost is O(K))
- `reranker.batch_size`, `reranker.max_length`

## 5. Run the prototype notebooks

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/week01_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/week02_retrieval.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/week03_generation.ipynb
```

First run downloads:
- HuggingFace `ms_marco` v2.1 validation split (~500 MB) — used by W1, W2
- T5-small (~250 MB) — used by W3

Notebook results are **prototypes**: small samples, closed-set retrieval, hand-written eval queries. Do not cite as benchmark numbers.

## 6. Configuration

All knobs live in [configs/baseline.yaml](configs/baseline.yaml). Key ones:

| Key | Effect |
|-----|--------|
| `retrieval.k1`, `retrieval.b` | BM25 hyperparameters |
| `retrieval.top_k` | Depth of saved run (default 1000) |
| `retrieval.chunk_size` | Checkpoint cadence in queries (default 200). Smaller = more durable, more I/O. |
| `retrieval.n_threads` | `0` = sequential (default, matches our 0.1703 baseline). `-1` = all CPUs. Try `-1` for new runs. |
| `data.corpus_limit` | Set to a small int for a smoke test that does **not** reproduce official numbers; leave `null` for real runs. |
| `generation.num_eval_queries` | Size of the W3 eval subset |

## 6.5. Reproducibility status

Current state of the engineering scaffold (what works today; what's still TODO).

| Area | Status | Notes |
|---|---|---|
| Default unit tests | ✅ | `make test` / `pytest -q` — 78 tests, no network, no heavy deps. Slow tests excluded by `[tool.pytest.ini_options]`. |
| Slow tests | ✅ (skips gracefully) | `make test-slow` includes `@pytest.mark.slow`. HF metric scripts skip if unavailable; never hard-fail offline. |
| Lockfile | ✅ basic | `requirements-lock.txt` is pip-freeze-style; sub-dep transitive closure + hash pinning are TODO (would need pip-tools / uv). |
| Installable package | ✅ basic | `pip install -e .` registers `src` via `pyproject.toml`. Existing `sys.path.insert` shims in `experiments/` and `scripts/` are kept for now to avoid touching unrelated code; removing them is a TODO. |
| CI | ✅ basic | `.github/workflows/ci.yml`: pytest + ruff on push/PR to main. Does not run slow tests or download MS MARCO data. |
| Lint | ✅ minimal | `ruff` with `F` + `W` (pyflakes + whitespace). Style rules (`E`, `I`, `UP`, …) intentionally OFF on the first pass. |
| Artifact manifest | ✅ wired | `src/util/manifest.py` provides `build_manifest()` / `write_manifest()` / `write_run_manifest()`. All 4 runners write `outputs/<week>/manifest.json` alongside `metrics.json`. Captures git commit + dirty flag, command, config hash, dependency-file hashes (requirements / lockfile / pyproject), and per-output sha256 (truncated). |
| Numbers in `reports/generated/*.pdf` | ⚠️ historical | Reflect the dev environment at the time the PDF was committed. Re-running with `requirements-lock.txt` is the closest we get to reproduction today. |

Current limitations to be aware of:

- The lockfile reflects the author's **macOS CPU-only** dev environment. Linux / CUDA may resolve different versions; install `torch` from the appropriate PyTorch index *first*.
- The corpus, encoder, and reranker checkpoints are downloaded by `ir_datasets` / HuggingFace at first run and **are not checksummed by the project**. If upstream changes silently, numbers may shift.
- `experiments/run_*.py` still rely on `sys.path.insert(0, PROJECT_ROOT)` at the top of the file. `pip install -e .` makes this unnecessary, but the shim is kept until a future pass removes them.

## 7. Known limitations

- **Tokenizer mismatch with Anserini.** Our 0.1703 vs reference 0.184 is mostly tokenizer-induced (`bm25s` default tokenizer ≠ Lucene `EnglishAnalyzer`). Acceptable for a single-machine pure-Python pipeline.
- **CPU-only retrieve is slow at 8.8M docs.** ~70 min for 6,980 queries. `n_threads=-1` may help; not yet benchmarked on this corpus.
- **Generation: pretrained T5-small, no fine-tuning.** Numbers will be low on overlap-based metrics. Fine-tuning is in scope for a future week.
- **NumPy 2.x runtime warning.** Some compiled deps (torch) were built against NumPy 1.x. Cosmetic on this codebase; downgrade to `numpy<2` if it ever causes a real failure.

## 8. Next

- Try `n_threads: -1` for a faster W2 re-run.
- Hybrid first stage (BM25 + dense fusion, e.g. RRF) → cross-encoder rerank.
- Rerank the BM25 top-100 as well, and compare delta-from-CE on a weak vs
  strong first stage.
- Supervised fine-tuning of the generation model on `(question, gold passage, answer)` triples from MS MARCO QA v2.1.

## 9. License

See [LICENSE](LICENSE).

# MS MARCO GenQA

Retrieval-augmented question answering on MS MARCO. The repo has two parallel tracks:

- **Script pipeline** (`experiments/`, `src/`) — reproducible, official-corpus, structured outputs + auto-generated reports. This is the source of truth for benchmark numbers.
- **Notebooks** (`notebooks/`) — small prototype experiments on sampled data. Useful for narrative and visualization, **not** the source of benchmark numbers.

## 1. Status

| Week | Pipeline | Result | Notebook (prototype) |
|------|----------|--------|----------------------|
| W1 — EDA | n/a | n/a | `notebooks/week01_eda.ipynb` ✅ runs end-to-end |
| W2 — BM25 retrieval | `experiments/run_retrieval.py` | **MRR@10 = 0.1703**, Recall@100 = 0.6212, Recall@1000 = 0.8154 on dev/small (6,980 q) | `notebooks/week02_retrieval.ipynb` ✅ MRR@10 = 0.1956 on sampled closed-set (n=30, optimistic) |
| W3 — RAG generation | `experiments/run_generation_baseline.py` | not yet run | `notebooks/week03_generation.ipynb` ✅ 3-passage toy demo with T5-small |

Reference: published Anserini/Lucene BM25 baseline on MS MARCO dev/small is MRR@10 ≈ 0.184. Our `bm25s`-based 0.1703 is in the same ballpark; the gap is consistent with tokenizer differences.

## 2. Directory layout

```
configs/        baseline.yaml — paths + retrieval/generation/eval knobs
experiments/    run_retrieval.py, run_generation_baseline.py
src/
  data/         msmarco.py — ir_datasets loader for the official corpus
  retrieval/    bm25.py    — bm25s wrapper with save/load + chunked retrieve
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
pip install -r requirements.txt
# optional: PDF report generation (markdown report works without these)
brew install pandoc                 # macOS
brew install --cask basictex        # macOS LaTeX engine
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

## 7. Known limitations

- **Tokenizer mismatch with Anserini.** Our 0.1703 vs reference 0.184 is mostly tokenizer-induced (`bm25s` default tokenizer ≠ Lucene `EnglishAnalyzer`). Acceptable for a single-machine pure-Python pipeline.
- **CPU-only retrieve is slow at 8.8M docs.** ~70 min for 6,980 queries. `n_threads=-1` may help; not yet benchmarked on this corpus.
- **Generation: pretrained T5-small, no fine-tuning.** Numbers will be low on overlap-based metrics. Fine-tuning is in scope for a future week.
- **NumPy 2.x runtime warning.** Some compiled deps (torch) were built against NumPy 1.x. Cosmetic on this codebase; downgrade to `numpy<2` if it ever causes a real failure.

## 8. Next

- Run the W3 RAG baseline against the W2 retrievals.
- Try `n_threads: -1` for a faster W2 re-run.
- Add a Sentence-BERT bi-encoder retriever and a hybrid BM25+dense fusion.
- Add a cross-encoder reranker over the BM25 top-100.
- Supervised fine-tuning of the generation model on `(question, gold passage, answer)` triples from MS MARCO QA v2.1.

## 9. License

See [LICENSE](LICENSE).

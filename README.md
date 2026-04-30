# MS MARCO GenQA

A research project on **retrieval-augmented question answering (RAG)** built on the MS MARCO dataset.

## 1. Current Status

The project is moving from notebook-only prototype to a reproducible,
script-based experiment pipeline.

| Week | Goal | Status |
|------|------|--------|
| Week 1 | EDA on MS MARCO (queries, passages, query types) | ✅ done — see [notebooks/week01_eda.ipynb](notebooks/week01_eda.ipynb) |
| Week 2 | **Official BM25 retrieval baseline** on the full MS MARCO Passage corpus | 🟡 pipeline implemented (`experiments/run_retrieval.py`); awaiting first end-to-end run |
| Week 3 | **RAG generation baseline** on real Week 2 retrievals | 🟡 pipeline implemented (`experiments/run_generation_baseline.py`); awaiting first end-to-end run |
| Weeks 4–6 | Dense retrieval, reranking, generation fine-tuning | ❌ not started |

The legacy notebooks (`notebooks/week02_*.ipynb`, `notebooks/week03_*.ipynb`)
remain in the repo as **prototype / pipeline-verification artifacts only**.
Their results — including the MRR@10 = 0.2775 number on a sampled 49k
closed-set corpus, and the 3-passage T5-small toy demo — are *not* the
official deliverables and should not be cited as benchmark results.

The official deliverables are produced by the script pipeline below and
written to `outputs/` and `reports/generated/`.

## 2. Repository Structure

```
msmarco-genqa/
├── configs/
│   └── baseline.yaml             # paths, retrieval params, eval set sizes
├── experiments/
│   ├── run_retrieval.py          # Week 2: build BM25 index + evaluate dev/small
│   └── run_generation_baseline.py # Week 3: RAG generation on Week 2 retrievals
├── src/
│   ├── data/msmarco.py           # ir_datasets loader for the official corpus
│   ├── retrieval/bm25.py         # bm25s wrapper with save/load
│   ├── generation/rag_generator.py
│   ├── evaluation/
│   │   ├── retrieval.py          # MRR@k, Recall@k, nDCG@k
│   │   └── generation.py         # ROUGE-L, BLEU, EM, Token-F1
│   ├── reporting/build_report.py # markdown + PDF report generator
│   └── bm25_retriever.py         # legacy rank_bm25 wrapper (used by week3 notebook only)
├── reports/
│   ├── templates/                # markdown templates with {{placeholders}}
│   └── generated/                # filled-in markdown + PDF (gitignored)
├── outputs/                      # experiment outputs (gitignored)
├── data/                         # raw/processed/cache (gitignored)
├── figures/                      # static plots from notebooks (committed)
└── notebooks/                    # exploratory + prototype notebooks (do not cite as official)
```

Conventions:

- Code runs from the project root. All scripts add the project root to
  `sys.path` themselves; you do not need to set `PYTHONPATH`.
- Paths in scripts come from `configs/baseline.yaml` or are derived from
  `PROJECT_ROOT`. No hardcoded relative paths.
- `outputs/`, `data/raw/`, `data/processed/`, `reports/generated/` are
  gitignored. Only the directory structure (`.gitkeep` files) is committed.

## 3. Setup

Recommended Python: 3.10+

```bash
pip install -r requirements.txt
# optional, for PDF report generation
brew install pandoc           # macOS
brew install --cask basictex  # macOS LaTeX engine; alternative: `mactex`
# Linux:
# sudo apt-get install pandoc texlive-xetex
```

Without pandoc the pipeline still produces the markdown report; only the
PDF step is skipped.

## 4. Running Week 2 — BM25 retrieval baseline

```bash
python experiments/run_retrieval.py
python -m src.reporting.build_report --week week02
```

What happens:

1. `run_retrieval.py` downloads (first run only) the official MS MARCO
   Passage corpus and dev/small queries via `ir_datasets`. Expect ~3 GB of
   downloads and tens of minutes of indexing on first run; the index is
   cached to `data/processed/bm25_index_msmarco/` for re-use.
2. Top-1000 retrieval is run for all dev/small queries.
3. Outputs:
   - `outputs/week02_bm25/metrics.json` (MRR@10 / Recall@100 / Recall@1000)
   - `outputs/week02_bm25/run.tsv` (TREC-format full top-1000 run)
   - `outputs/week02_bm25/examples.jsonl` (qualitative samples)
4. `build_report.py` fills `reports/templates/week02_bm25.md`, writes
   `reports/generated/week02_bm25.md`, and (if pandoc is installed)
   `reports/generated/week02_bm25.pdf`.

Reference number: the published Anserini/Lucene BM25 baseline on this split
is approximately MRR@10 ≈ 0.184. Our `bm25s`-based result should be in the
same ballpark.

## 5. Running Week 3 — RAG generation baseline

Requires Week 2 to have produced `outputs/week02_bm25/run.tsv`.

```bash
python experiments/run_generation_baseline.py
python -m src.reporting.build_report --week week03
```

What happens:

1. Loads the Week 2 BM25 run from `outputs/week02_bm25/run.tsv`.
2. Cross-references dev/small query ids with MS MARCO QA v2.1 (HuggingFace
   `ms_marco`, validation split) to recover human-written reference
   answers.
3. Samples a CPU-friendly evaluation subset (200 queries by default; see
   `generation.num_eval_queries` in `configs/baseline.yaml`).
4. Generates answers with `t5-small` conditioned on the BM25 top-3
   passages.
5. Outputs:
   - `outputs/week03_generation/predictions.jsonl`
   - `outputs/week03_generation/metrics.json`  (ROUGE-L, BLEU, EM, Token-F1)
   - `outputs/week03_generation/examples.jsonl`
6. `build_report.py` fills `reports/templates/week03_generation.md` and
   produces the corresponding markdown + PDF.

## 6. Configuration

All knobs live in [configs/baseline.yaml](configs/baseline.yaml):

- `retrieval.k1`, `retrieval.b` — BM25 hyperparameters
- `retrieval.top_k` — depth of the saved run (default 1000)
- `data.corpus_limit` — set to a small int (e.g. 200000) for a development
  smoke test that does **not** reproduce official numbers; leave `null`
  for the official baseline
- `generation.model_name` — swap `t5-small` for `t5-base`,
  `facebook/bart-base`, etc.
- `generation.num_eval_queries` — Week 3 eval-set size

## 7. Notebooks vs Scripts

- Notebooks are exploratory artifacts (data analysis, qualitative
  inspection). They are **not** the source of truth for benchmark numbers.
- Scripts in `experiments/` produce reproducible outputs and structured
  metrics.json files. Reports under `reports/generated/` are the canonical
  write-ups.

## 8. Roadmap

The next units of work are:

- Week 4: Sentence-BERT bi-encoder retriever + hybrid BM25/dense fusion.
- Week 5: cross-encoder reranker over the BM25 top-100.
- Week 6: supervised fine-tuning of the generation model on
  `(question, gold passage, answer)` triples.

Each will reuse the same data loader, evaluation metrics, output schema, and
report template / generator, so the comparison across weeks stays
apples-to-apples.

## 9. License

See [LICENSE](LICENSE).

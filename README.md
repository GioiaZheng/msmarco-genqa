# MS MARCO Generative QA System

A practical, engineering-oriented project to build a **retrieve-then-generate** QA system on **MS MARCO**.
The goal is to move from a solid lexical baseline to dense retrieval, reranking, answer generation, and end-to-end evaluation.

---

## 1) Project Objective

Build a production-style QA pipeline with clear milestones:

1. Understand and profile MS MARCO data.
2. Build a reliable lexical retriever baseline (BM25).
3. Add dense retrieval and retrieval optimization.
4. Add reranking and generation modules.
5. Integrate into a complete inference/evaluation pipeline.
6. Produce reproducible reports and final deliverables.

---

## 2) Current Repository Status

### Completed in this repository

- **Week 1: Dataset Analysis (Onboarding)**
  - Dataset exploration and descriptive analysis.
  - Query/passages length distribution analysis and reporting.
- **Week 2: Lexical Retrieval Baseline**
  - BM25-based retriever implementation.
  - Retrieval experiment and baseline reporting (MRR@10).

### Available artifacts

- Notebooks
  - `notebooks/week01_eda.ipynb`
  - `notebooks/week02_retrieval.ipynb`
- Source code
  - `src/bm25_retriever.py`
- Reports
  - `reports/week01_dataset_analysis.md`
  - `reports/week02_retrieval_report.md`
  - `reports/query_length_distribution.png`
  - `reports/passage_length_distribution.png`

---

## 3) Baseline Metric (Current)

| Module | Metric | Result |
|---|---|---|
| BM25 lexical retrieval | MRR@10 | **0.2716** |

---

## 4) 12-Week Work Plan (Professional Roadmap)

> The roadmap below is the full plan. As of now, this repo has completed Week 1–2 deliverables.

| Week | Stage | Core Task | Detailed Work | Weekly Deliverable |
|---|---|---|---|---|
| 1 | Onboarding | Understand MS MARCO | 1) Read MS MARCO docs and identify three key tasks: Passage Retrieval, Document Retrieval, QA Generation. 2) Download and inspect MS MARCO Passage Ranking (~880k passages) and MS MARCO QA v2.1. 3) Build Python/PyTorch environment. | **MS MARCO data understanding report**: task breakdown, dataset stats, query length distribution, answer type distribution. |
| 2 | Baseline Build | Lexical retrieval pipeline | 1) Build BM25 index on MS MARCO passages using Elasticsearch/Pyserini (or equivalent Lucene-based stack). 2) Evaluate and compute MRR@10. 3) Review top-100 retrieval quality and error cases. | **Runnable lexical retrieval demo + BM25 baseline report** (including Dev-set MRR). |
| 3 | Baseline Build | Generation baseline pipeline | 1) Load pretrained generative model (e.g., BART/T5 or Chinese-compatible alternatives when needed). 2) Build a simple retrieve-then-generate pipeline (top-k passages as context). 3) Measure ROUGE-L, BLEU, etc. | **Generation baseline report** with sample outputs and auto-metric analysis. |
| 4 | Core Optimization | Retrieval model optimization | 1) Implement dense retrieval with SBERT dual-encoder. 2) Build vector index on passages (e.g., FAISS). 3) Compare BM25 vs dense retrieval on multiple metrics. | **Improved retrieval report + retrieval optimization code** (with gains compared to lexical baseline). |
| 5 | Core Optimization | Reranking | 1) Introduce cross-encoder (e.g., MiniLM) reranker. 2) Train/fine-tune on MS MARCO qrels. 3) Evaluate MRR after reranking. | **Reranking effect analysis** including NDCG@10 and case studies. |
| 6 | Core Optimization | Generation model fine-tuning | 1) Fine-tune generation model on (query, retrieved passage, answer) tuples (e.g., SFT). 2) Compare answer quality before vs after fine-tuning (e.g., BERTScore + human checks). | **Fine-tuning generation model report** with loss curves and qualitative examples. |
| 7 | Advanced Exploration | Explainability and citation grounding | 1) Explore citation-based answer generation (prompt engineering or constrained decoding). 2) Add source highlighting/alignment mechanism to reduce hallucination (optionally reference MS MARCO passage spans). | **Citation-grounded QA analysis**: citation accuracy and source alignment metrics. |
| 8 | Advanced Exploration | Long-document handling | 1) Switch to or test long-context models for document-scale ranking. 2) Explore chunking + retrieval strategies for long passages/documents and merge for generation. | **Long-context demo + strategy note** for long-document processing. |
| 9 | Engineering | System integration and acceleration | 1) Integrate retrieval, reranking, generation into one Python inference pipeline. 2) Optional ONNX Runtime optimization for latency. 3) Run full pipeline evaluation on MS MARCO Eval set. | **Deployable inference code + performance report** (latency comparison pre/post optimization). |
| 10 | Final Output | Final evaluation | 1) Evaluate on official MS MARCO Eval setup (or CodaLab-compatible setup). 2) Aggregate all experiments and write full technical report draft. | **Final evaluation report** with side-by-side metrics vs baselines. |
| 11 | Final Output | Research-style report | 1) Complete project report/paper draft (background, methods, experiments, conclusions). 2) Prepare project presentation deck and demo video. | **Project technical report draft + presentation deck + demo video.** |
| 12 | Wrap-up | Final presentation and handoff | 1) Final presentation. 2) Code cleanup, README polish, and final repo handoff. | **Final presentation + cleaned GitHub repository.** |

---

## 5) Repository Structure

```text
msmarco-genqa/
├── notebooks/
│   ├── week01_eda.ipynb
│   └── week02_retrieval.ipynb
├── reports/
│   ├── week01_dataset_analysis.md
│   ├── week02_retrieval_report.md
│   ├── query_length_distribution.png
│   └── passage_length_distribution.png
├── src/
│   └── bm25_retriever.py
├── LICENSE
└── README.md
```

---

## 6) Next Recommended Repository Evolution

To keep later stages maintainable and professional, the following structure is recommended when Week 3+ implementation starts:

```text
msmarco-genqa/
├── configs/
├── data/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   ├── retrieval/
│   ├── rerank/
│   ├── generation/
│   ├── pipeline/
│   └── evaluation/
├── tests/
└── README.md
```

This README will be updated milestone-by-milestone as implementation expands.

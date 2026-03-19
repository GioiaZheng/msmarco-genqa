# MS MARCO Retrieval-Augmented QA System
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Task](https://img.shields.io/badge/task-Information%20Retrieval-blue)
![Focus](https://img.shields.io/badge/focus-RAG%20System-green)

🚧 Work in Progress — actively developing dense retrieval, reranking, and RAG components.

An end-to-end retrieval-augmented question answering (RAG) system built on the MS MARCO dataset, combining classical information retrieval methods with neural ranking models.

This project implements a multi-stage retrieval pipeline including:

- BM25 lexical retrieval (baseline)
- Dense retrieval (planned)
- FAISS-based indexing
- Transformer-based reranking
- Retrieval-augmented generation (RAG)

The goal is to explore scalable and modular architectures for modern information retrieval and open-domain QA systems.


---

# Project Overview


Modern QA systems typically follow a **retrieve → rerank → generate** pipeline.

This project implements that pipeline step by step:

```

User Query
   ↓
Retriever (BM25 / Dense)
   ↓
Top-K Candidates
   ↓
Reranker (Cross-Encoder)
   ↓
Ranked Context
   ↓
Generator (RAG / LLM)
   ↓
Final Answer

```

The goal is to move from a strong **lexical baseline** toward a full **retrieval-augmented generation system**.

## Why This Project

Retrieval-augmented generation (RAG) systems are becoming a core paradigm in modern NLP and search systems.

This project explores how combining classical IR techniques (BM25) with neural models can improve retrieval quality and downstream question answering performance.

It serves as a foundation for building scalable and modular QA systems aligned with real-world search and AI applications.

---

# Repository Status

### Current Progress

- Implemented BM25-based retrieval pipeline
- Built passage corpus and indexing pipeline
- Evaluated retrieval performance using MRR@10
- Conducted dataset analysis and query distribution studies

## Key Results

- BM25 baseline achieves **MRR@10 = 0.2327** on MS MARCO

## Tech Stack

- Python
- PyTorch (planned)
- FAISS (planned)
- BM25 (rank-bm25)
- HuggingFace Transformers (planned)
- MS MARCO dataset

---

# Baseline Performance

| Retrieval Model | Metric | Score |
|---|---|---|
| BM25 | MRR@10 | **0.2327** |

This score falls within the expected performance range for BM25 on MS MARCO.

---

# Repository Structure

```

msmarco-genqa/
├── notebooks/
│   ├── week01_eda.ipynb
│   └── week02_retrieval.ipynb
│
├── reports/
│   ├── week01_dataset_analysis.md
│   ├── week02_retrieval_report.md
│   ├── query_length_distribution.png
│   ├── passage_length_distribution.png
│   ├── rr_distribution.png
│   └── hit_rank_distribution.png
│
├── src/
│   └── bm25_retriever.py
│
├── LICENSE
└── README.md

````

---

# Installation

Clone the repository:

```bash
git clone https://github.com/GioiaZheng/msmarco-genqa.git
cd msmarco-genqa
````

Install dependencies:

```bash
pip install datasets
pip install rank-bm25
pip install matplotlib
pip install pandas
```

---

# Running the Experiments

### Week 1 — Dataset Exploration

Run the exploratory analysis notebook:

```
notebooks/week01_eda.ipynb
```

This notebook includes:

* Query length distribution
* Passage length distribution
* Keyword analysis
* Query type analysis

---

### Week 2 — BM25 Retrieval Baseline

Run:

```
notebooks/week02_retrieval.ipynb
```

This notebook implements:

* Passage corpus construction
* BM25 indexing
* Top-k passage retrieval
* Retrieval evaluation using **MRR@10**
* Retrieval result analysis

---

# Experimental Roadmap

The project follows a 12-week research roadmap.

| Week | Stage        | Task                         |
| ---- | ------------ | ---------------------------- |
| 1    | Onboarding   | MS MARCO dataset exploration |
| 2    | Baseline     | BM25 lexical retrieval       |
| 3    | Baseline     | Generation baseline          |
| 4    | Optimization | Dense retrieval              |
| 5    | Optimization | Reranking                    |
| 6    | Optimization | Generation fine-tuning       |
| 7    | Exploration  | Citation-grounded QA         |
| 8    | Exploration  | Long-document retrieval      |
| 9    | Engineering  | End-to-end QA pipeline       |
| 10   | Evaluation   | System benchmarking          |
| 11   | Reporting    | Final technical report       |
| 12   | Presentation | Final project presentation   |

---

# Planned Repository Evolution

As the project expands, the repository will evolve into a modular research framework:

```
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

---

# Future Work

Next steps include implementing **dense retrieval models** to overcome the limitations of lexical matching.

Planned improvements:

* Dense passage retrieval (SBERT)
* Cross-encoder reranking
* Retrieval-Augmented Generation (RAG)
* End-to-end QA pipeline evaluation

---

# References

Nguyen, Tri, et al.
**MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.**
CoRR abs/1611.09268 (2016).
](https://github.com/GioiaZheng/msmarco-genqa)

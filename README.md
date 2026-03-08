# MS MARCO Generative QA System

A practical **retrieve-then-generate question answering system** built on the **MS MARCO dataset**.

This project progressively builds a full QA pipeline starting from a **lexical retrieval baseline (BM25)** and moving towards **dense retrieval, reranking, and retrieval-augmented generation (RAG)**.

The repository is structured as an experimental framework for studying modern **information retrieval and open-domain question answering systems**.

---

# Project Overview

Modern QA systems typically follow a **retrieve → rerank → generate** pipeline.

This project implements that pipeline step by step:

```

User Query
↓
Retriever (BM25 / Dense Retrieval)
↓
Top-K Passages
↓
Reranker (Cross-Encoder)
↓
Filtered Context
↓
Generative Model
↓
Final Answer

```

The goal is to move from a strong **lexical baseline** toward a full **retrieval-augmented generation system**.

---

# Repository Status

### Completed

**Week 1 — Dataset Analysis**

- MS MARCO dataset exploration
- Query and passage distribution analysis
- Exploratory data analysis (EDA)

**Week 2 — Lexical Retrieval Baseline**

- BM25 retriever implementation
- Passage corpus construction
- Retrieval evaluation using **MRR@10**
- Retrieval behavior analysis

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

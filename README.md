# Generative Search QA System based on MS MARCO

Research Project  
Information Retrieval | Question Answering | MS MARCO

This project explores how to build a **Generative Search Question Answering (QA) system** using the **MS MARCO dataset**.

The goal is to combine **information retrieval** techniques with **natural language generation** to answer user queries effectively.

The overall system will follow a **retrieval-augmented pipeline**:

Query  
↓  
Passage Retrieval  
↓  
Context Selection  
↓  
Answer Generation

---

# Project Status

Current progress:

Week 1 — Dataset Exploration ✅ Completed  
Week 2 — Retrieval System 🚧 In Progress  

Future stages will include dense retrieval and generative QA.

---

# Project Objectives

The objectives of this project include:

- Understanding the MS MARCO dataset
- Performing exploratory data analysis (EDA)
- Implementing classical retrieval algorithms
- Building a passage retrieval pipeline
- Extending the system to retrieval-augmented generation (RAG)

This project aims to simulate the architecture of modern **search engines and QA systems**.

---

# Dataset

This project uses the **MS MARCO v2.1 dataset**, released by Microsoft for machine reading comprehension and information retrieval research.

The dataset is constructed from **real anonymized Bing search queries**.

Each example contains:

- **query** — user search query
- **answers** — human-generated answers
- **passages** — candidate passages retrieved from web pages
- **query_type**
- **urls**

Dataset splits:

| Split | Number of Queries |
|------|------|
| Train | 808,731 |
| Validation | 101,093 |
| Test | 101,092 |

Official dataset page:

https://microsoft.github.io/msmarco/

---

# MS MARCO Tasks

The MS MARCO dataset supports several research tasks.

### Passage Retrieval

Passage retrieval aims to identify **relevant passages** that contain information useful for answering a query.

Typical workflow:

Query → Retrieve passages → Use passages as answer context

Common models include:

- BM25
- Dense Passage Retrieval (DPR)
- BERT-based retrievers

---

### Document Retrieval

Document retrieval operates at the **document level**.

Instead of returning passages, the system retrieves entire documents related to a query.

| Task | Retrieval Unit |
|----|----|
| Passage Retrieval | Passages |
| Document Retrieval | Full documents |

Document retrieval is widely used in traditional search engines.

---

### Question Answering Generation

In QA generation tasks, the system must produce a **natural language answer**.

Typical pipeline:

Query  
↓  
Retrieve relevant passages  
↓  
Generate final answer

Modern QA systems often combine retrieval and generation in **Retrieval-Augmented Generation (RAG)** frameworks.

---

# Project Structure

```

msmarco-genqa
│
├── notebooks
│   └── week01_eda.ipynb
│
├── reports
│   └── week01_dataset_analysis.md
│
├── src
│
├── data
│
└── README.md

```

The repository is organized to separate:

- experiments (`notebooks`)
- analysis reports (`reports`)
- implementation code (`src`)
- datasets (`data`)

---

# Week 1 — Dataset Exploration

Week 1 focused on understanding the **MS MARCO dataset**.

Tasks completed:

- Loaded MS MARCO using HuggingFace datasets
- Explored dataset structure
- Performed exploratory data analysis
- Analyzed query length distribution
- Analyzed passage length distribution
- Analyzed answer types
- Generated visualizations
- Wrote a detailed analysis report

Key insights:

- Most queries contain **3–6 words**
- Query distribution is **right-skewed**
- Passages typically contain **40–80 words**
- Many answers are **short factual responses**

Notebook:

```

notebooks/week01_eda.ipynb

```

Report:

```

reports/week01_dataset_analysis.md

```

---

# Week 2 — Retrieval System (In Progress)

The objective of Week 2 is to build a **passage retrieval system**.

Planned tasks:

- Construct a passage corpus from MS MARCO
- Implement a **BM25 retrieval model**
- Retrieve top-k relevant passages for a query
- Inspect retrieval quality
- Document the retrieval pipeline

Planned outputs:

```

notebooks/week02_retrieval.ipynb
src/bm25_retriever.py
reports/week02_retrieval_report.md

```

---

# Retrieval Pipeline (Planned)

The retrieval system will follow this pipeline:

Query  
↓  
Tokenization  
↓  
BM25 Scoring  
↓  
Top-k Passage Retrieval

The retrieved passages will later be used as context for answer generation.

---

# How to Run

## Install dependencies

```

pip install datasets
pip install rank-bm25
pip install torch
pip install jupyter

```

---

## Launch Jupyter

```

jupyter notebook

```

Open the notebook:

```

notebooks/week01_eda.ipynb

```

---

# Future Work

Future extensions of this project include:

- Dense retrieval using **BERT embeddings**
- Vector search with **FAISS**
- Retrieval-Augmented Generation (RAG)
- Generative QA models such as **T5** or **GPT**

Future pipeline:

Query  
↓  
Dense Retriever  
↓  
Top-k Passages  
↓  
Generative Model  
↓  
Final Answer

---

# References

MS MARCO Dataset  
https://microsoft.github.io/msmarco/

HuggingFace Datasets  
https://huggingface.co/datasets

BM25 Retrieval Model  
https://en.wikipedia.org/wiki/Okapi_BM25

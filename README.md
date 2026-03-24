# MS MARCO GenQA

A lightweight research prototype for **retrieval-augmented question answering (RAG)** on the MS MARCO v2.1 dataset.

---

## 1. Overview

This repository implements a simplified pipeline for open-domain question answering:

Retrieve → (Re-rank, future) → Generate

The goal is to establish **reproducible baselines** and analyze the interaction between retrieval quality and generation performance under constrained experimental settings.

This project emphasizes:
- clarity of system design
- interpretability of results
- extensibility toward full-scale RAG systems

---

## 2. Pipeline

The current system follows a minimal retrieval-augmented generation workflow:

```

Query
↓
BM25 Retrieval
↓
Top-k Passages
↓
T5-small Generator
↓
Answer

````

---

## 3. Quick Start

### Environment

Recommended Python: 3.10+

```bash
pip install datasets rank-bm25 matplotlib pandas transformers torch
````

### Run Experiments

```bash
jupyter notebook notebooks/week02_retrieval.ipynb
```

---

## 4. Key Result

| Component      | Metric          |
| -------------- | --------------- |
| BM25 Retrieval | MRR@10 = 0.2775 |

Important:

* Computed on a sampled subset (~50k passages)
* Closed-set retrieval setting
* Not directly comparable to official MS MARCO benchmarks

---

## 5. Experimental Setting

To enable local experimentation, this project uses a **sampled passage corpus**:

* Constructed by flattening passages from a subset of training queries
* Duplicate passages removed
* Final corpus size: ~49,000 passages

This corresponds to a **closed-set retrieval setup**, where:

* candidate passages are query-associated
* retrieval difficulty is lower than real-world settings

The objective is **pipeline validation**, not benchmark reproduction.

---

## 6. Implementation Status

### Completed

**Week 1 — Data Analysis**

* Query and passage distribution analysis
* Query type statistics
* Visualization of dataset characteristics

**Week 2 — Retrieval Baseline**

* BM25 implementation using `rank-bm25`
* MRR@10 evaluation
* Retrieval behavior analysis

**Week 3 — Generation Baseline**

* Minimal RAG pipeline (BM25 + T5-small)
* Qualitative inspection of generated outputs
* Identification of failure cases

---

## 7. Key Observations

### Retrieval

* Strong performance under exact lexical overlap
* Weak performance under semantic variation (paraphrases, synonyms)

### Generation

* Sensitive to retrieval quality
* Hallucination occurs when relevant evidence is missing
* Output quality depends heavily on top-k passage selection

---

## 8. Repository Structure

```
msmarco-genqa/
├── notebooks/
│   ├── week01_eda.ipynb
│   ├── week02_retrieval.ipynb
│   └── week03_generation.ipynb
├── reports/        # visualization outputs only
├── src/
│   └── bm25_retriever.py
├── LICENSE
└── README.md
```

---

## 9. Roadmap

Planned extensions:

* Dense retrieval (Sentence-BERT + FAISS)
* Cross-encoder reranking
* Standardized evaluation (ROUGE, BLEU, BERTScore)
* End-to-end pipeline scripts and configuration
* Full-corpus retrieval experiments

---

## 10. Positioning

This repository should be viewed as:

a controlled experimental environment for studying retrieval–generation interaction

It is not intended as:

* a leaderboard submission
* a production-ready system

---

## 11. License

See `LICENSE`.

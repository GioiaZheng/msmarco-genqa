# MS MARCO GenQA

## 1. Project Scope
This repository is an iterative prototype for question answering on **MS MARCO v2.1**. The current architecture follows a standard IR pipeline:

**Retrieve → (Re-rank, planned) → Generate**

The immediate objective is to establish reproducible BM25 and generation baselines before adding more advanced retrieval and ranking modules.

---

## 2. Current Implementation Status

### 2.1 Implemented
- Lexical retriever based on `rank-bm25` (`BM25Retriever`).
- Week 1 exploratory data analysis (EDA) workflow and visualization outputs.
- Week 2 BM25 retrieval experiment with MRR@10 evaluation.
- Week 3 minimal generation baseline (BM25 + T5-small pipeline).

### 2.2 Planned / Not Yet Implemented
- Dense retrieval (dual-encoder/vector retrieval)
- Cross-encoder reranking
- Full end-to-end RAG evaluation framework
- Production-style training/inference scripts and test suite

> Note: the `reports/` directory currently contains image artifacts only. Stage report markdown files were removed.

---

## 3. Metric Statement and Interpretation

### 3.1 Observable Retrieval Result
- The Week 2 notebook reports: **MRR@10 = 0.2775** (sampled-subset experiment).

### 3.2 Interpretation Constraints
- This value is produced on sampled data with a local passage pool and does not represent full-corpus leaderboard performance.
- It should be treated as a **baseline feasibility check** for method comparison, not as a final system claim.

---

## 4. Repository Structure (Current)

```text
msmarco-genqa/
├── notebooks/
│   ├── week01_eda.ipynb
│   ├── week02_retrieval.ipynb
│   └── week03_generation.ipynb
├── reports/
│   ├── answer_type_by_query_type.png
│   ├── hit_rank_distribution.png
│   ├── passage_length_distribution.png
│   ├── query_length_by_query_type.png
│   ├── query_length_distribution.png
│   ├── query_type_distribution.png
│   ├── relevant_passages_by_query_type.png
│   └── rr_distribution.png
├── src/
│   └── bm25_retriever.py
├── LICENSE
└── README.md
```

---

## 5. Environment and Dependencies

Recommended Python: 3.10+

```bash
pip install datasets rank-bm25 matplotlib pandas transformers torch
```

Notes:
- Week 1/2 mainly require `datasets`, `rank-bm25`, `matplotlib`, and `pandas`.
- Week 3 generation baseline additionally requires `transformers` and `torch`.

---

## 6. Reproducibility Entry Points

### 6.1 Week 1: Data Exploration
- File: `notebooks/week01_eda.ipynb`
- Purpose: inspect query/passage distributions, query types, and relevant-passage statistics.

### 6.2 Week 2: BM25 Retrieval Baseline
- File: `notebooks/week02_retrieval.ipynb`
- Steps:
  1. Build sampled passage pool
  2. Run BM25 retrieval
  3. Evaluate with MRR@10
  4. Analyze RR and hit-rank distributions

### 6.3 Week 3: Generation Baseline
- File: `notebooks/week03_generation.ipynb`
- Purpose: run a minimal retrieval-augmented generation chain (BM25 + T5-small) and inspect failure modes.

---

## 7. Engineering Constraints and Known Limitations

1. **Scale limitation**: current experiments are mostly on sampled subsets and local candidate pools, not full-corpus retrieval.
2. **Preprocessing simplicity**: retrieval currently uses basic tokenization (lowercase + whitespace), without advanced normalization.
3. **Incomplete evaluation**: generation-side evaluation is preliminary and lacks a formal human-evaluation protocol.
4. **Deployment gap**: no standardized CLI, config management, automated tests, or model versioning pipeline yet.

---

## 8. Practical Next Steps

1. Add a dense retrieval baseline (e.g., DPR/Sentence-Transformer + FAISS).
2. Add cross-encoder reranking and measure gains in MRR/NDCG.
3. Build unified evaluation scripts with fixed splits and random seeds for reproducibility.
4. Move notebook prototypes into `src/` and `scripts/` for batch experimentation.
5. Run controlled side-by-side evaluation across BM25 / Dense / Rerank / RAG.

---

## 9. License

This project is licensed as specified in `LICENSE`.

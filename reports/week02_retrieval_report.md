# Week 2 Report: BM25 Retrieval Baseline for MS MARCO

---

# 1 Introduction

Open-domain question answering systems typically follow a **retrieve-then-read architecture**, where relevant documents or passages are first retrieved from a large corpus before answer generation is performed.

During **Week 2**, the objective of this project was to implement a classical **information retrieval baseline** using the **BM25 ranking algorithm** on the **MS MARCO dataset**.

The main goals of this stage were:

* Construct a searchable passage corpus from the MS MARCO dataset
* Implement a BM25 retrieval model
* Retrieve the top-k most relevant passages for a query
* Evaluate retrieval quality using **MRR@10**

This baseline retrieval system will later be compared with **dense neural retrieval models** and integrated into a **retrieval-augmented generation (RAG)** pipeline.

---

# 2 Dataset

This project uses the **MS MARCO v2.1 dataset** [1], a large-scale dataset introduced by Microsoft for machine reading comprehension and information retrieval tasks.

The dataset contains real anonymized search queries collected from the Bing search engine.

Each record includes:

* **query** – user search query
* **answers** – human-generated answers
* **passages** – candidate passages retrieved from web documents
* **is_selected** – binary relevance labels

Dataset statistics:

| Split      | Number of Queries |
| ---------- | ----------------- |
| Train      | ~808,000          |
| Validation | ~101,000          |
| Test       | ~101,000          |

Because the full dataset is extremely large, this experiment uses a **subset of 2000 training queries** for constructing the retrieval corpus.

After flattening all candidate passages associated with the sampled queries, the resulting corpus contains:

```
19,955 passages
```

---

# 3 Methodology

## 3.1 Corpus Construction

Each query in MS MARCO is associated with multiple candidate passages. These passages were extracted and flattened into a single corpus.

Example structure:

```
Query
 ├── Passage 1
 ├── Passage 2
 ├── Passage 3
 └── Passage N
```

All passages were appended to a global list:

```python
corpus = []
for sample in train_data:
    passages = sample["passages"]["passage_text"]
    for p in passages:
        corpus.append(p)
```

The final corpus serves as the document collection for retrieval.

---

## 3.2 Text Preprocessing

To prepare the corpus for BM25 indexing, a simple preprocessing pipeline was applied:

1. Convert text to lowercase
2. Tokenize using whitespace splitting
3. Remove empty passages

Example:

```text
"The Manhattan Project was a research program"

↓

["the", "manhattan", "project", "was", "a", "research", "program"]
```

Tokenization implementation:

```python
tokenized_corpus = [doc.lower().split() for doc in corpus]
```

---

## 3.3 BM25 Ranking Model

BM25 is a probabilistic retrieval model that ranks documents based on term frequency, inverse document frequency, and document length normalization.

The BM25 scoring function is defined as:

$$
score(D,Q)=\sum_{t \in Q} IDF(t) \cdot
\frac{f(t,D)(k_1+1)}{f(t,D)+k_1(1-b+b\frac{|D|}{avgdl})}
$$

Where:

* ( f(t,D) ) = term frequency
* ( |D| ) = document length
* ( avgdl ) = average document length
* ( k_1, b ) = hyperparameters

In this project, BM25 was implemented using the **rank-bm25** Python library.

Index construction:

```python
bm25 = BM25Okapi(tokenized_corpus)
```

BM25 retrieval requires scoring each document in the corpus for a given query.
Therefore, the retrieval complexity is approximately **O(N) per query**, where **N** is the number of passages in the corpus.

---

## 3.4 Query Retrieval

Given a query, the retrieval process follows these steps:

1. Tokenize the query
2. Compute BM25 scores for all passages
3. Rank passages by score
4. Return top-k passages

Example retrieval implementation:

```python
scores = bm25.get_scores(tokenized_query)
top_k = np.argsort(scores)[::-1][:k]
```

---

## 3.5 Retriever Module

To improve modularity and reusability, the retrieval logic was encapsulated into a reusable class:

```
src/bm25_retriever.py
```

Example usage:

```python
retriever = BM25Retriever(corpus)
results = retriever.retrieve(query, k=5)
```

This design allows the retrieval module to be easily integrated into future pipelines such as **dense retrieval** and **RAG-based QA systems**.

---

## 3.6 Retrieval Architecture

The retrieval system follows a classical **information retrieval pipeline** consisting of corpus construction, indexing, retrieval, and evaluation.

```mermaid
flowchart LR

A[MS MARCO Dataset] --> B[Passage Extraction]
B --> C[Corpus Construction]
C --> D[Text Tokenization]
D --> E[BM25 Index]
E --> F[Query Input]
F --> G[BM25 Scoring]
G --> H[Top-K Passage Retrieval]
H --> I[Evaluation MRR@10]
```

This pipeline represents a **lexical retrieval architecture**, where document ranking is based on term matching and statistical weighting.

---

# 4 Experimental Setup

## 4.1 Evaluation Metric

Retrieval performance was evaluated using **MRR@10 (Mean Reciprocal Rank at 10)**.

The reciprocal rank is defined as:

$$
RR = \frac{1}{rank}
$$

If the first relevant document appears at rank *r*, the reciprocal rank is:

| Rank | Reciprocal Rank |
| ---- | --------------- |
| 1    | 1.0             |
| 2    | 0.5             |
| 5    | 0.2             |
| 10   | 0.1             |

If no relevant document appears within the top 10 retrieved passages:

```
RR = 0
```

The final MRR score is the **mean of reciprocal ranks across all queries**.

---

## 4.2 Experimental Configuration

The experiment configuration is summarized below:

| Parameter          | Value           |
| ------------------ | --------------- |
| Corpus size        | 19,955 passages |
| Training subset    | 2000 queries    |
| Evaluation queries | 200             |
| Retrieval depth    | Top 10          |
| Model              | BM25            |

Evaluation procedure:

1. Compute BM25 scores for all passages
2. Retrieve top-10 passages
3. Identify the first relevant passage
4. Compute reciprocal rank
5. Average across all queries

---

# 5 Results

The BM25 baseline achieved the following performance:

| Model         | MRR@10     |
| ------------- | ---------- |
| BM25 Baseline | **0.2716** |

Typical BM25 performance on MS MARCO is reported between:

```
MRR@10 ≈ 0.18 – 0.28
```

The obtained result (**MRR@10 = 0.2716**) falls within the expected performance range for BM25 on MS MARCO, indicating that the implementation successfully reproduces a standard lexical retrieval baseline.

### Retrieval Evaluation Summary

```
Corpus size: 19,955 passages
Evaluation queries: 200
Retrieval depth: Top 10
MRR@10: 0.2716
```

### Example Retrieval

**Query**

```
what was the immediate impact of the success of the manhattan project
```

**Top Retrieved Passage**

```
The presence of communication amid scientific minds was equally important to the success of the Manhattan Project...
```

This example demonstrates that BM25 effectively retrieves passages containing strong lexical overlap with the query.

---

# 6 Discussion

The BM25 retrieval model demonstrates strong baseline performance due to its ability to capture lexical overlap between queries and passages.

### Advantages

* Efficient retrieval over large corpora
* Simple and interpretable ranking mechanism
* Strong baseline for information retrieval tasks

### Limitations

However, BM25 relies purely on **keyword matching** and lacks semantic understanding.

Challenges include:

* synonym mismatch
* paraphrased queries
* contextual reasoning

These limitations motivate the use of **dense neural retrieval models**, which encode semantic relationships using deep learning.

### Future Work

Future work will focus on improving retrieval performance using neural approaches:

**Week 3 – Dense Retrieval**

* Implement Sentence-BERT embeddings
* Encode queries and passages as dense vectors
* Use cosine similarity for semantic retrieval

**Week 4 – Retrieval-Augmented Generation (RAG)**

* Retrieve relevant passages
* Feed retrieved context into a generative model
* Generate answers conditioned on retrieved documents

---

# 7 Conclusion

In Week 2, we successfully implemented a **BM25-based retrieval system** on the MS MARCO dataset.

The system includes:

* corpus construction from MS MARCO passages
* text preprocessing and tokenization
* BM25 index creation
* top-k passage retrieval
* evaluation using **MRR@10**

The retrieval model achieved an **MRR@10 score of 0.2716**, demonstrating strong baseline performance.

This retrieval pipeline provides a strong lexical baseline and establishes the foundation for future experiments involving **dense retrieval models and retrieval-augmented generation systems**.

---

# References

[1] Nguyen, Tri, et al.
**MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.**
CoRR abs/1611.09268 (2016).

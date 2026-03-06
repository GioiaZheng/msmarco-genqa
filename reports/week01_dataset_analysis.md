# Week 1 Report: MS MARCO Dataset Exploration and Analysis

---

# 1 Introduction

In this project, we aim to develop a **generative search question answering system** based on the **MS MARCO dataset**. The goal of the system is to retrieve relevant passages from a large corpus and generate accurate answers to user queries.

During **Week 1**, the primary objective was to explore the MS MARCO dataset and understand its structure and statistical properties.

The key goals of this stage were:

* Set up the development environment
* Load and explore the MS MARCO dataset
* Analyze dataset structure and components
* Perform exploratory data analysis (EDA)
* Investigate query and passage characteristics

Understanding the dataset distribution is essential for designing effective **retrieval and generation pipelines** in open-domain question answering systems.

---

# 2 Dataset

This project uses the **MS MARCO (Microsoft MAchine Reading COmprehension) dataset** [1], a large-scale dataset designed for machine reading comprehension, information retrieval, and question answering tasks.

The dataset was constructed using **real anonymized Bing search queries**, paired with passages extracted from web documents.

Each dataset sample contains the following fields:

* **query** – the user search query
* **answers** – human-generated answers
* **passages** – candidate passages retrieved from web documents
* **query_type** – type of query
* **urls** – source web pages

Dataset statistics:

| Split      | Number of Queries |
| ---------- | ----------------- |
| Train      | 808,731           |
| Validation | 101,093           |
| Test       | 101,092           |

In total, the dataset contains **over one million search queries**, making it one of the largest datasets for:

* Information Retrieval (IR)
* Question Answering (QA)
* Retrieval-Augmented Generation (RAG)

---

# 3 Methodology

## 3.1 Dataset Structure

Each query in the MS MARCO dataset is associated with multiple candidate passages retrieved from web documents.

Among these passages, some are labeled as **relevant** to the query.

Example data structure:

```text
{
  query: "what was the immediate impact of the success of the manhattan project?",
  answers: [...],
  passages:
      passage_text: [...]
      is_selected: [...]
  url: [...]
}
```

Important observations include:

* Each query is associated with **multiple passages**
* Only a subset of passages is labeled as **relevant**
* Passages originate from **web documents**

This structure naturally supports **retrieval-based QA systems**, where relevant passages must first be retrieved before answer generation.

---

## 3.2 Research Tasks Supported by MS MARCO

The MS MARCO dataset supports several important tasks in information retrieval and question answering.

### Passage Retrieval

Passage retrieval focuses on retrieving the most relevant **text passages** for a given query.

Typical workflow:

```
Query → Retrieve Top-K Passages → Provide Context for Answer Generation
```

Common retrieval models include:

* BM25
* Dense Passage Retrieval (DPR)
* BERT-based retrievers

Passage retrieval is particularly important in **retrieval-augmented generation systems**.

---

### Document Retrieval

Document retrieval operates at the **document level** rather than smaller text segments.

| Task               | Retrieval Unit |
| ------------------ | -------------- |
| Passage Retrieval  | Short passages |
| Document Retrieval | Full documents |

While document retrieval is widely used in search engines, additional processing is often required to locate exact answers within retrieved documents.

---

### Question Answering Generation

Question answering generation aims to produce **natural language answers** for user queries.

Typical pipeline:

```
Query → Retrieve Relevant Passages → Generate Answer
```

This task is commonly addressed using **large language models (LLMs)** such as:

* T5
* BART
* GPT-based models

Modern QA systems often combine **retrieval and generation** to form **retrieval-augmented generation (RAG)** architectures.

---

# 4 Experimental Setup

To understand the characteristics of the dataset, we performed **exploratory data analysis (EDA)** on sampled subsets of the training data.

Sampling configuration:

| Parameter       | Value                                                        |
| --------------- | ------------------------------------------------------------ |
| Query samples   | 5,000                                                        |
| Passage samples | 2,000 queries                                                |
| Analysis tasks  | Query length, passage length, keyword frequency, answer type |

These analyses provide insights into query behavior and passage characteristics, which are important for designing effective retrieval models.

---

# 5 Results

## 5.1 Query Length Distribution

To analyze user search behavior, we measured the number of words in each query.

Query length distribution:

![Query Length Distribution](https://github.com/GioiaZheng/msmarco-genqa/blob/main/reports/query_length_distribution.png?raw=1)

### Observations

* Most queries contain **3–6 words**
* The distribution is **right-skewed**
* Very long queries are rare

This reflects typical **search engine behavior**, where users submit short keyword-style queries rather than long sentences.

---

## 5.2 Passage Length Distribution

We also analyzed the length distribution of candidate passages.

Passage length distribution:

![Passage Length Distribution](https://github.com/GioiaZheng/msmarco-genqa/blob/main/reports/passage_length_distribution.png?raw=1)

### Observations

* Most passages contain **40–80 words**
* Some passages exceed **150 words**
* Passage length distribution is moderately right-skewed

These lengths are suitable for transformer-based models that operate on relatively short context windows.

---

## 5.3 Query Keyword Analysis

To better understand linguistic patterns in queries, we computed the most frequent words.

| Word | Frequency |
| ---- | --------- |
| what | 1847      |
| is   | 1455      |
| a    | 914       |
| how  | 723       |
| in   | 722       |
| of   | 702       |
| the  | 648       |
| to   | 584       |

### Observations

Frequent query tokens include:

* **interrogative words** (what, how)
* **prepositions** (in, of, to)
* **articles** (a, the)

This confirms that many queries follow **question-style patterns**.

---

## 5.4 Answer Type Analysis

Answers in MS MARCO can vary in length and format.

We categorized answers into three types:

| Answer Type      | Count |
| ---------------- | ----- |
| Short answers    | 2462  |
| Sentence answers | 1669  |
| Numeric answers  | 869   |

### Observations

Short answers are the most common and typically correspond to:

* named entities
* definitions
* factual information

Sentence-level answers provide explanatory responses, while numeric answers represent **dates, quantities, or measurements**.

---

# 6 Discussion

The exploratory analysis reveals several important characteristics of the MS MARCO dataset.

### Dataset Characteristics

* Queries are typically **short and ambiguous**
* Each query is associated with **multiple candidate passages**
* Only a small subset of passages is **relevant**
* Answers are often **short factual responses**

These properties highlight the importance of effective **retrieval mechanisms** in open-domain QA systems.

### Implications for Retrieval Models

Because queries are short and ambiguous:

* lexical matching alone may be insufficient
* semantic retrieval models may be required
* ranking algorithms must effectively prioritize relevant passages

---

### Future Work

Based on these insights, the next stage of the project will focus on building a **retrieval system**.

Planned tasks for Week 2 include:

* Implementing a **BM25 retrieval baseline**
* Retrieving **top-k passages**
* Evaluating retrieval quality using **MRR**
* Preparing for integration with generative models

---

# 7 Conclusion

In Week 1, we conducted an exploratory analysis of the **MS MARCO dataset**.

The analysis included:

* dataset structure exploration
* query length distribution analysis
* passage length distribution analysis
* query keyword frequency analysis
* answer type classification

These insights provide a strong foundation for developing a **retrieval-based question answering system**.

The findings from this stage guide the design of the retrieval model implemented in **Week 2**.

---

# References

[1] Nguyen, Tri, et al.
**MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.**
CoRR abs/1611.09268 (2016).

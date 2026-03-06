# Week 1 Report

# MS MARCO Dataset Exploration and Analysis

## 1. Introduction

In this project, we aim to develop a **generative search question answering system** based on the **MS MARCO dataset**. The goal of the system is to retrieve relevant passages from a large corpus and generate accurate answers to user queries.

During **Week 1**, the primary objective was to:

- Set up the development environment
- Explore the MS MARCO dataset
- Understand the structure of the data
- Perform exploratory data analysis (EDA)
- Analyze query and passage characteristics

Understanding the dataset distribution is critical for designing an effective retrieval and generation pipeline.

---

# 2. Dataset Overview

The **MS MARCO (Microsoft Machine Reading Comprehension)** dataset is a large-scale dataset introduced by Microsoft for machine reading comprehension, information retrieval, and question answering tasks.

The dataset is constructed from **real anonymized Bing search queries**, paired with relevant passages and answers.

Each sample in the dataset contains:

- **query** – the user search query
- **answers** – human-generated answers
- **passages** – candidate passages retrieved from web documents
- **query_type** – type of query
- **urls** – source web pages

### Dataset Size

| Split | Number of Queries |
|------|------|
| Train | 808,731 |
| Validation | 101,093 |
| Test | 101,092 |

In total, the dataset contains **over one million search queries** and associated passages.

This large scale makes MS MARCO one of the most widely used datasets for:

- Information Retrieval (IR)
- Question Answering (QA)
- Retrieval-Augmented Generation (RAG)

---

# 3. Data Structure

Each training example contains a query and multiple candidate passages. Among these passages, some are labeled as relevant to the query.

Example structure:

```text
{
  query: "what was the immediate impact of the success of the manhattan project?",
  answers: [...],
  passages:
      passage_text: [...]
      is_selected: [...]
  url: [...]
}
````

Important observations:

* Each query is associated with **multiple passages**
* Only some passages are **relevant answers**
* Passages are extracted from **web documents**

This structure is suitable for **retrieval-based systems**, where the model retrieves relevant passages before generating answers.

---

# 4. Query Length Distribution

To better understand user query behavior, we analyzed the **length of queries** in the dataset.

The number of words in each query was computed using a sample of **5,000 queries from the training set**.

### Query Length Histogram

![Query Length Distribution](https://github.com/GioiaZheng/msmarco-genqa/blob/main/reports/query_length_distribution.png?raw=1)

### Observations

From the histogram we observe that:

* Most queries contain **3 to 6 words**
* The distribution is **right-skewed**
* Very long queries are rare

This reflects real-world search engine behavior where users typically submit **short keyword-based queries** rather than full sentences.

### Implications

Short queries create challenges for retrieval systems because:

* Queries are often **ambiguous**
* Important context may be missing
* Retrieval models must rely heavily on **semantic matching**

---

# 5. Passage Length Distribution

We also analyzed the length distribution of passages retrieved for each query.

Using a sample of **2,000 training examples**, all candidate passages were extracted and their word counts were computed.

### Passage Length Histogram

![Passage Length Distribution](https://github.com/GioiaZheng/msmarco-genqa/blob/main/reports/passage_length_distribution.png?raw=1)

### Observations

The distribution shows that:

* Most passages contain **40 to 80 words**
* Some passages extend to **over 150 words**
* The distribution is moderately right-skewed

This indicates that passages contain **sufficient context for answering queries**, but they are still relatively short compared to full documents.

### Implications

For retrieval and QA models:

* Passage length is manageable for transformer-based models
* Retrieval models must rank passages effectively
* Only a small subset of passages is actually relevant

---

# 6. Query Keyword Analysis

To better understand the linguistic patterns in search queries, we analyzed the **most frequent words** appearing in queries.

Using **5,000 queries**, we computed the top 20 most common tokens.

### Most Frequent Query Words

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

Common query terms include:

* **interrogative words**: what, how
* **prepositions**: in, of, to
* **articles**: a, the

This confirms that most queries are **question-based search queries**.

### Implications

Retrieval systems must handle:

* Question-style queries
* Informational search intents
* Natural language questions

---

# 7. Answer Type Analysis

We also analyzed the structure of answers provided in the dataset.

Answers were categorized into three types:

* **Short answers** (≤ 3 words)
* **Sentence-level answers**
* **Numeric answers**

Using a sample of **5,000 examples**, the distribution was:

| Answer Type      | Count |
| ---------------- | ----- |
| Short answers    | 2462  |
| Sentence answers | 1669  |
| Numeric answers  | 869   |

### Observations

Short answers are the most common form of response. These typically correspond to:

* named entities
* definitions
* short factual information

Sentence answers provide **explanatory information**, while numeric answers correspond to **dates, measurements, and quantities**.

### Implications

A QA system trained on this dataset must be able to:

* extract entities
* generate concise answers
* handle numerical responses

---

# 8. Key Insights

From the exploratory analysis, several important characteristics of the MS MARCO dataset emerge:

1. Queries are generally **short and ambiguous**
2. Each query is associated with **multiple candidate passages**
3. Only a subset of passages contains relevant answers
4. Answers are typically **short factual responses**

These characteristics highlight the importance of **effective retrieval mechanisms** before answer generation.

---

# 9. Next Steps

Based on the findings from Week 1, the next phase of the project will focus on building a **retrieval system**.

The following steps are planned:

1. Implement a **BM25 retrieval model**
2. Retrieve the **top-k relevant passages**
3. Evaluate retrieval quality
4. Integrate retrieval with **generative models**

This retrieval stage will form the foundation for a **retrieval-augmented generative QA system**.

---

# 10. Conclusion

In Week 1, we successfully:

* Set up the development environment
* Loaded and explored the MS MARCO dataset
* Analyzed query and passage characteristics
* Identified important patterns in search queries and answers

These insights provide a strong foundation for developing a **retrieval-based question answering system**.

Future work will focus on implementing and evaluating **information retrieval models** that can effectively retrieve relevant passages for user queries.

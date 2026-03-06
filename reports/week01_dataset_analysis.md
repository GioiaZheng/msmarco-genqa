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

# 3. MS MARCO Tasks

The MS MARCO dataset supports several important research tasks in information retrieval and question answering. Understanding these tasks is essential for designing an effective generative search system.

The three main tasks associated with MS MARCO are **Passage Retrieval**, **Document Retrieval**, and **Question Answering Generation**.

---

## 3.1 Passage Retrieval

Passage Retrieval focuses on retrieving the most relevant **passages** from a large collection of short text segments.

Given a user query, the system must identify the passages that are most likely to contain the correct answer.

Example workflow:

Query → Retrieve top-k relevant passages → Provide context for answer generation.

This task is commonly used to train and evaluate **retrieval models** such as:

- BM25
- Dense Passage Retrieval (DPR)
- BERT-based retrievers

Passage retrieval is particularly important in **retrieval-augmented generation (RAG)** systems where retrieved passages are used as input to generative models.

---

## 3.2 Document Retrieval

Document Retrieval is similar to passage retrieval but operates at the **document level** instead of smaller text segments.

In this task, the system retrieves entire documents that are relevant to the query.

Compared to passage retrieval:

| Task | Retrieval Unit |
|-----|-----|
| Passage Retrieval | Short passages |
| Document Retrieval | Full documents |

Document retrieval is commonly used in traditional search engines and large-scale information retrieval systems.

However, document retrieval often requires additional processing to locate the exact answer within the retrieved document.

---

## 3.3 Question Answering Generation

Question Answering Generation focuses on producing a **natural language answer** to a user query.

Instead of only retrieving relevant text, the model must generate an answer based on the retrieved context.

Typical pipeline:

Query → Retrieve relevant passages → Generate final answer.

This task is commonly addressed using **large language models (LLMs)** or sequence-to-sequence models such as:

- T5
- BART
- GPT-based models

In modern search systems, question answering generation is often combined with retrieval methods to form **retrieval-augmented generation (RAG)** architectures.

---

## 3.4 Relationship Between the Tasks

The three tasks are closely related and often form a unified pipeline in modern search systems:

Query  
↓  
Passage / Document Retrieval  
↓  
Context Selection  
↓  
Answer Generation  

In this project, the focus will be on building a **retrieval-based question answering system**, where relevant passages are first retrieved and then used to generate accurate answers.

---

# 4. Data Structure

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

# 5. Query Length Distribution

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

# 6. Passage Length Distribution

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

# 7. Query Keyword Analysis

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

# 8. Answer Type Analysis

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

# 9. Key Insights

From the exploratory analysis, several important characteristics of the MS MARCO dataset emerge:

1. Queries are generally **short and ambiguous**
2. Each query is associated with **multiple candidate passages**
3. Only a subset of passages contains relevant answers
4. Answers are typically **short factual responses**

These characteristics highlight the importance of **effective retrieval mechanisms** before answer generation.

---

# 10. Next Steps

Based on the findings from Week 1, the next phase of the project will focus on building a **retrieval system**.

The following steps are planned:

1. Implement a **BM25 retrieval model**
2. Retrieve the **top-k relevant passages**
3. Evaluate retrieval quality
4. Integrate retrieval with **generative models**

This retrieval stage will form the foundation for a **retrieval-augmented generative QA system**.

---

# 11. Conclusion

In Week 1, we successfully:

* Set up the development environment
* Loaded and explored the MS MARCO dataset
* Analyzed query and passage characteristics
* Identified important patterns in search queries and answers

These insights provide a strong foundation for developing a **retrieval-based question answering system**.

Future work will focus on implementing and evaluating **information retrieval models** that can effectively retrieve relevant passages for user queries.

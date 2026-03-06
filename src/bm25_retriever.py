from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:

    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, k=5):
        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_k = np.argsort(scores)[::-1][:k]

        results = []

        for idx in top_k:
            results.append({
                "score": float(scores[idx]),
                "passage": self.corpus[idx]
            })

        return results
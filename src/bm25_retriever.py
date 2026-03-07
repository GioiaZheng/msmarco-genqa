from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:
    """
    A simple BM25-based passage retriever.

    Parameters
    ----------
    corpus : list[str]
        List of passage texts used to build the BM25 index.

    Methods
    -------
    retrieve(query, k=5)
        Retrieve the top-k most relevant passages for a given query.
    """

    def __init__(self, corpus):
        # remove empty passages
        self.corpus = [doc for doc in corpus if doc and doc.strip()]

        # tokenize corpus
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]

        # build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, k=5):
        """
        Retrieve top-k passages for a given query using BM25.

        Parameters
        ----------
        query : str
            Input search query.
        k : int
            Number of passages to retrieve.

        Returns
        -------
        list[dict]
            A list of retrieved passages with scores.
        """

        if not query or not query.strip():
            return []

        # tokenize query
        tokenized_query = query.lower().split()

        # compute BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # get top-k indices
        top_k = np.argsort(scores)[::-1][:k]

        results = []

        for idx in top_k:
            results.append({
                "score": float(scores[idx]),
                "passage": self.corpus[idx]
            })

        return results
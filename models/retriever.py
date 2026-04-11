import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from config.settings import DENSE_MODEL, RERANK_MODEL

class HybridRetriever:
    def __init__(self, corpus):

        self.corpus = corpus
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.embed_model = SentenceTransformer(DENSE_MODEL)
        self.corpus_embeddings = self.embed_model.encode(
            corpus, convert_to_numpy=True
        )

        dim = self.corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.corpus_embeddings)

        self.reranker = CrossEncoder(RERANK_MODEL)

    def retrieve(self, query, top_k=3):

        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top = np.argsort(bm25_scores)[-top_k:]

        query_vec = self.embed_model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)
        dense_top = I[0]

        candidate_ids = list(set(bm25_top.tolist() + dense_top.tolist()))
        candidates = [self.corpus[i] for i in candidate_ids]

        pairs = [[query, doc] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return list(ranked)[:top_k]

"""
retrieval.py
============
Stage 4 of the pipeline: combines the vector retriever (Chroma) and the
keyword retriever (BM25) from indexing.py into one hybrid retriever using
LangChain's `EnsembleRetriever`.

WHY EnsembleRetriever:
Vector search and BM25 return scores on different scales (cosine
similarity vs. BM25's term-weighting formula) that can't be compared
directly. `EnsembleRetriever` fuses multiple retrievers with Reciprocal
Rank Fusion (RRF) internally -- for each document, it looks at the RANK it
holds in each retriever's result list (not the raw score) and combines
them, so documents ranking highly across multiple retrievers rise to the
top regardless of how differently the underlying scores are scaled. This
is the same RRF algorithm the project's original hand-rolled fusion used,
now provided directly by the framework.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

from config import TOP_K_VECTOR, ENSEMBLE_WEIGHTS
from indexing import KnowledgeBase


def build_hybrid_retriever(kb: KnowledgeBase) -> BaseRetriever:
    """
    Builds the fused vector + keyword retriever. `kb.bm25_retriever`
    already has its own `k` set (TOP_K_BM25, in indexing.py); the vector
    side is configured here via `.as_retriever()`.
    """
    vector_retriever = kb.vector_store.as_retriever(search_kwargs={"k": TOP_K_VECTOR})

    return EnsembleRetriever(
        retrievers=[vector_retriever, kb.bm25_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )


if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.load()

    retriever = build_hybrid_retriever(kb)
    results = retriever.invoke("What are the key findings in the report?")
    for r in results[:5]:
        print(f"  {r.metadata.get('chunk_id')}")

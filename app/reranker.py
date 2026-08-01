"""
reranker.py
===========
Stage 5 of the pipeline: wraps the hybrid retriever from retrieval.py
with a cross-encoder reranking step, using LangChain's
`ContextualCompressionRetriever` + `CrossEncoderReranker`.

WHY RERANK AT ALL:
The embeddings used in Chroma (bi-encoders) are fast but approximate --
they encode the query and each chunk separately, then compare vectors. A
cross-encoder is slower but far more accurate because it reads the query
and chunk together. Running a cross-encoder over an entire corpus would be
too slow, so the pattern is: retrieve many candidates cheaply with hybrid
search (stage 4), then rerank just those few candidates precisely here.

`ContextualCompressionRetriever` is LangChain's standard wrapper for this
pattern -- it calls a base retriever, then passes the results through a
"document compressor" (here, `CrossEncoderReranker`) before returning
them. The result is still just a retriever from the caller's point of
view, which is what lets generation.py plug it straight into
`create_history_aware_retriever` / `create_retrieval_chain`.
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from config import CROSS_ENCODER_MODEL_NAME, TOP_K_FINAL

_cross_encoder = None  # loaded lazily so importing this module stays fast


def _get_cross_encoder() -> HuggingFaceCrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


def build_reranking_retriever(base_retriever: BaseRetriever) -> BaseRetriever:
    """Wraps any retriever with cross-encoder reranking, keeping only the top TOP_K_FINAL results."""
    reranker = CrossEncoderReranker(model=_get_cross_encoder(), top_n=TOP_K_FINAL)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


if __name__ == "__main__":
    from indexing import KnowledgeBase
    from retrieval import build_hybrid_retriever

    kb = KnowledgeBase()
    kb.load()

    hybrid = build_hybrid_retriever(kb)
    reranking = build_reranking_retriever(hybrid)

    results = reranking.invoke("How does hybrid search improve retrieval precision?")
    for r in results:
        print(f"  {r.metadata.get('chunk_id')}")

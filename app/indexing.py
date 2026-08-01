"""
indexing.py
===========
Stage 3 of the pipeline: builds the two search indexes over the chunked
Documents from chunking.py, both via LangChain integrations:

  1. VECTOR index -- `langchain_chroma.Chroma`, backed by
     `langchain_huggingface.HuggingFaceEmbeddings` running the local
     all-MiniLM-L6-v2 model. Finds chunks semantically similar to a
     question, even without shared exact words.
  2. KEYWORD index -- `langchain_community.retrievers.BM25Retriever`, a
     classic statistical search method that finds chunks sharing exact
     words with the query. Catches things vector search can miss, like
     exact codes, names, or acronyms.

Both are built once in `KnowledgeBase.build()` and reused for every
question. retrieval.py combines them into one hybrid retriever.
"""

import pickle

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, CHROMA_COLLECTION_NAME, BM25_INDEX_PATH, EMBEDDING_MODEL_NAME, TOP_K_BM25

_embeddings = None  # loaded lazily so importing this module stays fast


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


class KnowledgeBase:
    """
    Wraps the Chroma vector store and the BM25 keyword retriever in one
    object, so the rest of the app just calls `.build()` once per upload
    and `.vector_store` / `.bm25_retriever` whenever it needs a retriever.
    """

    def __init__(self):
        self.vector_store: Chroma | None = None
        self.bm25_retriever: BM25Retriever | None = None
        self.indexed_files: list[str] = []

    def build(self, chunks: list[Document]):
        """Builds both indexes from scratch. Call whenever the document set changes."""
        if not chunks:
            raise ValueError("No chunks to index -- check that documents were uploaded and parsed correctly.")

        # --- Vector index: Chroma, wiped and rebuilt so re-running build() doesn't duplicate data ---
        self.vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        try:
            existing_ids = self.vector_store.get()["ids"]
            if existing_ids:
                self.vector_store.delete(ids=existing_ids)
        except Exception:
            pass  # empty/fresh collection -- nothing to delete
        self.vector_store.add_documents(chunks)

        # --- Keyword index: BM25Retriever, persisted to disk with pickle ---
        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=TOP_K_BM25)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25_retriever, f)

        self.indexed_files = sorted({c.metadata.get("source_file", "unknown") for c in chunks})

    def load(self):
        """Loads a previously-built knowledge base from disk (used if the app restarts mid-session)."""
        self.vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25_retriever = pickle.load(f)
        self.indexed_files = sorted({
            d.metadata.get("source_file", "unknown") for d in self.bm25_retriever.docs
        })

    def remove_source(self, source_file: str):
        """
        Removes one document's chunks from both indexes IN PLACE, without
        re-embedding anything -- this is what makes document removal fast.

        `build()` re-embeds every chunk from scratch, which is the
        expensive step (each chunk is a call into the embedding model).
        Removal doesn't need that: Chroma can delete the target file's
        vectors directly with a metadata filter, and BM25 -- being plain
        keyword statistics, not embeddings -- can be rebuilt from the
        chunks Chroma still has on hand (fetched as text, not
        re-embedded) in milliseconds. Net effect: removing a document
        costs roughly the same as a database delete plus a keyword-index
        rebuild, not a full knowledge-base rebuild.
        """
        if self.vector_store is None:
            return

        self.vector_store.delete(where={"source_file": source_file})

        remaining = self.vector_store.get(include=["documents", "metadatas"])
        remaining_chunks = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(remaining["documents"], remaining["metadatas"])
        ]

        if remaining_chunks:
            self.bm25_retriever = BM25Retriever.from_documents(remaining_chunks, k=TOP_K_BM25)
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump(self.bm25_retriever, f)
            self.indexed_files = sorted({c.metadata.get("source_file", "unknown") for c in remaining_chunks})
        else:
            self.bm25_retriever = None
            if BM25_INDEX_PATH.exists():
                BM25_INDEX_PATH.unlink()
            self.indexed_files = []


if __name__ == "__main__":
    from config import DATA_DIR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
    from ingestion import load_all_documents
    from chunking import chunk_all_documents

    docs = load_all_documents(DATA_DIR)
    chunks = chunk_all_documents(docs, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)

    kb = KnowledgeBase()
    kb.build(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(kb.indexed_files)} files.")
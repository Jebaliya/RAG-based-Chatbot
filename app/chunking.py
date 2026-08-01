"""
chunking.py
===========
Stage 2 of the pipeline: splits the LangChain Documents produced by
ingestion.py into small, overlapping chunks using LangChain's
`RecursiveCharacterTextSplitter`.

WHY RecursiveCharacterTextSplitter:
It tries a list of separators in order ("\\n\\n", "\\n", sentence-ending
punctuation, then plain spaces), splitting on the first one that produces
pieces within the size budget. In practice this keeps chunks coherent --
paragraph and sentence boundaries are preferred over mid-word/mid-sentence
cuts -- without needing a hand-rolled sentence splitter.

TOKEN-AWARE SIZING (and the bug fix):
`length_function` is set to a tiktoken-based token counter rather than
raw character count, so `CHUNK_MAX_TOKENS` in config.py means what it
says regardless of how dense a document's text is.

  ROOT CAUSE OF "disallowed special token" ERRORS:
  tiktoken's `Encoding.encode()` treats strings like "<|endofprompt|>" or
  "<|endoftext|>" as RESERVED CONTROL TOKENS by default, and raises a
  ValueError if it finds one literally inside the text being encoded --
  even though here we're just counting tokens in an uploaded document,
  not sending a prompt to a model. Any uploaded PDF/DOCX/TXT/PPTX that
  happens to literally contain one of these substrings (technical docs
  about LLMs, exported chat transcripts, prompt-engineering guides, etc.)
  would crash "Build Knowledge Base" with exactly the error you saw.

  THE FIX: pass `disallowed_special=()` to `encode()`, which tells
  tiktoken "treat every one of these strings as ordinary text, never as a
  special control token." That's correct here because we only ever use
  the encoder for length measurement -- the text is never replayed back
  through a chat completion API as a raw prompt where a real control
  token would matter.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        # disallowed_special=() is the fix: never raise on literal
        # "<|...|>"-style substrings inside arbitrary document text.
        return len(_ENCODER.encode(text, disallowed_special=()))
except Exception:
    # Falls back to a word-count approximation if tiktoken isn't installed,
    # or if its encoding file can't be downloaded (e.g. restricted network
    # egress on some hosting platforms). Chunking still works correctly --
    # chunk sizes are just approximate rather than exact token counts.
    def count_tokens(text: str) -> int:
        return int(len(text.split()) / 0.75)


def build_text_splitter(max_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    """Builds a token-aware RecursiveCharacterTextSplitter with the project's standard separators."""
    return RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


def chunk_all_documents(
    documents: list[Document],
    max_tokens: int = 300,
    overlap_tokens: int = 40,
) -> list[Document]:
    """
    Splits every loaded Document into chunk-sized Documents. Each output
    chunk keeps its parent's `source_file` / `page_number` metadata and
    gains a unique `chunk_id` (e.g. "report.pdf-p4-c2"), used throughout
    the rest of the pipeline for citation and retrieval-source tracking.
    """
    splitter = build_text_splitter(max_tokens, overlap_tokens)
    chunks = splitter.split_documents(documents)

    counters: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "unknown")
        page_number = chunk.metadata.get("page_number", 1)
        key = (source_file, page_number)
        counters[key] = counters.get(key, 0) + 1
        chunk.metadata["chunk_id"] = f"{source_file}-p{page_number}-c{counters[key]}"

    return chunks


if __name__ == "__main__":
    from config import DATA_DIR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
    from ingestion import load_all_documents

    docs = load_all_documents(DATA_DIR)
    chunks = chunk_all_documents(docs, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    print(f"Produced {len(chunks)} chunks from {len(docs)} pages")
    for c in chunks[:3]:
        print(f"--- {c.metadata['chunk_id']} ---")
        print(c.page_content[:200], "...\n")

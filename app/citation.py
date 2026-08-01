"""
citation.py
===========
Stage 7 of the pipeline, unchanged in purpose from the original project,
adapted to work with LangChain `Document` objects (the retrieved chunks
now come back as `Document`, with `chunk_id` / `source_file` /
`page_number` in `.metadata` instead of a plain dict).

Two responsibilities:

1. GROUNDING CHECK -- after the LLM generates an answer containing inline
   citation tags like "[report.pdf-p4-c2]", check_citations() verifies
   each tag actually points to a chunk that was really retrieved (not a
   fabricated citation), and flags any sentence with no citation at all
   (a likely hallucination). This is a lightweight, rule-based,
   explainable version of the idea behind attribution-scoring tools like
   ContextCite.

2. DISPLAY FORMATTING -- the raw "[chunk_id]" tags are for internal
   verification, not end users. format_citations_for_display() strips
   them out and appends a clean, human-readable source list to the END of
   each paragraph, e.g. "(Source: report.pdf, Page 5)" -- never inline in
   the middle of a sentence.
"""

import re
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class SentenceCheck:
    """
    Result of checking one sentence from the LLM's answer.

    Fields:
        sentence: the sentence text
        cited_chunk_ids: chunk IDs the sentence claims to cite
        valid_citations: which of those chunk IDs were ACTUALLY retrieved
        is_grounded: True if at least one valid citation supports this sentence
    """
    sentence: str
    cited_chunk_ids: list[str]
    valid_citations: list[str]
    is_grounded: bool


CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def check_citations(answer: str, retrieved_chunks: list[Document]) -> list[SentenceCheck]:
    """
    Goes sentence-by-sentence through the LLM's answer and checks whether
    each citation tag matches a chunk that was actually retrieved (i.e.
    present in `retrieved_chunks`, the "context" returned by the RAG chain).
    """
    valid_ids = {c.metadata["chunk_id"] for c in retrieved_chunks}
    results = []

    for sentence in _split_sentences(answer):
        cited_ids = CITATION_PATTERN.findall(sentence)
        valid_cited = [cid for cid in cited_ids if cid in valid_ids]
        results.append(SentenceCheck(
            sentence=sentence,
            cited_chunk_ids=cited_ids,
            valid_citations=valid_cited,
            is_grounded=len(valid_cited) > 0,
        ))

    return results


def grounded_ratio(checks: list[SentenceCheck]) -> float:
    """Fraction of sentences that are grounded (0.0 to 1.0)."""
    if not checks:
        return 0.0
    grounded = sum(1 for c in checks if c.is_grounded)
    return grounded / len(checks)


def format_citations_for_display(answer: str, retrieved_chunks: list[Document]) -> str:
    """
    Converts an answer containing inline "[chunk_id]" tags into clean
    reader-facing text: tags are removed from mid-sentence, and each
    paragraph gets ONE consolidated citation line appended to its end,
    e.g. "(Source: report.pdf, Page 5; notes.docx, Page 1)".
    """
    chunk_lookup = {c.metadata["chunk_id"]: c.metadata for c in retrieved_chunks}
    paragraphs = [p for p in answer.split("\n\n") if p.strip()]
    formatted_paragraphs = []

    for paragraph in paragraphs:
        cited_ids = CITATION_PATTERN.findall(paragraph)
        clean_text = CITATION_PATTERN.sub("", paragraph)
        clean_text = re.sub(r"[ \t]{2,}", " ", clean_text).strip()

        sources = []
        seen = set()
        for cid in cited_ids:
            meta = chunk_lookup.get(cid)
            if not meta:
                continue
            key = (meta["source_file"], meta["page_number"])
            if key in seen:
                continue
            seen.add(key)
            sources.append(f"{meta['source_file']}, Page {meta['page_number']}")

        if sources:
            citation_line = "(Source: " + "; ".join(sources) + ")"
            formatted_paragraphs.append(f"{clean_text}\n\n{citation_line}")
        else:
            formatted_paragraphs.append(clean_text)

    return "\n\n".join(formatted_paragraphs)


if __name__ == "__main__":
    fake_chunks = [
        Document(page_content="", metadata={"chunk_id": "report.pdf-p4-c2", "source_file": "report.pdf", "page_number": 4}),
        Document(page_content="", metadata={"chunk_id": "report.pdf-p5-c1", "source_file": "report.pdf", "page_number": 5}),
    ]
    fake_answer = (
        "BM25 is a keyword-based ranking function [report.pdf-p4-c2]. "
        "It was invented in the 1990s. "
        "Semantic search uses embeddings instead of exact word matches [report.pdf-p5-c1]."
    )
    checks = check_citations(fake_answer, fake_chunks)
    for c in checks:
        status = "GROUNDED" if c.is_grounded else "UNSUPPORTED"
        print(f"[{status}] {c.sentence}")
    print(f"\nGrounded ratio: {grounded_ratio(checks):.0%}")
    print("\n--- Display format ---")
    print(format_citations_for_display(fake_answer, fake_chunks))

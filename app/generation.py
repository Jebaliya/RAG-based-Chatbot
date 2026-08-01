"""
generation.py
=============
Stage 6 of the pipeline, and where the LangChain "retrieval chain"
pattern ties everything together. Builds the full conversational RAG
chain from three LangChain primitives:

1. `create_history_aware_retriever(llm, retriever, contextualize_prompt)`
   -- given the chat history and a new question, first asks the LLM to
   rewrite the question as a standalone query (resolving things like "what
   about that?"), THEN calls the reranking retriever from reranker.py with
   the rewritten query. On the first turn (empty history) it skips the
   rewrite and queries directly. This replaces the project's old
   hand-rolled `condense_question()` function with the framework's
   equivalent.

2. `create_stuff_documents_chain(llm, qa_prompt, document_prompt=...)` --
   "stuffs" the retrieved chunks into the prompt's `{context}` slot (via
   `document_prompt`, which renders each chunk's `chunk_id` /
   `source_file` / `page_number` metadata alongside its text) and asks the
   LLM to answer, citing chunk IDs as instructed.

3. `create_retrieval_chain(history_aware_retriever, question_answer_chain)`
   -- wires the two together: retrieve -> stuff -> generate. Invoking it
   returns both `"answer"` (the generated text) and `"context"` (the
   Documents actually used), which citation.py needs to verify grounding.

GENERATION PARAMETERS:
- temperature=0.2: low randomness -- focused, repeatable, factual answers,
  which matters for a RAG system where the goal is grounded answers, not
  creative ones.
"""

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, TEMPERATURE, MAX_TOKENS_ANSWER


class GenerationError(Exception):
    """Raised when the LLM chain fails, so the UI can show a clean error message."""


_llm = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise GenerationError(
                "GROQ_API_KEY is not set. Add it to your .env file locally, or to the "
                "app's Secrets panel on Streamlit Community Cloud."
            )
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_ANSWER,
        )
    return _llm


CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question which \
might reference context in the chat history, rewrite it as a standalone question which can \
be understood without the chat history. Do NOT answer the question, just reformulate it if \
needed, and otherwise return it unchanged."""

QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
document excerpts provided below -- never use outside knowledge, even if you know the answer.

Rules:
1. If the provided excerpts don't contain enough information to answer, say so honestly \
instead of guessing.
2. After each claim in your answer, cite the chunk ID it came from in square brackets, like \
this: [report.pdf-p4-c2].
3. Keep answers clear, well-organized, and concise.
4. Use the conversation history only to understand context (e.g. what "it" or "that" refers \
to) -- still answer strictly from the document excerpts below.

Document excerpts:
{context}"""

# Renders each retrieved chunk with its chunk_id and source before the LLM sees it --
# this is what makes "cite the chunk ID" in the system prompt actually possible, and
# what lets citation.py later verify each citation against a real, retrieved chunk.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[{chunk_id}] (source: {source_file}, page {page_number})\n{page_content}"
)


def build_rag_chain(retriever: BaseRetriever) -> Runnable:
    """
    Assembles the full conversational RAG chain from a (reranking) retriever.
    Call once per built knowledge base; the returned chain is invoked per-turn
    with `{"input": question, "chat_history": [...]}`.
    """
    llm = get_llm()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=DOCUMENT_PROMPT)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def generate_answer(rag_chain: Runnable, question: str, chat_history: list) -> dict:
    """
    Invokes the chain for one turn. Returns a dict with "answer" (raw text,
    containing inline [chunk_id] citation tags) and "context" (the list of
    Documents actually retrieved and used -- needed by citation.py).
    """
    try:
        return rag_chain.invoke({"input": question, "chat_history": chat_history})
    except Exception as exc:
        raise GenerationError(f"The language model request failed: {exc}") from exc


if __name__ == "__main__":
    from indexing import KnowledgeBase
    from retrieval import build_hybrid_retriever
    from reranker import build_reranking_retriever

    kb = KnowledgeBase()
    kb.load()

    retriever = build_reranking_retriever(build_hybrid_retriever(kb))
    chain = build_rag_chain(retriever)

    result = generate_answer(chain, "What are the main risks discussed in the document?", [])
    print(result["answer"])

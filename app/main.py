"""
main.py
=======
The Streamlit web app -- the entry point you run to launch the chatbot.
It ties together every pipeline stage, in order:

    ingestion -> chunking -> indexing -> retrieval -> reranker
    -> generation (LangChain conversational RAG chain) -> citation

This file contains no RAG logic itself. Its job is to:
  1. Let the user upload documents (PDF, DOCX, PPTX, TXT) in the browser.
  2. Build the vector + keyword indexes from whatever was uploaded, then
     assemble the LangChain retrieval chain on top of them.
  3. Run the chat loop: invoke the chain -> verify citations -> display.
  4. Present a clean, production-style chat UI.

RUN LOCALLY WITH:
    streamlit run app/main.py

DEPLOY ON STREAMLIT COMMUNITY CLOUD:
    Point the app at app/main.py, and set GROQ_API_KEY (and optionally
    PHOENIX_API_KEY / PHOENIX_COLLECTOR_ENDPOINT) in the app's Secrets panel.
"""

import os
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

# --------------------------------------------------------------------------
# SECRETS: st.secrets does NOT automatically populate os.environ, but our
# other modules (config.py) read keys via os.environ.get(...). Copy them
# over BEFORE importing any of our own modules, since tracing.py reads
# these environment variables the moment it's imported.
# --------------------------------------------------------------------------
for _key in ["GROQ_API_KEY", "PHOENIX_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT"]:
    try:
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
    except Exception:
        pass  # st.secrets raises if no secrets.toml exists at all -- fine locally with .env

from chunking import chunk_all_documents
from citation import check_citations, grounded_ratio, format_citations_for_display
from config import (
    APP_NAME, APP_TAGLINE, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, DATA_DIR,
    SUPPORTED_EXTENSIONS, CONVERSATION_HISTORY_TURNS, PHOENIX_DASHBOARD_URL,
    GROQ_API_KEY,
)
from generation import build_rag_chain, generate_answer, GenerationError
from indexing import KnowledgeBase
from ingestion import load_all_documents
from reranker import build_reranking_retriever
from retrieval import build_hybrid_retriever
from styles import CUSTOM_CSS
from tracing import start_tracing

st.set_page_config(page_title=APP_NAME, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_TYPES = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]


# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
if "kb" not in st.session_state:
    st.session_state.kb = None           # built KnowledgeBase, or None if not built yet
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None    # the assembled LangChain retrieval chain
if "messages" not in st.session_state:
    st.session_state.messages = []        # full chat transcript for display + memory
if "tracing_started" not in st.session_state:
    start_tracing()
    st.session_state.tracing_started = True


def build_knowledge_base(uploaded_files) -> tuple[KnowledgeBase, int, int, list[str]]:
    """
    Saves each uploaded file to disk, then runs the full offline pipeline:
    load -> chunk -> embed -> index, and assembles the LangChain retrieval
    chain (hybrid retrieval -> reranking -> history-aware -> generation)
    on top of the freshly built knowledge base.
    """
    for uploaded_file in uploaded_files:
        dest_path = DATA_DIR / uploaded_file.name
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    raw_docs = load_all_documents(DATA_DIR)
    chunks = chunk_all_documents(raw_docs, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)

    kb = KnowledgeBase()
    kb.build(chunks)

    parsed_files = {d.metadata["source_file"] for d in raw_docs}
    failed_files = [f.name for f in uploaded_files if f.name not in parsed_files]

    return kb, len(raw_docs), len(chunks), failed_files


def get_recent_history(max_turns: int) -> list:
    """
    Returns the last `max_turns` user/assistant turns as LangChain message
    objects, the format `MessagesPlaceholder("chat_history")` expects.
    """
    history = st.session_state.messages[-(max_turns * 2):]
    formatted = []
    for m in history:
        if m["role"] == "user":
            formatted.append(HumanMessage(content=m["content"]))
        else:
            formatted.append(AIMessage(content=m["content"]))
    return formatted


def remove_document(filename: str):
    """
    Deletes one uploaded file from disk and removes its chunks from the
    knowledge base IN PLACE (see `KnowledgeBase.remove_source` -- this
    does NOT re-embed the remaining documents, so it's fast regardless of
    knowledge base size). Starts a fresh conversation, since the
    retrievable content changed.
    """
    file_path = DATA_DIR / filename
    if file_path.exists():
        file_path.unlink()

    kb = st.session_state.kb
    if kb is not None:
        kb.remove_source(filename)
        if kb.indexed_files:
            retriever = build_reranking_retriever(build_hybrid_retriever(kb))
            st.session_state.rag_chain = build_rag_chain(retriever)
        else:
            # No documents left -- go back to the empty state.
            st.session_state.kb = None
            st.session_state.rag_chain = None

    st.session_state.messages = []
    st.rerun()


def export_chat_as_pdf() -> bytes:
    """
    Renders the full conversation as a formatted PDF using reportlab, and
    returns the raw PDF bytes (built entirely in memory -- nothing is
    written to disk on the server).
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        title=f"{APP_NAME} -- Conversation Export",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ExportTitle", parent=styles["Title"], fontSize=18, spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "ExportMeta", parent=styles["Normal"], textColor=colors.grey, fontSize=9, spaceAfter=18,
    )
    speaker_style = ParagraphStyle(
        "Speaker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10.5,
        textColor=colors.HexColor("#333333"), spaceBefore=14, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10.5, leading=15,
    )
    citation_style = ParagraphStyle(
        "Citation", parent=styles["Normal"], fontSize=9, leading=13,
        textColor=colors.HexColor("#555555"), leftIndent=10, spaceBefore=2,
    )

    def render_content(text: str) -> list:
        """Splits a message into paragraphs, rendering '(Source: ...)' lines in a distinct style."""
        flowables = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            safe = escape(para).replace("\n", "<br/>")
            style = citation_style if para.startswith("(Source:") else body_style
            flowables.append(Paragraph(safe, style))
            flowables.append(Spacer(1, 4))
        return flowables

    story = [
        Paragraph(f"{APP_NAME} -- Conversation Export", title_style),
        Paragraph(f"Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")),
    ]

    for m in st.session_state.messages:
        speaker = "You" if m["role"] == "user" else "Assistant"
        story.append(Paragraph(speaker, speaker_style))
        story.extend(render_content(m["content"]))

    doc.build(story)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown(
    f"""<div class="app-header"><h1>{APP_NAME}</h1><p>{APP_TAGLINE}</p></div>""",
    unsafe_allow_html=True,
)

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not configured. Add it to a local .env file, or to this app's "
        "Secrets panel if deployed on Streamlit Community Cloud."
    )

# --------------------------------------------------------------------------
# SIDEBAR -- exactly six sections: Upload Document, Build Knowledge Base,
# Export Chat, Clear Chat, About, Trace.
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {APP_NAME}")

    st.markdown("### Upload Document")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, PPTX, or TXT",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.session_state.kb is not None and st.session_state.kb.indexed_files:
        with st.expander(f"Indexed documents ({len(st.session_state.kb.indexed_files)})"):
            for fname in st.session_state.kb.indexed_files:
                doc_col, remove_col = st.columns([5, 1])
                doc_col.caption(fname)
                if remove_col.button("✕", key=f"remove_{fname}", help=f"Remove {fname}"):
                    with st.spinner(f"Removing {fname} and rebuilding the knowledge base..."):
                        remove_document(fname)

    st.markdown("### Build Knowledge Base")
    build_disabled = not uploaded_files
    if st.button("Build knowledge base", disabled=build_disabled, use_container_width=True, type="primary"):
        with st.spinner("Reading, chunking, and indexing your documents..."):
            try:
                kb, n_docs, n_chunks, failed = build_knowledge_base(uploaded_files)
                retriever = build_reranking_retriever(build_hybrid_retriever(kb))
                st.session_state.kb = kb
                st.session_state.rag_chain = build_rag_chain(retriever)
                st.session_state.messages = []  # new knowledge base -> fresh conversation
            except Exception as exc:
                st.error(f"Failed to build the knowledge base: {exc}")
            else:
                st.success(f"Indexed {n_docs} sections into {n_chunks} chunks from {len(kb.indexed_files)} document(s).")
                if failed:
                    st.warning(f"Could not parse: {', '.join(failed)}")

    st.markdown("### Export Chat")
    if st.session_state.messages:
        st.download_button(
            "Download conversation (PDF)",
            data=export_chat_as_pdf(),
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.caption("Nothing to export yet.")

    st.markdown("### Clear Chat")
    if st.button("Clear conversation", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### About")
    with st.expander("About this app"):
        st.markdown(
            f"""
**{APP_NAME}** answers questions grounded strictly in the documents you upload.

**Pipeline (LangChain):** document loaders → token-aware text splitter →
Chroma vector store + BM25 keyword retriever → `EnsembleRetriever` (hybrid,
RRF fusion) → `ContextualCompressionRetriever` (cross-encoder reranking) →
`create_history_aware_retriever` (follow-up question rewriting) →
`create_stuff_documents_chain` (grounded generation) → citation verification.

**Model:** Llama 3.3 70B via Groq (`langchain-groq`).
**Embeddings:** all-MiniLM-L6-v2, local (`langchain-huggingface`).
**Tracing:** Arize Phoenix, auto-instrumented via OpenInference's LangChain integration.
            """
        )

    st.markdown("### Trace")
    with st.expander("Observability"):
        st.caption("Every retrieval, reranking, and generation step is traced in Arize Phoenix.")
        st.markdown(f"[Open trace dashboard]({PHOENIX_DASHBOARD_URL})")

kb = st.session_state.kb

# --------------------------------------------------------------------------
# EMPTY STATE
# --------------------------------------------------------------------------
if kb is None or st.session_state.rag_chain is None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No knowledge base yet</h3>
            <p>Upload a document in the sidebar and click <b>Build knowledge base</b> to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------------------
# CHAT HISTORY DISPLAY
# --------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("grounded_ratio") is not None:
            st.markdown(
                f'<span class="grounding-badge">{message["grounded_ratio"]:.0%} of this answer is grounded in cited sources</span>',
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------
# MAIN CHAT LOOP
# --------------------------------------------------------------------------
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chat_history = get_recent_history(CONVERSATION_HISTORY_TURNS)[:-1]  # exclude the turn just added

        try:
            with st.spinner("Retrieving, reranking, and generating..."):
                result = generate_answer(st.session_state.rag_chain, question, chat_history)

            raw_answer = result["answer"]
            top_chunks = result["context"]

            checks = check_citations(raw_answer, top_chunks)
            ratio = grounded_ratio(checks)
            display_answer = format_citations_for_display(raw_answer, top_chunks)

            st.markdown(display_answer)
            st.markdown(
                f'<span class="grounding-badge">{ratio:.0%} of this answer is grounded in cited sources</span>',
                unsafe_allow_html=True,
            )

            with st.expander("Show retrieved sections"):
                for chunk in top_chunks:
                    meta = chunk.metadata
                    st.markdown(f"**{meta.get('source_file')}, page {meta.get('page_number')}**")
                    st.text(chunk.page_content[:300] + ("..." if len(chunk.page_content) > 300 else ""))
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": display_answer,
                "grounded_ratio": ratio,
            })
            # Rerun immediately so the sidebar (Export Chat / Clear Chat)
            # reflects this turn right away, instead of only updating on
            # the NEXT question -- Streamlit doesn't automatically re-run
            # the sidebar after messages are appended later in the script.
            st.rerun()

        except GenerationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Something went wrong while answering: {exc}")
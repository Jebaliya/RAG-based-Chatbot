# RAG Chatbot

A document-grounded chat assistant built on **LangChain**. Upload PDF, DOCX,
PPTX, or TXT files and ask questions answered strictly from their content,
with citations back to the source document and page.

## Architecture

```
Upload -> ingestion.py -> chunking.py -> indexing.py
                                              |
                                    retrieval.py (EnsembleRetriever, RRF)
                                              |
                                  reranker.py (ContextualCompressionRetriever
                                                + CrossEncoderReranker)
                                              |
                          generation.py (create_history_aware_retriever
                                          + create_stuff_documents_chain
                                          + create_retrieval_chain)
                                              |
                                     citation.py (verify + format)
```

| Stage | File | LangChain components used |
|---|---|---|
| 1. Ingestion | `app/ingestion.py` | `PDFPlumberLoader`, `Docx2txtLoader`, `TextLoader`, + a custom `BaseLoader` for PPTX |
| 2. Chunking | `app/chunking.py` | `RecursiveCharacterTextSplitter` (token-aware) |
| 3. Indexing | `app/indexing.py` | `Chroma` vector store, `HuggingFaceEmbeddings`, `BM25Retriever` |
| 4. Retrieval | `app/retrieval.py` | `EnsembleRetriever` (hybrid, Reciprocal Rank Fusion) |
| 5. Reranking | `app/reranker.py` | `ContextualCompressionRetriever` + `CrossEncoderReranker` |
| 6. Generation | `app/generation.py` | `create_history_aware_retriever`, `create_stuff_documents_chain`, `create_retrieval_chain`, `ChatGroq`, `ChatPromptTemplate` |
| 7. Citation | `app/citation.py` | Custom (LangChain has no built-in citation-grounding check) |
| Tracing | `app/tracing.py` | Arize Phoenix via `openinference-instrumentation-langchain` |
| UI | `app/main.py`, `app/styles.py` | Streamlit chat interface |

**Where LangChain was deliberately NOT used:** PPTX parsing stays on a small
custom loader (`PPTXLoader` in `ingestion.py`, built on `python-pptx`)
rather than LangChain's `UnstructuredPowerPointLoader`, which pulls in the
heavy `unstructured` package and often needs system-level dependencies
(e.g. LibreOffice) that don't belong in a lightweight Streamlit Cloud
deployment. It still implements LangChain's `BaseLoader` interface, so it
slots into the pipeline exactly like the built-in loaders.

## Fixed: "disallowed special token" crash

**Symptom:** `Build Knowledge Base` failed with:
```
Encountered text corresponding to disallowed special token '<|endofprompt|>'.
```

**Root cause:** tiktoken's `Encoding.encode()` treats strings like
`<|endofprompt|>` or `<|endoftext|>` as reserved control tokens by default,
and raises if it finds one literally inside the text being encoded — even
when that text is just an uploaded document being measured for length, not
a prompt being sent to a model. Any uploaded file that happens to contain
one of these substrings (e.g. documentation about LLMs, exported prompts,
chat transcripts) crashed chunking.

**Fix:** `app/chunking.py`'s token counter now calls
`encoder.encode(text, disallowed_special=())`, which tells tiktoken to
treat every such substring as ordinary text rather than a control token.
This is correct here because the encoder is only ever used to *measure*
chunk length — the raw text is never replayed through a chat completion
API where a real control token would matter.

## Features

- Multi-format document upload (PDF, DOCX, PPTX, TXT)
- Hybrid retrieval (semantic + keyword) fused with Reciprocal Rank Fusion
- Cross-encoder reranking for precision
- Conversation memory with automatic follow-up question rewriting
  (`create_history_aware_retriever`) — no manual condensing logic needed
- Citation verification — flags any unsupported claim in an answer
- Clean, paragraph-level source citations: `(Source: file.pdf, Page 5)`
- Export conversation to Markdown
- Full observability via Arize Phoenix, auto-tracing the entire LangChain
  chain (not just the raw LLM call)
- Graceful error handling (missing API key, parse failures, LLM errors)

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your keys (or use the provided .env)
streamlit run app/main.py
```

The app opens at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud

1. Push this project to a GitHub repository. **Do not commit `.env` or
   `.streamlit/secrets.toml`** — both are already covered by `.gitignore`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app:
   - Repository: your repo
   - Branch: `main`
   - Main file path: `app/main.py`
3. Under **Advanced settings -> Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   PHOENIX_API_KEY = "your-phoenix-api-key"
   PHOENIX_COLLECTOR_ENDPOINT = "https://app.phoenix.arize.com/s/your-space-id"
   ```
   (`PHOENIX_API_KEY` / `PHOENIX_COLLECTOR_ENDPOINT` are optional — omit
   them and tracing falls back to a local endpoint, which won't be
   reachable from the cloud, so tracing is silently skipped.)
4. Deploy. First build takes a few minutes (downloading the embedding and
   cross-encoder models from Hugging Face, plus the LangChain packages).

**Note on package versions:** `requirements.txt` pins the LangChain
packages (`langchain==0.3.27` and matching `langchain-core` /
`langchain-community` / `langchain-groq` / `langchain-chroma` /
`langchain-huggingface` / `langchain-text-splitters` versions) rather than
leaving them unpinned. LangChain's APIs move quickly between major
versions — pinning keeps the deployed app reproducible instead of silently
picking up a breaking change on a future rebuild. Bump these deliberately,
together, when you want to upgrade.

**Note on storage:** Streamlit Community Cloud's filesystem is ephemeral —
uploaded documents and the built index are lost on redeploy or app sleep.
Users rebuild the knowledge base by re-uploading files after a cold start,
exactly as the "Build Knowledge Base" flow is designed for.

## Evaluation

`eval/run_ragas.py` compares keyword-only vs. semantic-only vs.
hybrid+reranked retrieval on a test question set and produces a chart.
It's a local/offline tool with its own dependency file so it doesn't
bloat the deployed app:

```bash
pip install -r eval/requirements-eval.txt
python eval/run_ragas.py
```

Fill in `eval/test_queries.json` with real questions (and, optionally,
ground-truth relevant chunk IDs) before running.

## Testing checklist

- [ ] Upload a PDF, a DOCX, a PPTX, and a TXT file together and build the knowledge base
- [ ] Upload a document that contains a literal `<|endofprompt|>`-style string and confirm it builds without error
- [ ] Ask a question answerable from the documents — verify the answer and citation
- [ ] Ask a follow-up question using a pronoun ("what about that?") — verify it's resolved correctly
- [ ] Ask a question NOT covered by the documents — verify the app says so instead of guessing
- [ ] Check the "Show retrieved sections" panel shows plausible matches
- [ ] Export the conversation and confirm the Markdown file looks correct
- [ ] Clear the conversation and confirm the chat resets
- [ ] Remove `GROQ_API_KEY` and confirm a clean error message (not a crash)
- [ ] Confirm traces appear in the Phoenix dashboard linked in the sidebar, showing retrieval + reranking + generation as nested spans

## Project structure

```
rag-chatbot/
├── app/
│   ├── main.py          # Streamlit UI + chat loop
│   ├── styles.py         # custom dark-theme CSS
│   ├── config.py         # all settings in one place
│   ├── ingestion.py       # stage 1: LangChain document loaders
│   ├── chunking.py       # stage 2: RecursiveCharacterTextSplitter (token-aware)
│   ├── indexing.py       # stage 3: Chroma vector store + BM25Retriever
│   ├── retrieval.py      # stage 4: EnsembleRetriever (hybrid, RRF)
│   ├── reranker.py       # stage 5: ContextualCompressionRetriever + CrossEncoderReranker
│   ├── generation.py     # stage 6: LangChain conversational retrieval chain
│   ├── citation.py       # stage 7: grounding check + citation formatting
│   └── tracing.py        # Arize Phoenix instrumentation (LangChain-aware)
├── data/
│   └── uploaded_docs/     # uploaded files land here (gitignored contents)
├── eval/
│   ├── run_ragas.py
│   ├── test_queries.json
│   └── requirements-eval.txt
├── .streamlit/
│   ├── config.toml        # dark theme
│   └── secrets.toml.example
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

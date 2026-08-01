"""
config.py
=========
Centralized configuration for the RAG Chatbot. Every setting the app needs
(API keys, folder paths, chunk sizes, model names) lives here in one place.
Every other module imports from here instead of hardcoding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# APP METADATA
# ---------------------------------------------------------------------------
APP_NAME = "RAG Based Chatbot"
APP_TAGLINE = "Ask questions grounded in your own documents."

# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "uploaded_docs"     # user-uploaded source documents
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_store"    # Chroma (LangChain vector store) persistence
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "bm25_index.pkl"

CHROMA_COLLECTION_NAME = "document_chunks"

# ---------------------------------------------------------------------------
# DOCUMENT UPLOAD SETTINGS
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".pptx"]
MAX_FILE_SIZE_MB = 50
MAX_FILES_PER_UPLOAD = 20

# ---------------------------------------------------------------------------
# API KEYS
# ---------------------------------------------------------------------------
#   Locally:          set them in a .env file (see .env.example)
#   Streamlit Cloud:   set them in the app's "Secrets" panel
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# CHUNKING SETTINGS (LangChain RecursiveCharacterTextSplitter)
# ---------------------------------------------------------------------------
CHUNK_MAX_TOKENS = 300      # target chunk size, measured in tokens (see chunking.py)
CHUNK_OVERLAP_TOKENS = 40   # overlap between consecutive chunks, in tokens

# ---------------------------------------------------------------------------
# RETRIEVAL SETTINGS
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"      # local, free, no API call needed
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_VECTOR = 10   # candidates pulled from the Chroma vector retriever
TOP_K_BM25 = 10     # candidates pulled from the BM25 keyword retriever
TOP_K_FUSED = 8     # candidates kept after EnsembleRetriever fuses both lists (RRF)
TOP_K_FINAL = 4     # chunks kept after cross-encoder reranking -> sent to the LLM

# Relative weight each retriever gets in Reciprocal Rank Fusion. Equal
# weighting treats semantic and keyword matches as equally trustworthy;
# tune this if one consistently outperforms the other on your documents.
ENSEMBLE_WEIGHTS = [0.5, 0.5]

# ---------------------------------------------------------------------------
# CONVERSATION SETTINGS
# ---------------------------------------------------------------------------
# How many previous turns to keep as working memory (passed to the
# history-aware retriever and the answer chain). Bounds token usage while
# still letting the chatbot resolve follow-ups like "what about that?".
CONVERSATION_HISTORY_TURNS = 4

# ---------------------------------------------------------------------------
# GENERATION SETTINGS
# ---------------------------------------------------------------------------
TEMPERATURE = 0.2
MAX_TOKENS_ANSWER = 600

# ---------------------------------------------------------------------------
# ARIZE PHOENIX (tracing / observability)
# ---------------------------------------------------------------------------
PHOENIX_ENABLED = True
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY", "")
PHOENIX_COLLECTOR_ENDPOINT = os.environ.get(
    "PHOENIX_COLLECTOR_ENDPOINT",
    "http://localhost:6006/v1/traces",
)
PHOENIX_DASHBOARD_URL = os.environ.get("PHOENIX_DASHBOARD_URL", "")
if not PHOENIX_DASHBOARD_URL:
    if "app.phoenix.arize.com" in PHOENIX_COLLECTOR_ENDPOINT:
        PHOENIX_DASHBOARD_URL = PHOENIX_COLLECTOR_ENDPOINT.split("/v1/traces")[0]
    else:
        PHOENIX_DASHBOARD_URL = "http://localhost:6006"

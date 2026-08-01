"""
tracing.py
==========
Connects the app to Arize Phoenix so every stage of the LangChain RAG
chain (retriever calls, the condense/rewrite LLM call, the final
generation call) shows up as an inspectable trace, automatically.

Uses `openinference-instrumentation-langchain`, which patches LangChain's
callback system directly -- unlike the previous Groq-specific
instrumentor, this captures the WHOLE chain (retrieval + reranking +
generation) as nested spans, not just the raw LLM call, which is a much
more useful trace now that the pipeline is built from LangChain Runnables.

TWO WAYS THIS CAN CONNECT (controlled by config.py / secrets):
  1. Phoenix Cloud (app.phoenix.arize.com) -- if PHOENIX_API_KEY is set,
     traces go to the cloud dashboard, viewable from any browser.
  2. Local Phoenix (`phoenix serve`) -- if no API key is set, traces fall
     back to http://localhost:6006/v1/traces.
"""

import os

from config import PHOENIX_ENABLED, PHOENIX_API_KEY, PHOENIX_COLLECTOR_ENDPOINT, APP_NAME

if PHOENIX_ENABLED:
    if PHOENIX_API_KEY:
        os.environ["PHOENIX_API_KEY"] = PHOENIX_API_KEY
    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", PHOENIX_COLLECTOR_ENDPOINT)

    try:
        from phoenix.otel import register

        # auto_instrument=True activates every installed OpenInference
        # instrumentor -- here, openinference-instrumentation-langchain --
        # so the whole chain is traced with no manual instrument() call needed.
        tracer_provider = register(
            project_name=APP_NAME.lower().replace(" ", "-"),
            auto_instrument=True,
        )

        destination = "Phoenix Cloud" if PHOENIX_API_KEY else "local Phoenix (phoenix serve)"
        print(f"Phoenix tracing connected -> {destination}")
    except Exception as exc:
        # Tracing is an observability nice-to-have -- if it can't connect
        # (e.g. no network, bad key), the app should still run normally.
        print(f"[tracing] Phoenix tracing disabled -- could not connect: {exc}")


def start_tracing():
    """Kept for compatibility with main.py -- setup already ran above at import time."""
    pass

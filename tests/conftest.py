"""Shared pytest setup.

Every test runs offline. ``OPENAI_API_KEY`` is a dummy so ``load_settings()``
succeeds without a real key, and baseline ``RAG_*`` values are set explicitly
so the suite does not depend on whatever the host shell exported before the
modules were imported.
"""

from __future__ import annotations

import pytest

_BASELINE_ENV = {
    "RAG_CHUNK_SIZE": "500",
    "RAG_CHUNK_OVERLAP": "50",
    "RAG_TOP_K": "4",
    "RAG_DATA_DIR": "data",
    "RAG_VECTORSTORE_DIR": "vectorstore",
    "RAG_COLLECTION_NAME": "local_rag",
    "RAG_EMBEDDING_MODEL": "text-embedding-3-small",
    "RAG_LLM_MODEL": "gpt-4o-mini",
}


def _clear_caches() -> None:
    import rag_chain

    rag_chain._get_retriever.cache_clear()
    rag_chain._get_llm.cache_clear()
    rag_chain._get_prompt.cache_clear()
    rag_chain._get_parser.cache_clear()
    rag_chain._get_answer_chain.cache_clear()


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Give each test a deterministic, offline environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-not-real")
    for name, value in _BASELINE_ENV.items():
        monkeypatch.setenv(name, value)

    _clear_caches()
    try:
        yield
    finally:
        _clear_caches()

"""Central configuration for the local RAG pipeline.

``load_settings()`` is the runtime source of truth: it reads the environment
(and ``.env``) on every call, validates the values and returns an immutable
``Settings`` object. The module-level constants below remain part of the public
surface and act as import-time defaults, but runtime code (ingestion and query
chain) reads every value through ``load_settings()`` so both sides cannot drift
apart when an environment variable overrides a default.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

# Load .env once, early, so the module-level constants below honor it as well.
# load_dotenv() is idempotent: values already present in the environment win.
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATA_DIR: Final[Path] = _env_path("RAG_DATA_DIR", PROJECT_ROOT / "data")
VECTORSTORE_DIR: Final[Path] = _env_path("RAG_VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore")

COLLECTION_NAME: Final[str] = _env_str("RAG_COLLECTION_NAME", "local_rag")

EMBEDDING_MODEL: Final[str] = _env_str("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: Final[str] = _env_str("RAG_LLM_MODEL", "gpt-4o-mini")

# Chunk sizes are measured in tiktoken tokens, not characters, so chunk_size is
# comparable with the context window budget. The assignment requires at least
# 500 tokens per chunk with 50 tokens of overlap.
CHUNK_SIZE: Final[int] = _env_int("RAG_CHUNK_SIZE", 500)
CHUNK_OVERLAP: Final[int] = _env_int("RAG_CHUNK_OVERLAP", 50)

# 4 chunks of ~500 tokens (~2000 tokens of context) keeps the prompt focused.
# Larger k pushes the relevant fragment into the middle of a long context, where
# models attend to it less reliably: the "Lost in the Middle" effect. Hard rule:
# keep top_k between 3 and 5; never inflate context to "cover" bad retrieval.
TOP_K: Final[int] = _env_int("RAG_TOP_K", 4)

# Determinism and groundedness: the same corpus + prompt should produce the same
# answer without creative variation.
TEMPERATURE: Final[float] = 0.0

# Sentinel the LLM must emit when the answer is not present in the context.
NO_ANSWER_SENTINEL: Final[str] = "NO_LO_SE"

_PLACEHOLDER_KEYS: Final[frozenset[str]] = frozenset(
    {"tu_api_key_aqui", "your_api_key_here", "changeme", "sk-..."}
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved runtime settings.

    The API key is excluded from ``repr`` so it never leaks into logs.
    """

    project_root: Path
    data_dir: Path
    vectorstore_dir: Path
    collection_name: str
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    openai_api_key: str = field(repr=False)


def load_settings() -> Settings:
    """Load .env and return validated settings.

    Raises:
        RuntimeError: when ``OPENAI_API_KEY`` is missing or still holds the
            placeholder value from ``.env.example``.
        ValueError: when retrieval or chunking parameters are out of range.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key in _PLACEHOLDER_KEYS:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Copy .env.example to .env and set "
            "OPENAI_API_KEY to a real key, or export it in your shell before "
            "running the scripts."
        )

    chunk_size = _env_int("RAG_CHUNK_SIZE", CHUNK_SIZE)
    chunk_overlap = _env_int("RAG_CHUNK_OVERLAP", CHUNK_OVERLAP)
    top_k = _env_int("RAG_TOP_K", TOP_K)

    if chunk_size < 500:
        raise ValueError(
            f"RAG_CHUNK_SIZE must be >= 500 tokens (got {chunk_size}); the "
            "assignment requires at least 500 tokens per chunk."
        )
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"RAG_CHUNK_OVERLAP must satisfy 0 <= overlap < chunk_size "
            f"(got overlap={chunk_overlap}, chunk_size={chunk_size})."
        )
    if not 3 <= top_k <= 5:
        raise ValueError(
            f"RAG_TOP_K must be between 3 and 5 (got {top_k}) to avoid the "
            "Lost in the Middle effect and an inflated context."
        )

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=_env_path("RAG_DATA_DIR", DATA_DIR),
        vectorstore_dir=_env_path("RAG_VECTORSTORE_DIR", VECTORSTORE_DIR),
        collection_name=_env_str("RAG_COLLECTION_NAME", COLLECTION_NAME),
        embedding_model=_env_str("RAG_EMBEDDING_MODEL", EMBEDDING_MODEL),
        llm_model=_env_str("RAG_LLM_MODEL", LLM_MODEL),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        temperature=TEMPERATURE,
        openai_api_key=api_key,
    )


_embeddings: "OpenAIEmbeddings | None" = None
_embeddings_lock = threading.Lock()


def get_embeddings() -> "OpenAIEmbeddings":
    """Return the process-wide OpenAI embeddings client.

    SINGLE SOURCE OF TRUTH: both ``ingest.py`` and ``rag_chain.py`` must call
    this factory. Constructing a second embeddings object elsewhere is what
    causes the classic "embeddings no coincidentes" bug: documents and queries
    end up in different vector spaces and similarity results become meaningless
    without raising an exception.

    The client is built lazily so importing this module never requires an API
    key, and cached so every caller reuses the same instance.
    """
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                from langchain_openai import OpenAIEmbeddings

                settings = load_settings()
                _embeddings = OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key,
                )
    return _embeddings


__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "COLLECTION_NAME",
    "DATA_DIR",
    "EMBEDDING_MODEL",
    "LLM_MODEL",
    "NO_ANSWER_SENTINEL",
    "PROJECT_ROOT",
    "TEMPERATURE",
    "TOP_K",
    "VECTORSTORE_DIR",
    "Settings",
    "get_embeddings",
    "load_settings",
]

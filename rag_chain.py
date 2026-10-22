"""Async LCEL RAG chain with a groundedness guard and Pydantic output.

Flow: query -> top-k similarity search in ChromaDB -> formatted context ->
ChatPromptTemplate -> ChatOpenAI -> PydanticOutputParser -> RagAnswer.

The Pydantic model is not only a parser target: it is the contract that forces
the model to expose whether it answered from the context or refused with the
``NO_LO_SE`` sentinel.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import openai
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import NO_ANSWER_SENTINEL, TEMPERATURE, TOP_K, load_settings
from ingest import get_vectorstore


class RagAnswer(BaseModel):
    answer: str = Field(
        description=(
            "Answer synthesized only from the provided context. Use exactly "
            f"'{NO_ANSWER_SENTINEL}' when the context does not contain the answer."
        )
    )
    sources: list[str] = Field(
        default_factory=list,
        description="File names from the context that support the answer. Empty when there is no answer.",
    )
    grounded: bool = Field(
        description=(
            "True when the answer comes from the provided context; False when it "
            "is the no-answer sentinel."
        )
    )


_SYSTEM_PROMPT = """Eres un asistente técnico especializado en pipelines RAG (Retrieval-Augmented Generation).

Reglas obligatorias:
1. Responde únicamente con información incluida en el CONTEXTO.
2. Si el CONTEXTO no contiene la respuesta, escribe exactamente la cadena "NO_LO_SE" (cinco letras mayúsculas separadas por guiones bajos) en el campo "answer", deja "sources" como lista vacía y marca "grounded" como false. No inventes, no supongas y no uses conocimiento externo. La cadena "NO_LO_SE" es la única respuesta válida cuando falta la información.
3. Si respondes desde el CONTEXTO, marca "grounded" como true e incluye en "sources" los nombres de los archivos citados.
4. Responde en español neutro, de forma clara y concisa.
5. Devuelve solo el objeto JSON solicitado, sin texto adicional.

CONTEXTO:
{context}

{format_instructions}"""

_RETRY_INSTRUCTION = (
    "Tu respuesta anterior no pudo interpretarse como JSON válido. "
    "Corrige el error indicado y devuelve únicamente un objeto JSON que cumpla "
    "el esquema, sin explicaciones ni bloques de código."
)


@lru_cache(maxsize=1)
def _get_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=RagAnswer)


@lru_cache(maxsize=1)
def _get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "PREGUNTA:\n{question}"),
        ]
    ).partial(
        # langchain-core 1.x exposes this as a method, not a property.
        format_instructions=_get_parser().get_format_instructions(),
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    settings = load_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=TEMPERATURE,
        api_key=settings.openai_api_key,
    )


def _ensure_vectorstore_ready(vectorstore: Chroma) -> None:
    """Fail clearly before querying an empty or incompatible collection."""
    if vectorstore._collection.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Run `python ingest.py` before querying."
        )

    stored_model = (vectorstore._collection.metadata or {}).get("embedding_model")
    current_model = load_settings().embedding_model
    if stored_model and stored_model != current_model:
        raise RuntimeError(
            f"Embedding model mismatch: the index was built with '{stored_model}' "
            f"but the current configuration uses '{current_model}'. Run "
            "`python ingest.py --force` to rebuild the index."
        )


@lru_cache(maxsize=1)
def _get_retriever():
    vectorstore = get_vectorstore()
    _ensure_vectorstore_ready(vectorstore)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def format_documents(documents: Sequence[Document]) -> str:
    """Render retrieved chunks as context blocks that name their source file."""
    blocks: list[str] = []
    for document in documents:
        source = Path(str(document.metadata.get("source", "unknown"))).name
        blocks.append(f"[Fuente: {source}]\n{document.page_content.strip()}")
    return "\n\n".join(blocks)


def _deduplicate_documents(
    documents: Sequence[Document],
    limit: int = TOP_K,
) -> list[Document]:
    """Keep at most ``limit`` unique chunks without inflating the context."""
    seen: set[tuple[str, object, str]] = set()
    unique: list[Document] = []

    for document in documents:
        key = (
            str(document.metadata.get("source", "")),
            document.metadata.get("chunk_index"),
            document.page_content[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
        if len(unique) >= limit:
            break

    return unique


async def retrieve_documents(query: str) -> list[Document]:
    """Run the similarity search and return deduplicated top-k chunks."""
    documents = await _get_retriever().ainvoke(query)
    return _deduplicate_documents(documents)


@lru_cache(maxsize=1)
def _get_answer_chain() -> Runnable:
    """LCEL chain: inputs -> parallel context/question -> prompt -> LLM -> parser."""
    return (
        RunnableParallel(
            context=RunnableLambda(
                lambda inputs: format_documents(
                    _deduplicate_documents(inputs["documents"])
                )
            ),
            question=RunnableLambda(lambda inputs: inputs["question"]),
        )
        | _get_prompt()
        | _get_llm()
        | _get_parser()
    )


def _normalize_answer(answer: RagAnswer) -> RagAnswer:
    """Enforce the sentinel contract even if the model contradicts itself."""
    cleaned_sources = list(
        dict.fromkeys(source.strip() for source in answer.sources if source.strip())
    )
    if answer.answer.strip() == NO_ANSWER_SENTINEL:
        return RagAnswer(
            answer=NO_ANSWER_SENTINEL,
            sources=[],
            grounded=False,
        )
    return RagAnswer(
        answer=answer.answer.strip(),
        sources=cleaned_sources,
        grounded=answer.grounded,
    )


async def _retry_after_parser_error(
    inputs: dict,
    error: OutputParserException,
) -> RagAnswer:
    """One corrective retry: append the parse error to the rendered prompt.

    If the retry also fails to parse, fall back to the safe sentinel answer
    instead of returning a malformed or fabricated result.
    """
    prompt_inputs = {
        "context": format_documents(
            _deduplicate_documents(inputs["documents"])
        ),
        "question": inputs["question"],
    }
    messages = _get_prompt().format_messages(**prompt_inputs)

    if error.llm_output:
        messages.append(AIMessage(content=error.llm_output))
    messages.append(
        HumanMessage(content=f"{_RETRY_INSTRUCTION}\n\nError: {error}")
    )

    response = await _get_llm().ainvoke(messages)
    try:
        return _normalize_answer(_get_parser().invoke(response))
    except OutputParserException:
        return RagAnswer(answer=NO_ANSWER_SENTINEL, sources=[], grounded=False)


async def _answer_with_documents(query: str, documents: Sequence[Document]) -> RagAnswer:
    inputs = {"question": query, "documents": list(documents)}
    try:
        answer = await _get_answer_chain().ainvoke(inputs)
    except OutputParserException as exc:
        return await _retry_after_parser_error(inputs, exc)
    return _normalize_answer(answer)


async def get_rag_response(query: str) -> RagAnswer:
    """Answer ``query`` from the indexed corpus using the async LCEL chain.

    Steps: (a) similarity search in ChromaDB, (b) build the grounded prompt with
    the retrieved fragments, (c) call the LLM asynchronously, (d) parse the
    response into ``RagAnswer``.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        raise ValueError("Query must be a non-empty string.")

    documents = await retrieve_documents(clean_query)
    return await _answer_with_documents(clean_query, documents)


async def get_rag_response_with_context(
    query: str,
) -> tuple[RagAnswer, list[Document]]:
    """Same as ``get_rag_response`` but also returns the chunks used."""
    clean_query = (query or "").strip()
    if not clean_query:
        raise ValueError("Query must be a non-empty string.")

    documents = await retrieve_documents(clean_query)
    answer = await _answer_with_documents(clean_query, documents)
    return answer, documents


async def _retrieve_with_scores(query: str) -> list[tuple[Document, float]]:
    """Fetch chunks with their Chroma distance, for demo diagnostics only."""
    vectorstore = get_vectorstore()
    _ensure_vectorstore_ready(vectorstore)
    return await vectorstore.asimilarity_search_with_score(query, k=TOP_K)


async def _run_demo() -> None:
    questions = [
        "¿Cuál es el error número uno en un pipeline RAG y cómo se evita?",
        "¿Cuál es la capital de Australia y en qué año se fundó Canberra?",
    ]

    for index, question in enumerate(questions, start=1):
        print("=" * 74)
        print(f"PREGUNTA {index}: {question}")
        print("-" * 74)

        answer = await get_rag_response(question)
        scored_chunks = await _retrieve_with_scores(question)

        print(f"RESPUESTA:\n{answer.answer}\n")
        print(f"FUENTES CITADAS: {', '.join(answer.sources) if answer.sources else '(ninguna)'}")
        print(f"GROUNDED: {answer.grounded}")
        print("FRAGMENTOS RECUPERADOS (distancia; menor = más similar):")
        for document, score in scored_chunks:
            source = Path(str(document.metadata.get("source", "unknown"))).name
            chunk_index = document.metadata.get("chunk_index", "?")
            print(f"  - {score:.4f}  {source}  (chunk {chunk_index})")
        print()

    print("=" * 74)


async def _demo_entrypoint() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        await _run_demo()
    except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
        print(
            "ERROR: OpenAI rejected the configured API key. Check OPENAI_API_KEY "
            f"in your .env file. Details: {exc}"
        )
    except openai.APIError as exc:
        print(f"ERROR: OpenAI API request failed: {exc}")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")


if __name__ == "__main__":
    asyncio.run(_demo_entrypoint())

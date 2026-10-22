# Local RAG over Spanish Technical Notes

A small, production-minded RAG pipeline that indexes four Spanish markdown
documents about RAG engineering and answers questions **only** from that corpus.
When the answer is not in the corpus, the system refuses with the explicit
`NO_LO_SE` sentinel instead of hallucinating.

## What it is

- **Ingestion** (`ingest.py`): loads `.md`/`.txt`, splits by tokens, embeds with
  OpenAI, and persists to ChromaDB. Idempotent: it skips re-indexing when the
  collection already has vectors.
- **Querying** (`rag_chain.py`): async LCEL chain that retrieves the top 4
  chunks, builds a grounded Spanish prompt, calls the LLM, and parses a
  Pydantic `RagAnswer` (`answer`, `sources`, `grounded`).
- **Configuration** (`config.py`): one source of truth for models, chunking,
  paths, `top_k`, and the embeddings client.

## Architecture

```
data/*.md, *.txt
      |
      v
TextLoader (UTF-8) -> RecursiveCharacterTextSplitter (tiktoken, 500/50)
      |
      v
OpenAIEmbeddings (text-embedding-3-small)  <-- config.get_embeddings()
      |
      v
ChromaDB PersistentClient (./vectorstore, collection: local_rag)
      |
      v  top_k = 4 similarity search
Retriever -> formatted context ([Fuente: file.md] + chunk)
      |
      v
LCEL: RunnableParallel(context, question) | ChatPromptTemplate
      | ChatOpenAI(gpt-4o-mini, temperature=0)
      | PydanticOutputParser
      |
      v
RagAnswer { answer, sources, grounded }
```

## Requirements

- CPython 3.12
- An OpenAI API key
- Dependencies pinned in `requirements.txt` (`langchain`, `langchain-openai`,
  `langchain-chroma`, `chromadb`, `tiktoken`, `python-dotenv`, ...)

## Setup

```powershell
# from the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# edit .env and set OPENAI_API_KEY
```

On macOS/Linux the equivalent is `python3 -m venv .venv`,
`source .venv/bin/activate`, `cp .env.example .env`.

## Ingest usage

```powershell
python ingest.py           # indexes only if the collection is empty
python ingest.py --force   # deletes the collection and rebuilds it
```

The script prints the number of files, chunks, target directory, and embedding
model. Re-running it without `--force` is cheap: it detects the persisted
collection with `count() > 0` and skips the work (and the API cost).

Run ingestion again with `--force` whenever you change the corpus, the chunking
parameters, or the embedding model.

## Query usage

```powershell
python rag_chain.py
```

Or from Python:

```python
import asyncio
from rag_chain import get_rag_response

answer = asyncio.run(get_rag_response("¿Qué es el chunking recursivo?"))
print(answer.answer, answer.sources, answer.grounded)
```

The public API is exactly:

```python
async def get_rag_response(query: str) -> RagAnswer
```

## The two required tests

Running `python rag_chain.py` executes both cases and prints the answer, cited
sources, `grounded`, and the retrieved chunks with their distances.

| # | Question | Expected result |
|---|----------|-----------------|
| 1 | `¿Cuál es el error número uno en un pipeline RAG y cómo se evita?` | Grounded answer: mismatched embeddings between ingestion and query; fix with a single embeddings source of truth and full re-index. `grounded=True`. |
| 2 | `¿Cuál es la capital de Australia y en qué año se fundó Canberra?` | `answer=NO_LO_SE`, `sources=[]`, `grounded=False` (the corpus does not cover it). |

## Configuration

| Setting | Default | Why |
|---------|---------|-----|
| `RAG_CHUNK_SIZE` | `500` tokens | Large enough for a complete idea, small enough to keep the embedding specific. Measured with tiktoken, not characters. The splitter window is `500 + 50 = 550` tokens (overlap is counted inside the window) so chunks actually approach the required ~500-token size. |
| `RAG_CHUNK_OVERLAP` | `50` tokens | Preserves context across boundaries without duplicating too much. |
| `RAG_TOP_K` | `4` | Within the recommended 3-5 range. More chunks push the relevant one into the middle of a long context, where attention degrades (Lost in the Middle) and token cost grows. |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Good quality/cost tradeoff; 1536 dimensions. |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Fast and inexpensive for grounded extraction/synthesis. |
| `RAG_COLLECTION_NAME` | `local_rag` | Stable Chroma collection name shared by both scripts. |
| `RAG_DATA_DIR` | `./data` | Corpus location. |
| `RAG_VECTORSTORE_DIR` | `./vectorstore` | Persistent Chroma directory (git-ignored). |
| `TEMPERATURE` | `0.0` | Deterministic, grounded answers. |

## Design decisions and common errors avoided

- **Embeddings mismatch**: `config.get_embeddings()` is the only embeddings
  factory, used by both ingestion and query. It returns a cached singleton, so
  documents and queries always live in the same vector space. The index also
  stores the embedding model name in collection metadata and the query path
  refuses to run against a mismatched index.
- **Infinite context**: `top_k` is capped at 4 (validated to 3-5) and retrieved
  chunks are deduplicated before formatting. The prompt never grows with the
  corpus size.
- **Missing persistence**: ingestion checks the persistent collection before
  doing any work, so reruns are idempotent. langchain-chroma 1.x persists
  automatically; there is no `.persist()` call (it was removed).
- **Lost in the Middle**: few, high-quality chunks beat many mediocre ones; a
  focused context keeps the relevant fragment near the edges of the prompt.
- **Secrets**: the API key is read from `.env`/environment only, excluded from
  settings `repr`, and `.env` is git-ignored.
- **Hallucination on out-of-domain questions**: the Spanish system prompt
  requires the `NO_LO_SE` sentinel when the context does not answer, and code
  normalizes the result so a sentinel answer is always `grounded=False` with no
  sources.
- **Parser failures**: one corrective retry appends the parse error to the
  prompt; if it fails again, the chain returns the safe sentinel answer.

## Project structure

```
preentrega3-rag/
├── config.py            # settings, paths, embeddings singleton
├── ingest.py            # loaders, token splitter, Chroma indexing, CLI
├── rag_chain.py         # Pydantic model, Spanish prompt, LCEL chain, demo
├── data/                # 4 Spanish markdown documents
├── vectorstore/         # generated by ingest.py (git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `OPENAI_API_KEY is not configured` | Copy `.env.example` to `.env` and set a real key; the placeholder is rejected. |
| `The vector store is empty` | Run `python ingest.py` before querying. |
| `Embedding model mismatch` | The index was built with another model. Run `python ingest.py --force`. |
| `Indexing skipped` | The collection already has vectors. Use `--force` to rebuild. |
| Answers are irrelevant | The index is stale or was built with a different embedding model; rebuild with `--force`. |
| `openai.AuthenticationError` | Invalid/revoked key or no billing access; check the OpenAI dashboard. |
| Accents look broken in a terminal | The files and loaders use UTF-8; set the terminal to UTF-8 (`chcp 65001` on Windows). |

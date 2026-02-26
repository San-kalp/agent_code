# RAG with Metadata Filtering

A RAG (Retrieval-Augmented Generation) project that demonstrates **metadata-aware retrieval** using LangChain's Self-Query Retriever. Unlike basic RAG which only does semantic similarity search, this project can automatically filter documents by metadata fields like year, topic, and subtopic.

## Project Structure

| File               | Description                                                    |
| ------------------ | -------------------------------------------------------------- |
| `llm.py`           | LLM configuration using `ChatOpenAI`                           |
| `config.json`      | API keys and base URL configuration                            |
| `document.py`      | Sample documents with metadata (renewable energy dataset)      |
| `vector_store.py`  | Chroma vector store — stores and embeds documents              |
| `retriever.py`     | Self-Query Retriever setup with metadata field definitions     |

## Setup

### 1. Install dependencies

```bash
pip install langchain langchain-openai langchain-community langchain-classic chromadb lark
```

> `lark` is required by the Self-Query Retriever to parse filter expressions.

### 2. Configure API keys

Edit `config.json`:

```json
{
  "API_KEY": "your-api-key-here",
  "OPENAI_BASE_URL": "https://your-api-endpoint.com/v1"
}
```

## How it works

### LLM Setup (`llm.py`)

Uses `ChatOpenAI` from langchain-openai (not smolagents' `OpenAIServerModel` like the Research Assistant project).

```python
from llm import llm, api_key, base_url
```

- `llm` — a `ChatOpenAI` instance with `temperature=0` for deterministic outputs
- `api_key`, `base_url` — extracted from `config.json`, reusable in other files

#### Key difference from Research Assistant

| | Research Assistant | RAG with Metadata |
|---|---|---|
| **LLM class** | `OpenAIServerModel` (smolagents) | `ChatOpenAI` (langchain) |
| **Usage** | Agent reasoning loop | Chain/retriever pipelines |

### Documents (`document.py`)

8 hardcoded sample documents about renewable energy, each with **metadata**:

```python
Document(
    id=1,
    page_content="Renewable energy is derived from natural processes...",
    metadata={"year": 2023, "topics": "introduction", "subtopic": "renewable energy"}
)
```

#### Metadata fields

| Field      | Type    | Example values                                         |
| ---------- | ------- | ------------------------------------------------------ |
| `year`     | integer | 2023, 2024, 2025                                      |
| `topics`   | string  | "solar power", "wind energy", "hydroelectric", "biomass" |
| `subtopic` | string  | "renewable energy", "policy" (not present on all docs) |

These metadata fields are what enable filtered retrieval — without them, you can only do semantic search.

### Vector Store (`vector_store.py`)

Stores documents in a **persistent** Chroma vector store with embeddings.

```python
from vector_store import load_vector_store

vs = load_vector_store()
```

- Uses `OpenAIEmbeddings` with `text-embedding-ada-002` model to convert text to vectors
- `persist_directory="chroma_db"` saves the store to disk
- `add_documents()` is used (two-step: create store, then add docs) instead of `from_documents()` (one-step)

### Self-Query Retriever (`retriever.py`)

The core of this project. Defines **metadata field descriptions** so the LLM can automatically extract filters from natural language queries.

#### `AttributeInfo`

Tells the LLM what metadata fields exist and what they mean:

```python
metadata_field_info = [
    AttributeInfo(name="year", description="The year the document was created or published", type="integer"),
    AttributeInfo(name="topics", description="The main topic of the document", type="string"),
    AttributeInfo(name="subtopic", description="A more specific subcategory of the main topic.", type="string"),
]
```

The LLM reads these descriptions to decide **which fields to filter on** based on the user's question.

#### `document_content_description`

A high-level description of what the documents contain. Helps the LLM understand the domain:

```python
document_content_description = "Brief overview of various aspects related to Renewable Energy..."
```

#### How Self-Query Retriever works

With a **normal retriever**, the query `"What solar energy documents were published after 2023?"` would just do semantic similarity on the entire string — imprecise.

With the **Self-Query Retriever**, the LLM parses it into two parts:

```
Semantic query: "solar energy"        --> similarity search on page_content
Metadata filter: year > 2023          --> exact filter on metadata
```

This separation gives much more accurate results because it combines:
1. **Semantic search** — finds documents with similar meaning
2. **Structured filtering** — narrows results by exact metadata constraints

#### Full usage

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from vector_store import load_vector_store
from retriever import metadata_field_info, document_content_description
from llm import llm

vs = load_vector_store()

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vs,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)

# The LLM auto-extracts filters from natural language
results = retriever.invoke("What solar documents were published after 2023?")
```

## Comparison: Basic RAG vs Metadata RAG

| Feature | Basic RAG (Research Assistant) | Metadata RAG (this project) |
|---|---|---|
| **Search method** | Semantic similarity only | Semantic + metadata filtering |
| **Query parsing** | Entire query used for similarity | LLM splits query into search + filters |
| **Documents** | Real PDFs from files | Hardcoded samples with metadata |
| **Retriever** | `similarity_search(query, k=3)` | `SelfQueryRetriever.from_llm(...)` |
| **Best for** | "What is agentic AI?" | "What solar docs were published after 2023?" |
| **Requires** | Just embeddings | Embeddings + LLM (for query parsing) + `lark` |

## Key concepts

- **`AttributeInfo`** — describes a metadata field (name, type, description) so the LLM knows it can filter on it
- **`SelfQueryRetriever`** — a retriever that uses an LLM to parse queries into semantic search + metadata filters
- **`document_content_description`** — tells the LLM what the document collection is about
- **`ChatOpenAI` vs `OpenAIServerModel`** — both are LLM clients; `ChatOpenAI` is LangChain's, `OpenAIServerModel` is smolagents'
- **`lark`** — parser library used internally by the Self-Query Retriever to parse filter expressions

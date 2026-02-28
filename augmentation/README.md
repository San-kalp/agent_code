# Augmentation — Self-Query Retrieval + Cross-Encoder Reranking

## What this module does

A two-stage RAG retrieval pipeline over a small renewable energy document corpus:

1. **Self-Query Retrieval** — LLM parses the natural language query into a semantic search + structured metadata filter, then fetches candidates from ChromaDB.
2. **Cross-Encoder Reranking** — A cross-encoder model scores each (query, document) pair together and returns only the top-N most relevant results.

---

## Pipeline at a Glance

```
User Query
    │
    ▼
SelfQueryRetriever          ← LLM converts query into vector search + metadata filter
    │
    ▼
ChromaDB (vector store)     ← Returns candidate documents
    │
    ▼
HuggingFaceCrossEncoder     ← Scores each (query, doc) pair together
    │
    ▼
CrossEncoderReranker        ← Sorts by score, keeps top_n=5
    │
    ▼
Final Ranked Documents
```

---

## Key Concepts

### Self-Query Retriever
- Uses the LLM (`gpt-4o-mini`) to **parse the query** into two parts:
  - A semantic search string (vector similarity)
  - A metadata filter (e.g., `year == 2024`, `topics == "hydroelectric"`)
- This lets you answer questions like *"find hydroelectric documents from 2024"* without keyword matching.

### Cross-Encoder vs Bi-Encoder (Embeddings)
| | Bi-Encoder (embeddings) | Cross-Encoder |
|---|---|---|
| How it works | Encodes query and doc **separately**, compares vectors | Reads query + doc **together** in one pass |
| Speed | Fast | Slow |
| Accuracy | Lower | Higher |
| Use in pipeline | Stage 1: retrieve candidates | Stage 2: rerank candidates |

### CrossEncoderReranker
```python
reranker = CrossEncoderReranker(model=crossencoder, top_n=5)
reranked_docs = reranker.compress_documents(documents, query)
```
- Takes the rough retrieval results and re-scores them
- Returns the top `n` most relevant documents sorted by cross-encoder score

### ContextualCompressionRetriever
```python
reranker_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever_with_metadata
)
reranked_result = reranker_retriever.invoke(question)
for i, doc in enumerate(reranked_result):
    print(doc.page_content)
    print(doc.metadata)
```
- A **pipeline wrapper** that chains a retriever with a compressor/reranker into one object
- `base_retriever` — fetches candidates (SelfQueryRetriever here)
- `base_compressor` — filters/reranks those candidates (CrossEncoderReranker here)
- Calling `.invoke()` runs **both steps automatically** in a single call — no manual two-step needed
- Returns documents sorted by relevance score, best match first
- Makes the full pipeline pluggable — drop `reranker_retriever` directly into any RAG chain

---

## File Structure

| File | Purpose |
|------|---------|
| `document.py` | 8 hardcoded `Document` objects with metadata (`year`, `topics`, `subtopic`) |
| `vector_store.py` | Creates/loads ChromaDB collection, embeds documents using `text-embedding-ada-002` |
| `retriever.py` | Builds `SelfQueryRetriever` with metadata field info describing the corpus |
| `llm.py` | Loads `ChatOpenAI` (gpt-4o-mini) from `config.json` |
| `main.py` | Runs the full pipeline: retrieve → score → rerank |
| `config.json` | API key and base URL (git-ignored) |

---

## Document Corpus (document.py)

8 documents about renewable energy, each with metadata:

| ID | Topic | Subtopic | Year |
|----|-------|----------|------|
| 1 | introduction | renewable energy | 2023 |
| 2 | solar power | — | 2023 |
| 3 | wind energy | — | 2023 |
| 4 | hydroelectric | — | 2024 |
| 5 | geothermal | — | 2024 |
| 6 | biomass | — | 2025 |
| 7 | energy storage | — | 2025 |
| 8 | environment | policy | 2025 |

---

## Metadata Fields (retriever.py)

The LLM knows about these fields when building filters:

| Field | Type | Description |
|-------|------|-------------|
| `year` | integer | Year published |
| `topics` | string | Main topic |
| `subtopic` | string | Specific subcategory |

---

## How to Run

```bash
cd augmentation
python main.py
```

Sample query hardcoded: `"How is Hydroelectric power used ?"`

---

## Dependencies

- `langchain-community` — HuggingFaceCrossEncoder
- `langchain-classic` — SelfQueryRetriever, CrossEncoderReranker, ContextualCompressionRetriever
- `langchain-openai` — ChatOpenAI, OpenAIEmbeddings
- `chromadb` — vector store
- `sentence-transformers` — required by HuggingFaceCrossEncoder
- Model used: `cross-encoder/ms-marco-MiniLM-L-6-v2` (downloaded from HuggingFace on first run)

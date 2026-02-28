# Contextual Compression Retrieval

## Idea

Standard RAG pipelines retrieve whole chunks from a vector store and pass them verbatim to the LLM. This is wasteful: a retrieved chunk often contains only a sentence or two that actually answers the question, surrounded by irrelevant context. Feeding the noise to the LLM increases token cost, dilutes the signal, and can hurt answer quality.

**Contextual Compression** fixes this by adding a post-retrieval compression step. After documents are fetched from the vector store, an LLM reads each one and extracts only the parts that are relevant to the user's query. The LLM acts as an intelligent filter — the final context passed downstream is lean and query-focused.

```
User Query
    │
    ▼
┌─────────────────────────────┐
│   Self-Query Retriever      │  ← parses query into semantic search + metadata filters
│   (base retriever)          │
└─────────────┬───────────────┘
              │  raw chunks (full text)
              ▼
┌─────────────────────────────┐
│   LLMChainExtractor         │  ← LLM reads each chunk, keeps only relevant sentences
│   (compressor)              │
└─────────────┬───────────────┘
              │  compressed, query-relevant snippets
              ▼
         Final Answer
```

## Why It Matters

| Without Compression | With Compression |
|---|---|
| Whole chunks forwarded | Only relevant sentences forwarded |
| More tokens consumed | Fewer tokens consumed |
| Noise can confuse the LLM | Clean, focused context |
| Works for generic queries | Works well even with specific, narrow queries |

## How This Implementation Works

### Base Retriever — `SelfQueryRetriever`

[retriever.py](retriever.py) builds a `SelfQueryRetriever` on top of a ChromaDB vector store. It understands three metadata fields:

- `year` — publication year of the document
- `topics` — main topic (e.g. `"wind energy"`, `"solar power"`)
- `subtopic` — more specific subcategory

When a user asks something like *"What wind energy documents from 2023 exist?"*, the self-query retriever automatically translates that into both a vector similarity search and a metadata filter (`year == 2023`, `topics == "wind energy"`), without requiring the user to write filter syntax manually.

### Compressor — `LLMChainExtractor`

[main.py](main.py) wraps the base retriever with a `ContextualCompressionRetriever`. The compressor (`LLMChainExtractor`) sends each retrieved document and the original query to the LLM and asks it to extract only the relevant passage. Documents that have no relevant content are dropped entirely.

### Data

[document.py](document.py) defines 8 documents covering renewable energy topics — solar, wind, hydro, geothermal, biomass, energy storage, and environment/policy — each tagged with `year` and `topics` metadata.

Embeddings are stored in a local [ChromaDB](chroma_db/) instance using `text-embedding-ada-002`.

## File Structure

```
contextual_compression/
├── main.py          # Entry point: wires compressor + retriever, runs a sample query
├── retriever.py     # SelfQueryRetriever with metadata field definitions
├── vector_store.py  # ChromaDB setup and document ingestion
├── document.py      # Sample renewable energy documents with metadata
├── llm.py           # LLM client (ChatOpenAI via config)
└── config.json      # API key and base URL (not committed in production)
```

## Running

```bash
python main.py
```

The sample query is:

```
What are the principles behind wind turbines?
```

Expected output: only the sentences from the retrieved chunks that directly discuss wind turbine principles — not the surrounding filler text.

## Key Dependencies

- `langchain-classic` — `ContextualCompressionRetriever`, `LLMChainExtractor`, `SelfQueryRetriever`
- `langchain-openai` — `ChatOpenAI`, `OpenAIEmbeddings`
- `chromadb` — local vector store persistence

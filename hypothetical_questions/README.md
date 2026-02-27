# Hypothetical Questions Strategy for Advanced RAG

## The Core Problem

In standard RAG, you embed your **document chunks** and then embed the **user's query** — hoping their embeddings land close together in vector space. But there's often a **semantic gap**: a document chunk is written as a *statement* while the user query is a *question*. These have different linguistic structures, so their embeddings may not align well.

## How Hypothetical Questions Fix This

The idea is simple but powerful: **instead of embedding the raw document chunks, embed LLM-generated questions that each chunk could answer.** Now at query time, you're comparing a *question* (user's) against *questions* (generated) — a much better semantic match.

## How This Implementation Works

The code follows a clean pipeline:

1. **Start with documents** — `documents.py` defines 8 document chunks about renewable energy, each with metadata.

2. **Generate hypothetical questions per chunk** — For each document, the LLM is prompted (`prompts.py`) to generate exactly 3 questions that document could answer. For example, the wind energy chunk might produce:
   - *"How is wind energy generated?"*
   - *"What technology converts wind into electricity?"*
   - *"What is the role of turbines in wind energy?"*

3. **Store questions as new documents** — In `documents.py`, the generated questions become the `page_content` of new `Document` objects, with the **original chunk stored in metadata** (`parent_chunk`). This is the key trick.

4. **Embed and index the questions** — `vector_store.py` adds these question-documents into a Chroma vector store using OpenAI embeddings. The embeddings are of the *questions*, not the original text.

5. **Retrieve at query time** — When a user asks "How does wind energy work?" (`main.py`), the retriever in `retriever.py` does similarity search against the embedded questions. Since the user's question is semantically similar to the generated questions, it gets a strong match. The original document chunk is available via the `parent_chunk` metadata.

## Visual Summary

```
┌─────────────────────────────────────────────────┐
│              INDEXING TIME                       │
│                                                  │
│  Doc Chunk ──► LLM generates questions ──► Embed │
│  "Wind uses    "How is wind energy        questions
│   turbines..."  generated?"               into vector
│                 "What are turbines?"       store
│                                                  │
│  Original chunk stored in metadata               │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              QUERY TIME                          │
│                                                  │
│  User Question ──► Embed ──► Similarity search   │
│  "How does wind     against question embeddings  │
│   energy work?"                                  │
│                 ──► Return matched doc chunk      │
│                     from metadata                 │
└─────────────────────────────────────────────────┘
```

## Why It Works Better Than Vanilla RAG

- **Question-to-question matching** has higher cosine similarity than question-to-statement matching
- The LLM "pre-thinks" about what queries each chunk is relevant for
- It's a form of **query-document alignment** done at indexing time rather than query time

It's essentially the inverse of **HyDE** (Hypothetical Document Embeddings), where you generate a hypothetical *answer* from the query. Here, you generate hypothetical *questions* from the documents.

## Project Structure

| File | Description |
|------|-------------|
| `config.json` | API key and base URL configuration |
| `llm.py` | LLM client setup (GPT-4o-mini via OpenAI API) |
| `prompts.py` | Prompt template for generating hypothetical questions |
| `documents.py` | Source documents and hypothetical question generation logic |
| `vector_store.py` | Chroma vector store setup and document indexing |
| `retriever.py` | Similarity-based retriever configuration |
| `main.py` | Entry point — runs a sample query |

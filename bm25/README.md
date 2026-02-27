# Hybrid Search for Advanced RAG

## What is Hybrid Search?

Hybrid search combines two retrieval strategies to get the best of both worlds:

1. **Keyword-based search** (sparse) — exact term matching
2. **Semantic search** (dense) — meaning-based vector similarity

Neither approach alone is perfect. Keyword search misses synonyms and paraphrases, while semantic search can miss exact terms or names. Hybrid search fuses both to produce more robust retrieval.

## Key Components

### BM25Retriever

BM25 (Best Matching 25) is a classic information retrieval algorithm that ranks documents based on:

- **Term Frequency (TF)** — how often the query term appears in a document
- **Inverse Document Frequency (IDF)** — how rare the term is across all documents
- **Document Length Normalization** — penalizes very long documents to avoid unfair advantage

BM25 works purely on keyword overlap — no embeddings or neural networks involved. It excels at:
- Exact keyword matching (e.g., product names, error codes, acronyms)
- Queries where specific terms are critical
- Fast, low-cost retrieval with no GPU needed

### EnsembleRetriever

The EnsembleRetriever combines multiple retrievers and merges their results using **Reciprocal Rank Fusion (RRF)**. The typical setup pairs:

- A **BM25Retriever** (keyword-based / sparse)
- A **Vector Store Retriever** (embedding-based / dense)

Each retriever returns its own ranked list. RRF merges them by assigning a score to each document based on its rank position across all retrievers:

```
RRF_score(doc) = sum( 1 / (k + rank_i) )  for each retriever i
```

This means a document ranked highly by *both* retrievers gets boosted, while a document ranked highly by only one still appears in the results.

## How It Works

```
                    User Query
                        |
           ┌────────────┴────────────┐
           v                         v
    BM25 Retriever            Vector Retriever
    (keyword match)           (semantic match)
           |                         |
           v                         v
    Ranked List A             Ranked List B
           |                         |
           └────────────┬────────────┘
                        v
              Reciprocal Rank Fusion
                        |
                        v
               Merged Ranked Results
```

## Why Hybrid Search Works Better

| Scenario | BM25 | Semantic | Hybrid |
|----------|------|----------|--------|
| Exact term lookup (e.g., "HTTP 404") | Great | May miss | Great |
| Synonym matching (e.g., "car" vs "automobile") | Misses | Great | Great |
| Named entities (e.g., "LangChain") | Great | Inconsistent | Great |
| Conceptual queries (e.g., "how to save energy") | Weak | Great | Great |

## Project Structure

| File | Description |
|------|-------------|
| `config.json` | API key and base URL configuration |
| `retriever.py` | Retriever setup with BM25 and Ensemble |

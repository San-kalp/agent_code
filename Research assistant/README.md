# Research Assistant

A research assistant built using **smolagents** with an OpenAI-compatible LLM backend. It loads research papers (PDFs), splits them into chunks, stores them in a Chroma vector store, and uses an LLM for querying.

## Project Structure

| File                 | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `llm.py`             | LLM configuration and initialization                     |
| `config.json`        | API keys and base URL configuration                      |
| `document_loader.py` | Loads and chunks PDFs from `research_papers/`            |
| `vector_store.py`    | Chroma vector store — create, load, and query embeddings |
| `tools.py`           | Custom smolagents Tool for retrieval (RAG)               |
| `state.py`           | State management                                         |
| `nodes.py`           | Node definitions                                         |
| `graph.py`           | Graph/workflow setup                                     |
| `main.py`            | Application entry point                                  |
| `test_file.py`       | Quick test to verify LLM connectivity                    |
| `research_papers/`   | Directory containing PDF research papers                 |
| `chroma_db/`         | Persisted vector store (auto-created on first run)       |

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install smolagents pypdf langchain-community langchain-openai python-dotenv chromadb ddgs tiktoken langchain-classic
```

### 3. Configure API keys

Edit `config.json` with your credentials:

```json
{
  "API_KEY": "your-api-key-here",
  "OPENAI_BASE_URL": "https://your-api-endpoint.com/v1"
}
```

### 4. Add research papers

Place your PDF files in the `research_papers/` directory.

## How it works

### LLM Setup (`llm.py`)

- Reads API credentials from `config.json` using `json.load()`
- `config` becomes a Python dict, access values with `config["API_KEY"]`
- Creates an `OpenAIServerModel` instance from smolagents
- `OpenAIServerModel` is a **callable class** — it implements `__call__`, so you can use it like a function
- The model can be imported and used anywhere:

```python
from llm import llm

# Direct usage — pass OpenAI chat format messages
response = llm([{"role": "user", "content": "Your question here"}])

# Or plug it into a smolagents Agent
from smolagents import CodeAgent
agent = CodeAgent(tools=[], model=llm)
agent.run("Your question here")
```

### Document Loading (`document_loader.py`)

Two functions are available:

#### `load_documents()`

- Uses `PyPDFDirectoryLoader` to scan `research_papers/` for all `.pdf` files
- `loader.load()` opens each PDF, extracts text from every page using `pypdf`
- Returns a **list of `Document` objects** — one per page (not per file)
- Each `Document` has:
  - `doc.page_content` — the extracted text (string)
  - `doc.metadata` — dict with `{"source": "research_papers/file.pdf", "page": 0}`

#### `load_and_split_documents(chunk_size=1000, chunk_overlap=200)`

- Calls `load_documents()` first
- Uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with `cl100k_base` encoding
- Splits each page's text into smaller chunks for RAG use
- `chunk_size=1000` — max ~1000 tokens per chunk
- `chunk_overlap=200` — 200 token overlap between chunks so context isn't lost at boundaries
- Returns a list of smaller `Document` objects (same structure, same metadata)

```python
from document_loader import load_documents, load_and_split_documents

# Just load all pages
docs = load_documents()

# Load and chunk for RAG / vector store
chunks = load_and_split_documents()
```

### Vector Store (`vector_store.py`)

Uses **Chroma** to store document embeddings on disk so you don't re-embed every time.

#### How persistence works

- First run: embeds all chunks and saves to `chroma_db/` folder on disk
- Subsequent runs: loads directly from `chroma_db/` — no re-embedding needed
- To re-create the store (e.g. after adding new papers), delete the `chroma_db/` folder and run again

#### Three functions available:

| Function                | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `create_vector_store()` | Embeds chunks and saves to `chroma_db/`              |
| `load_vector_store()`   | Loads existing store from `chroma_db/`               |
| `get_vector_store()`    | Smart — loads if `chroma_db/` exists, creates if not |

#### Usage

```python
from vector_store import get_vector_store

# Automatically loads from disk or creates if first time
vs = get_vector_store()

# Search for similar documents
results = vs.similarity_search("agentic AI", k=3)
for r in results:
    print(r.page_content[:100])
    print(r.metadata)  # {"source": "research_papers/file.pdf", "page": 0}
```

#### Under the hood

- `OpenAIEmbeddings` converts text chunks into numerical vectors using the API
- `Chroma.from_documents()` takes chunks + embedding model, embeds them, and stores in a collection
- `collection_name="Research_Papers"` is the name of the collection inside Chroma
- `persist_directory="chroma_db"` tells Chroma to save/load from that folder
- Without `persist_directory`, the store is **in-memory only** and lost when the script ends

### Retriever Tool (`tools.py`)

A custom smolagents `Tool` that lets the agent search the vector store during reasoning.

#### How smolagents tools work

smolagents requires tools to be classes that extend `Tool` with these **class attributes**:
- `name` — how the agent refers to the tool (e.g. `"retriever"`)
- `description` — the agent reads this to decide **when** to use the tool
- `inputs` — dict defining what parameters the tool accepts (the agent uses this to construct calls)
- `output_type` — what the tool returns (`"string"`)

These are **not kwargs** — they're class-level attributes that smolagents reads to build a "tool card" for the agent.

#### Key methods

- **`__init__(self, vs, **kwargs)`** — receives the vector store, calls `super().__init__(**kwargs)` to let the parent `Tool` class do its setup (registering name, description, etc.)
- **`forward(self, query)`** — the actual logic that runs when the agent calls the tool. Takes a query string, runs `similarity_search(query, k=3)`, and returns formatted document results

#### Usage

```python
from tools import retriever_tool
from smolagents import CodeAgent
from llm import llm

# The retriever_tool is pre-initialized with the vector store
agent = CodeAgent(tools=[retriever_tool], model=llm)
agent.run("What is agentic AI?")
```

The agent autonomously decides when to call `retriever` based on the tool's description, constructs the query, and uses the returned documents to form its answer.

### Application Entry Point (`main.py`)

Ties everything together — creates a `CodeAgent` with the retriever tool and runs a query.

```python
from smolagents import CodeAgent
from tools import *
from llm import llm

agent = CodeAgent(
    tools = [retriever_tool],
    model= llm,
    max_steps = 8,
    verbosity_level= 2
)

agent.run("What are the key aspects of an Agentic AI System")
```

#### What is `CodeAgent`?

`CodeAgent` is smolagents' primary agent type. It's called "Code" agent because **it writes and executes Python code** as its reasoning mechanism, rather than just outputting text.

#### How it works step-by-step

1. **Receives your query** — e.g. `"What are the key aspects of an Agentic AI System"`
2. **Thinks in code** — the LLM generates Python code to call tools:
   ```python
   # Step 1 — agent generates and executes:
   docs = retriever("key aspects of agentic AI systems")
   print(docs)
   ```
3. **Executes the code** — smolagents runs it in a sandboxed Python environment, which triggers `RetrieverTool.forward()` and hits the vector store
4. **Reads the output** — the agent sees the retrieved document chunks
5. **Repeats if needed** — up to `max_steps` reasoning cycles, refining queries or combining results
6. **Returns a final answer** — once it has enough info:
   ```python
   final_answer("Based on the research papers, the key aspects are: ...")
   ```

#### Why code instead of plain text?

Code lets the agent do things that plain tool-calling can't:
- **Variables** — store and reuse results: `result = retriever("query")`
- **Loops** — iterate over multiple queries: `for topic in topics: retriever(topic)`
- **Conditionals** — `if "not found" in result: retriever("different query")`
- **Combining results** — `combined = result1 + result2`

The alternative is `ToolCallingAgent`, which uses structured JSON tool calls (like OpenAI function calling) — simpler but less flexible.

#### Parameters

| Parameter | What it does |
|---|---|
| `tools=[retriever_tool]` | Tools the agent can use — it reads each tool's `name`, `description`, and `inputs` to know when/how to call them |
| `model=llm` | The LLM powering the agent's reasoning (from `llm.py`) |
| `max_steps=8` | Cap on reasoning cycles — prevents infinite loops. Each step = one code generation + execution |
| `verbosity_level=2` | Shows detailed logs — agent's generated code, tool outputs, and reasoning at each step |

### Key Python concepts used

- **`enumerate(list)`** — iterate with an index: `for i, doc in enumerate(documents)`
- **String slicing** — `doc.page_content[:10]` gives first 10 characters, `[10]` gives just the 10th character
- **Splitting into lines** — `doc.page_content.split("\n")[:10]` gives first 10 lines
- **f-strings** — `print(f"Doc {i}")` for formatted output
- **`if __name__ == "__main__"`** — code block only runs when file is executed directly, not when imported
- **`os.path.exists()`** — check if a file/directory exists before deciding what to do
- **`super().__init__()`** — calls the parent class constructor so its setup runs first
- **Class attributes vs instance attributes** — `name`, `description` are class-level (shared); `self.vs` is instance-level (per object)

### Importing across files

Any variable/object defined at module level can be imported:

```python
from llm import llm                    # import the LLM instance
from llm import api_key, base_url      # import config values
from document_loader import load_and_split_documents  # import functions
from vector_store import get_vector_store             # import vector store
from tools import retriever_tool                      # import the pre-built tool
```

The module code runs **once** on first import and is cached — subsequent imports reuse the same instance.

## Testing

Run the test file to verify your LLM setup:

```bash
python test_file.py
```

Test document loading:

```bash
python document_loader.py
```

Test vector store creation and search:

```bash
python vector_store.py
```

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from document_loader import load_and_split_documents
from llm import api_key, base_url

PERSIST_DIR = "chroma_db"

embedding_model = OpenAIEmbeddings(
    api_key=api_key,
    base_url=base_url,
)

def create_vector_store():
    """Load chunks and store them in a persistent Chroma vector store."""
    chunks = load_and_split_documents()
    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        collection_name="Research_Papers",
        persist_directory=PERSIST_DIR
    )
    print(f"Vector store created and saved to '{PERSIST_DIR}/' with {vectorstore._collection.count()} documents")
    return vectorstore

def load_vector_store():
    """Load an existing Chroma vector store from disk."""
    vectorstore = Chroma(
        collection_name="Research_Papers",
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR
    )
    print(f"Loaded vector store from '{PERSIST_DIR}/' with {vectorstore._collection.count()} documents")
    return vectorstore

def get_vector_store():
    """Get vector store — loads from disk if exists, otherwise creates it."""
    if os.path.exists(PERSIST_DIR):
        return load_vector_store()
    return create_vector_store()

if __name__ == "__main__":
    vs = get_vector_store()
    results = vs.similarity_search("agentic AI", k=3)
    for r in results:
        print(f"  - {r.metadata['source']} (page {r.metadata['page']})")
        print(f"    {r.page_content[:100]}...")

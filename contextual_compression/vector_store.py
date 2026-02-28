from langchain_classic.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from llm import api_key, base_url
from document import documents_with_metadata

PERSIST_DIR ="chroma_db"


embedding_producer = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key, base_url=base_url)

def load_vector_store():
    vectorstore = Chroma(collection_name="renewable_energy",embedding_function=embedding_producer,persist_directory=PERSIST_DIR)
    vectorstore.add_documents(documents=documents_with_metadata)
    return vectorstore
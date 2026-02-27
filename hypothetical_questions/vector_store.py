from langchain_classic.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from llm import api_key, base_url
from documents import hypothetical_question_documents


PERSIST_DIR ="chroma_db"


embedding_producer = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key, base_url=base_url)


def load_vector_store():
    vectorstore = Chroma(collection_name="hypothetical_questions",embedding_function=embedding_producer,persist_directory=PERSIST_DIR)
    vectorstore.add_documents(
        id = [d.id for d in hypothetical_question_documents], 
        documents= hypothetical_question_documents
    )
    return vectorstore

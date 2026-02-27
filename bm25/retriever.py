from langchain_community.retrievers import BM25Retriever
from documents import documents


bm25_rt = BM25Retriever.from_documents(documents)

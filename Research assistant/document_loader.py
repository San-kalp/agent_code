from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

RESEARCH_PAPERS_DIR = "research_papers"

def load_documents():
    """Load all PDFs from the research_papers directory."""
    loader = PyPDFDirectoryLoader(RESEARCH_PAPERS_DIR)
    documents = loader.load()
    return documents

def load_and_split_documents(chunk_size=1000, chunk_overlap=200):
    """Load all PDFs and split them into chunks."""
    documents = load_documents() #You get a list of docs here. Each doc is an object which has two things, one will be the page_content and one will be the meta data    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0].page_content[:100]}...")
    return chunks

# if __name__ == "__main__":
#     docs = load_documents()
#     num_files = len(set(doc.metadata["source"] for doc in docs))
#     print(f"Loaded {len(docs)} pages from {num_files} research papers from research papers folder")
#     for doc in docs[:3]:
#         print(f"  - {doc.metadata['source']} (page {doc.metadata['page']})")

#     chunks = load_and_split_documents()
#     print(f"Splitted the documents into {len(chunks)} chunks")

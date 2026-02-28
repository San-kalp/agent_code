from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from llm import llm
from langchain_classic.retrievers import ContextualCompressionRetriever
from retriever import retriever_with_metadata


compressor = LLMChainExtractor.from_llm(llm)
compressor_retriver = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_with_metadata)

def main(question : str):
    result = compressor_retriver.invoke(input=question)
    print(f"\nQuestion: {question}")
    print(f"Documents found: {len(result)}\n")
    for i, doc in enumerate(result):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print()
        
question = "What are the principles behind wind turbines?"
main(question)
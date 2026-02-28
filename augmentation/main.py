from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from retriever import retriever_with_metadata
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

crossencoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


def main(question : str):
    result = retriever_with_metadata.invoke(input=question)
    print(f"\nQuestion: {question}")
    print(f"Documents found: {len(result)}\n")
    for i, doc in enumerate(result):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print()
    
    print("Code for cross encoding")
    #First step is to map the question with the doc
    context_question_pairs = [[question, doc.page_content] for doc in result] #This is a list of list 
    print("below is the output of cross encoder")
    cross_encoder_output = crossencoder.score(context_question_pairs)
    print(cross_encoder_output)
    
    #Now let us do re ranking 
    reranker=CrossEncoderReranker(model=crossencoder,top_n=5)
    reranker_retriever = ContextualCompressionRetriever(base_compressor=reranker,base_retriever=retriever_with_metadata)

    reranked_result = reranker_retriever.invoke(question)
    print(f"\nAfter reranking — top {len(reranked_result)} documents:")
    for i, doc in enumerate(reranked_result):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print()



main("How is Hydroelectric power used ?")
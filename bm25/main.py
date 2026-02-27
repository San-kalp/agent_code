from retriever import bm25_rt


def main(question:str) :
    result = bm25_rt.invoke(input=question)
    print(f"\nQuestion: {question}")
    print(f"Documents found: {len(result)}\n")
    for i, doc in enumerate(result):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print()
        
main("How does wind energy work?")
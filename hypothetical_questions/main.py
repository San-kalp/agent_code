from retriever import hypo_que_rt

def main(question:str) :
    result = hypo_que_rt.invoke(question)
    print(f"\nQuestion: {question}")
    print(f"Documents found: {len(result)}\n")
    for i, doc in enumerate(result):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print()
        
main("How does wind energy work?")
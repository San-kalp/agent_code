from retriever import retriever_with_metadata

def main(question : str):
    result = retriever_with_metadata.invoke(input=question)
    print(f"\nQuestion: {question}")
    print(f"Documents found: {len(result)}\n")
    for i, doc in enumerate(result):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print()

main("Who got the dogs out ??")
    
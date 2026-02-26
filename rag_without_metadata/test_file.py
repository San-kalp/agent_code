from llm import llm
from vector_store import load_vector_store

def main():
    # result =llm.invoke("Hi how are you ?")
    vs = load_vector_store()
    retriever = vs.as_retriever(enable_limit = True)
    question = "How does wind energy work ?"
    result = retriever.invoke(question)
    print(result)
    
main()
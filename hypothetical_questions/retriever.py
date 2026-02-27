from vector_store import load_vector_store

vs = load_vector_store()

hypo_que_rt = vs.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":1}
)
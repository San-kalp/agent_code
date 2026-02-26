from smolagents import Tool
from vector_store import  get_vector_store

class RetrieverTool(Tool):
    name = "retriever"
    description = "Leverages semantic search to retrieve the most contextually relevant sections from the documentation based on a user query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to match against documentation. Phrase it as a natural language statement that reflects the kind of information you're seeking (e.g., 'How do I configure logging?')."
        }
    }
    output_type = "string"
    def __init__(self, vs, **kwargs):
        super().__init__(**kwargs)
        self.vs = vs
    
    def forward(self, query : str) -> str :
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        results = self.vs.similarity_search(query, k=3)
        formatted_docs = "\n\n".join(
            f"===== Document {i} =====\nSource: {doc.metadata['source']} (page {doc.metadata['page']})\n{doc.page_content}"
            for i, doc in enumerate(results)
        )
        return formatted_docs

vs = get_vector_store()
retriever_tool = RetrieverTool(vs)
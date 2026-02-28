from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from llm import llm
from vector_store import load_vector_store

metadata_field_info = [
    AttributeInfo(
        name="year",
        description="The year the document was created or published",
        type = "integer",
    ), 
    AttributeInfo(
        name="topics",
        description="The main topic of the document",
        type="string",
    ),
    AttributeInfo(
        name ="subtopic",
        description="A more specific subcategory of the main topic.",
        type="subtring"
    )
]

document_content_description = """
Brief overview of various aspects related to Renewable Enery 
and differt types of it like Wind, solar, hydroelectric, geothermal energies,...etc
"""

vector_store_with_metadata = load_vector_store()

retriever_with_metadata = SelfQueryRetriever.from_llm(
    llm= llm,
    vectorstore=vector_store_with_metadata,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose = True,
    # search_kwargs={"score_threshold": 0.5}
)


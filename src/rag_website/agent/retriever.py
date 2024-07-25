from rag_website.vectorstore import vectorstore
from rag_website.state import GraphState

# Instantinate retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

def retrieve_documents(state: GraphState):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("   RETRIEVING DOCUMENTS...   ")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"   RETRIEVED {len(documents)} DOCUMENTS")
    return state | {"documents": documents}
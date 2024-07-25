from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_website.state import GraphState
from rag_website.model import llm

def generate_answer(state: GraphState):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("   GENERATING ANSWER   ")
    question = state["question"]
    documents = state["documents"]
    
    # Prompt
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
    )

    # RAG generation
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"question": question, "context": documents})
    return state | {"generation": generation}

def decide_to_answer(state: GraphState):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("   DECIDING TO ANSWER OR NOT....   ")
    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("   DECISION: INCLUDE WEB SEARCH   ")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("   DECISION: GENERATE ANSWER   ")
        return "generate"

if __name__ == "__main__":
    import time
    from rag_website.agent.retriever import retrieve_documents
    from rag_website.agent.retrieval_checker import filter_documents

    question = "llm agent memory"
    state = {
        "question": question
        }
    state = retrieve_documents(state)
    state = filter_documents(state)

    start = time.time()
    state = generate_answer(state)
    end = time.time()

    print(f"Answer : {state["generation"]}")
    print(f"The time required to generate response by Router Chain in seconds: {end - start}")
    
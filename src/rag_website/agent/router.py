from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from rag_website.state import GraphState
from rag_website.model import llm

def route_question(state: GraphState):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("   ROUTING QUESTION...   ")
    question = state["question"]

    # Prompt
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()
    source = question_router.invoke({"question": question})

    # Select route
    if source['datasource'] == 'web_search':
        print("   ROUTE QUESTION TO WEB SEARCH   ")
        return "vectorstore"
    elif source['datasource'] == 'vectorstore':
        print("   ROUTE QUESTION TO RAG   ")
        return "vectorstore"

if __name__ == "__main__":
    import time

    question = "llm agent memory"
    state = {
        "question": question
        }
    start = time.time()
    response = route_question(state)
    end = time.time()
    print(f"Datasource : {response}")
    print(f"The time required to generate response by Router Chain in seconds: {end - start}")
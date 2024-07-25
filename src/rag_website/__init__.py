import os
os.environ['USER_AGENT'] = 'myagent'

from langgraph.graph import END, StateGraph

from rag_website.state import GraphState
from rag_website.agent import (
    answerer,
    hallucination_checker,
    retrieval_checker,
    router,
    retriever,
    web_searcher,
    )

def app():

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("Web Search", web_searcher.web_search)
    workflow.add_node("Retrieve Documents", retriever.retrieve_documents)
    workflow.add_node("Filter Documents", retrieval_checker.filter_documents)
    workflow.add_node("Generate Answer", answerer.generate_answer)

    workflow.set_conditional_entry_point(
        router.route_question,
        {
            "websearch": "Web Search",
            "vectorstore": "Retrieve Documents",
        },
    )
    workflow.add_edge("Retrieve Documents", "Filter Documents")
    workflow.add_conditional_edges(
        "Filter Documents",
        answerer.decide_to_answer,
        {
            "websearch": "Web Search",
            "generate": "Generate Answer",
        },
    )
    workflow.add_edge("Web Search", "Generate Answer")
    workflow.add_conditional_edges(
        "Generate Answer",
        hallucination_checker.check_hallucination,
        {
            "hallucinate": "Generate Answer",
            "useful": END,
            "not useful": "Web Search",
        },
    )

    app = workflow.compile()

    return app

def run_app(msg):
    for output in app.stream(msg):
        for key, value in output.items():
            print(f"\033[94mFinished running : {key}\033[0m")
    print(f"""
Question = 
{msg["question"]}
Answer =
{value["generation"]}
""")

if __name__ == "__main__":

    app = app()

    msg = [
        {"question": "What is prompt engineering?"},
        {"question": "Who is the CEO of Thinking Machines Data Science Inc. from Philippines?"},
        {"question": "What are the types of agent memory?"},]
    
    run_app(msg[0])
    run_app(msg[1])
    run_app(msg[2])
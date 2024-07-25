from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from rag_website.state import GraphState
from rag_website.model import llm
from rag_website.agent.retriever import retriever


def filter_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("   CHECK DOCUMENT FROM RETRIEVER   ")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
    )
    retrieval_grader = prompt | llm | JsonOutputParser()

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("   DECISION: DOCUMENTS \033[92mRELEVANT   \033[0m")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("   DECISION: DOCUMENTS \033[91mNOT RELEVANT   \033[0m")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    print(f"   Total of {len(filtered_docs)} RELEVENT DOCUMENTS")
    return state | {"documents": filtered_docs, "web_search": web_search}

if __name__ == "__main__":
    import time
    from rag_website.agent.retriever import retrieve_documents
    
    question = "llm agent memory"
    state = {
        "question" : question
    }
    state = retrieve_documents(state)

    start = time.time()
    state = filter_documents(state)
    end = time.time()
    
    print(f"Web Search : {state["web_search"]}")
    print(f"The time required to generate response by the retrieval grader in seconds: {end - start}")
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from rag_website.state import GraphState
from rag_website.model import llm
from rag_website.agent.answer_checker import check_answer

def check_hallucination(state: GraphState):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("   CHECKING HALLUCINATIONS   ")
    documents = state["documents"]
    generation = state["generation"]

    prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    not_hallucinate = score['score']

    # Check hallucination
    if not_hallucinate == "yes":
        print("   DECISION: \033[92mDOES NOT HALLUCINATE   \033[0m")
        # Check question-answering
        score = check_answer(state)
        answer_correct = score['score']
        if answer_correct == "yes":
            print("   DECISION: \033[92mGENERATION ANSWERS QUESTION   \033[0m")
            return "useful"
        else:
            print("   DECISION: \033[91mGENERATION DOES NOT ANSWERS QUESTION   \033[0m")
            return "not useful"
    else:
        print("   DECISION: \033[91mHALLUCINATE   \033[0m")
        return "hallucinate"

if __name__ == "__main__":
    import time
    
    from rag_website.agent.retriever import retrieve_documents
    from rag_website.agent.answerer import generate_answer
    from rag_website.agent.retrieval_checker import filter_documents
    
    question = "llm agent memory"
    state = {
        "question": question
        }
    state = retrieve_documents(state)
    state = filter_documents(state)
    state = generate_answer(state)

    start = time.time()
    response = check_hallucination(state)
    end = time.time()
    
    print(f"Generation is : {response}")
    print(f"The time required to generate response by the generation chain in seconds: {end - start}")
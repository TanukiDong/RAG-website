from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from rag_website.state import GraphState
from rag_website.model import llm

def check_answer(state: GraphState):
    print("   CHECK ANSWER   ")
    question = state["question"]
    generation = state["generation"]

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()

    score = answer_grader.invoke({"question": question, "generation": generation})
    return score

if __name__ == "__main__":
    import time

    from rag_website.agent.answerer import generate_answer
    from rag_website.agent.retriever import retrieve_documents
    from rag_website.agent.retrieval_checker import filter_documents

    question = "llm agent memory"
    state = {
        "question" : question
    }
    state = retrieve_documents(state)
    state = filter_documents(state)
    state = generate_answer(state)

    start = time.time()
    response = check_answer(state)
    end = time.time()
    
    print(f"The time required to generate response by the answer grader in seconds: {end - start}")
    print(f"Score : {response["score"]}")


from typing import TypedDict, List

class GraphState(TypedDict):
    question : str
    generation : str
    web_search : str
    documents : List[str]
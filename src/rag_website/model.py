import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureOpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

from rag_website.settings import (
    OPENAI_ENDPOINT,
    EMBEDDING_MODEL_ID,
    GROQ_API_KEY,
    CHAT_MODEL_ID,
    TRAVILY_KEY)

azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)
os.environ['TAVILY_API_KEY'] = TRAVILY_KEY

# Instantiate embedding model
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_ID)

embeddings = AzureOpenAIEmbeddings(
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=EMBEDDING_MODEL_ID
    )

# Instantiate LLM
llm = ChatGroq(temperature=0,
                      model_name=CHAT_MODEL_ID,
                      api_key=GROQ_API_KEY,)

# Instantiate Tavily Search
api_wrapper = TavilySearchAPIWrapper()
web_search_tool = TavilySearchResults(api_wrapper=api_wrapper)
import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

from rag_website.settings import (
    OPENAI_ENDPOINT,
    EMBEDDING_MODEL_ID,
    OPENAI_API_VERSION,
    LLM_MODEL_ID,
    TRAVILY_KEY)

azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)
os.environ['TAVILY_API_KEY'] = TRAVILY_KEY

# # Instantiate embedding model
embeddings = AzureOpenAIEmbeddings(
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=EMBEDDING_MODEL_ID
    )

# Instantiate LLM
llm = AzureChatOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=LLM_MODEL_ID,
    )

# Instantiate Tavily Search
api_wrapper = TavilySearchAPIWrapper()
web_search_tool = TavilySearchResults(api_wrapper=api_wrapper)
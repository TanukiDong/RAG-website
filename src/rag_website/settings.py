import os
from dotenv import load_dotenv

load_dotenv(".envrc")
# load_dotenv(".env")

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TRAVILY_KEY = os.getenv("TRAVILY_KEY", "")

EMBEDDING_MODEL_ID = "ada-002"
LLM_MODEL_ID = "gpt-4o"
OPENAI_API_VERSION = "2024-02-01"


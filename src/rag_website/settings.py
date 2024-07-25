import os
from dotenv import load_dotenv

load_dotenv(".envrc")
# load_dotenv(".env")

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TRAVILY_KEY = os.getenv("TRAVILY_KEY", "")
EMBEDDING_MODEL_ID = "ada-002"
CHAT_MODEL_ID = "Llama3-8b-8192"

# EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"
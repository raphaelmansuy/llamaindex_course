""" This script queries the index that was created and persisted in the previous step """

import os.path

# import the logging and sys modules
import logging
import sys

from dotenv import load_dotenv

from llama_index.core import (
    ServiceContext,
    StorageContext,
    load_index_from_storage
)

# Import Ollama
# You need to install llama-index-llms-ollama to use Ollama
from llama_index.llms.ollama import Ollama

# you need to install llama-index-embeddings-langchainto to use OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# nomic-embed-text is a powerful model that can be used for embeddings https://ollama.com/library/nomic-embed-text
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:latest"


# Import load_dotenv from the dotenv module
# Load the .env file

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# check if storage already exists
PERSIST_DIR = os.environ.get("PERSIST_DIR")

if not os.path.exists(PERSIST_DIR):
    print(" No index found, please run 08a_index_and_persist.py first")
    sys.exit(1)

# load the index from storage
service_context = ServiceContext.from_defaults(
    llm=Ollama(model=LLM_MODEL, tokens=1000, request_timeout=60),
    embed_model=OllamaEmbeddings(model=EMBEDDING_MODEL)
)

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(
    storage_context, service_context=service_context)

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query(
    "Explain the result of Air Liquide in 2023 ? Make a summary")
# print the response
print(response)

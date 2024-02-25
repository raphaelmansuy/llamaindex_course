""" Load the documents from the data directory and create an index """

import os.path

# import the logging and sys modules
import logging
import sys

from llama_index.core import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
)

# Import Ollama
# You need to install llama-index-llms-ollama to use Ollama
from llama_index.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings


# Import load_dotenv from the dotenv module
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# nomic-embed-text is a powerful model that can be used for embeddings https://ollama.com/library/nomic-embed-text
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:latest"


# check if storage already exists
PERSIST_DIR = os.environ.get("PERSIST_DIR")

if os.path.exists(PERSIST_DIR):
    # remove the existing storage
    import shutil
    shutil.rmtree(PERSIST_DIR)

if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()

    service_context = ServiceContext.from_defaults(
        llm=Ollama(model=LLM_MODEL),
        embed_model=OllamaEmbeddings(model=EMBEDDING_MODEL)
    )
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context)

    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)

""" Query a JSON file"""

import os.path

# import the logging and sys modules
import logging
import sys

from llama_index.core import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    load_index_from_storage,
)

# Import Ollama
# You need to install llama-index-llms-ollama to use Ollama
from llama_index.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
#from llama_index.core.node_parser import JSONNodeParser

# Import load_dotenv from the dotenv module
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logger = logging.getLogger(__name__)

# nomic-embed-text is a powerful model that can be used for embeddings https://ollama.com/library/nomic-embed-text
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:latest"

# Define the service context
service_context = ServiceContext.from_defaults(
    llm=Ollama(model=LLM_MODEL, tokens=8096, request_timeout=120, temperature=0.1),
    embed_model=OllamaEmbeddings(model=EMBEDDING_MODEL, show_progress=True),
)

# check if storage already exists
PERSIST_DIR = os.environ.get("PERSIST_DIR")

# You need to set this variable to True if you want to remove the existing storage
# and recreate the index from the documents
REMOVE_EXISTING_STORAGE = False

if REMOVE_EXISTING_STORAGE and os.path.exists(PERSIST_DIR):
    # remove the existing storage
    import shutil

    logger.info("Removing existing storage at %s", PERSIST_DIR)
    shutil.rmtree(PERSIST_DIR)
    # load the documents and create the index
    documents = SimpleDirectoryReader("data_json").load_data(show_progress=True)

    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, show_progress=True
    )

    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # We can load the existing index from the storage
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)



QUERY = """
Please summarize the patient's medical status, including his/her condition, diagnostic findings, and treatment plan. Use Markdown formatting.
"""

logger.info("Querying the index with the query: %s", QUERY)

# Get a query engine from the index
query_engine = index.as_query_engine()
response = query_engine.query(QUERY)

# print the response
logger.info("Answser: %s\n", response)

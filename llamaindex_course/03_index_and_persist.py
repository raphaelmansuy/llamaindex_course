""" Load the documents from the data directory and create an index """

import os.path

# import the logging and sys modules
import logging
import sys

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)

# Import load_dotenv from the dotenv module
from dotenv import load_dotenv
# Load the .env file
# Use OPENAI_API_KEY to get the API key from the .env file

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# check if storage already exists
PERSIST_DIR = os.environ.get("PERSIST_DIR")

if os.path.exists(PERSIST_DIR):
    # remove the existing storage
    import shutil
    shutil.rmtree(PERSIST_DIR)

if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)

""" Load the documents from the data directory and create an index """

import os.path

# import the logging and sys modules
import logging
import sys

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)


# Load a sentence splitter
from llama_index.core.node_parser import SentenceSplitter


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

    # log the number of documents
    logging.info("Loaded %s documents", len(documents))

    split_size = 512

    logging.info("Creating index with smaller chunks of %s tokens", split_size)

    index = VectorStoreIndex.from_documents(
      documents, transformations=[SentenceSplitter(chunk_size=split_size)])

    logging.info("Index created")

    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)

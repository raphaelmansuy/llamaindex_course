""" """

import os.path

# import the logging and sys modules
import logging
import sys

from llama_index.core import (
    StorageContext,
    load_index_from_storage
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

if not os.path.exists(PERSIST_DIR):
    print(" No index found, please run 03_index_and_persist.py first")
    sys.exit(1)

# load the index from storage
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("Explain the result of Air Liquide in 2023 ? Make a summary")
print(response)


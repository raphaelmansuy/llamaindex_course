import logging
import sys

from pathlib import Path
from typing import List

import json 

from llama_index.core.node_parser import JSONNodeParser
from llama_index.readers.file import FlatReader
from llama_index_client import Document



DOCUMENTS_DIR = "./data_json/0d5f5a77-b49d-4f8b-887f-70e9de390751.json"

## get a loggger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logger = logging.getLogger(__name__)

## Step1 : Load the documents from the directory
logger.info("Loading the documents from the directory %s", DOCUMENTS_DIR)
json_docs : List[Document] = FlatReader().load_data(Path(DOCUMENTS_DIR))

logger.info("%s Documents loaded successfully.", len(json_docs))

## Step2 : Parse the documents to get the nodes
logging.info("Parsing the documents to get the nodes")
parser = JSONNodeParser()
nodes = parser.get_nodes_from_documents(json_docs,show_progress=True)

logger.info("%s Nodes parsed successfully.", len(nodes))

## display the first 5 nodes
#logger.info("Displaying the first node of %s", len(nodes))
#for node in nodes[:1]:
#    # Assuming node.to_dict() returns a dictionary representation of the node
#    logger.info(json.dumps(node.to_dict(), indent=4))
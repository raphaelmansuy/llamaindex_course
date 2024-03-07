import os.path

# import the logging and sys modules
import logging
import shutil

from typing import List

from rich.console import Console

from llama_index.core import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.schema import TextNode

console = Console()

# Import Ollama
# You need to install llama-index-llms-ollama to use Ollama
from llama_index.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Import load_dotenv from the dotenv module
from dotenv import load_dotenv

from llamaindex_course.demo_medical.fhir_reader import (
    read_fhir_file,
    get_fhir_resource_types,
)

# Load the .env file
load_dotenv()

# nomic-embed-text is a powerful model that can be used for embeddings https://ollama.com/library/nomic-embed-text
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:latest"
# check if storage already exists
PERSIST_DIR = os.environ.get("PERSIST_DIR")

# You need to set this variable to True if you want to remove the existing storage
# and recreate the index from the documents
REMOVE_EXISTING_STORAGE = True

# Example of a FHIR file
FILE_TO_OPEN = "data_json/0d5f5a77-b49d-4f8b-887f-70e9de390751.json"

# Get the full path of the file
FULL_PATH = os.path.join(os.getcwd(), FILE_TO_OPEN)


def get_service_context() -> ServiceContext:
    """
    Get the service context
    """
    service_context = ServiceContext.from_defaults(
        llm=Ollama(model=LLM_MODEL, tokens=8096, request_timeout=120, temperature=0.0),
        embed_model=OllamaEmbeddings(model=EMBEDDING_MODEL, show_progress=True),
    )
    return service_context


def index_fhir_file(file_path: str) -> VectorStoreIndex:
    """ "
    Index the FHIR file
    """
    resources = []
    data = read_fhir_file(file_path)
    # Get patient resource type
    patient = get_fhir_resource_types(data, "Patient")
    resources.append(patient)
    # Get Observation resource type
    observations = get_fhir_resource_types(data, "Observation")
    # Add the observations to the resources list
    resources.extend(observations)
    # Get Condition resource type
    conditions = get_fhir_resource_types(data, "Condition")
    resources.extend(conditions)
    # Get Encounter resource type
    encounters = get_fhir_resource_types(data, "Encounter")
    resources.extend(encounters)

    # Add the resources to the vector store
    resource_nodes: List[TextNode] = []
    id = 0
    for resource in resources:
        console.print(f"Adding resource {resource} to the index")
        # generate an unique id for the resource
        id = id + 1
        resource_node = TextNode(text=str(resource), id=str(id) )
        resource_nodes.append(resource_node)

    vector_store = VectorStoreIndex(nodes=resource_nodes)
    return vector_store


def get_vector_store_index() -> VectorStoreIndex:
    """
    Get the vector store index
    """
    if REMOVE_EXISTING_STORAGE and os.path.exists(PERSIST_DIR):
        # remove the existing storage

        logging.info("Removing existing storage at %s", PERSIST_DIR)
        shutil.rmtree(PERSIST_DIR)
        # load the documents and create the index from the documents
        if not os.path.exists(PERSIST_DIR):
            os.makedirs(PERSIST_DIR)

        index = index_fhir_file( FULL_PATH)

    else:
        # load the index from the storage
        index = load_index_from_storage(PERSIST_DIR)

    return index


def main():
    """
    Main function
    """
    console.print("Ollama Index FHIR Example")

    index = get_vector_store_index()
    QUERY = "Can you create a detailled summary the patient's medical status?"
    console.print(f"Querying the index with the query: {QUERY}")
    # Get a query engine from the index
    query_engine = index.as_query_engine()
    response = query_engine.query(QUERY)
    # print the response
    console.print(response)


if __name__ == "__main__":
    main()

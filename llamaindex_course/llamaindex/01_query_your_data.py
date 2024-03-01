"""
A simple example of how to query your data using LlamaIndex and OpenAI API
"""
# Import load_dotenv from the dotenv module
from dotenv import load_dotenv
# Import the VectorStoreIndex and SimpleDirectoryReader from the llama_index.core module
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load the .env file

# Use OPENAI_API_KEY to get the API key from the .env file

load_dotenv()

# Load the data from the "data" directory using the SimpleDirectoryReader
documents = SimpleDirectoryReader("data").load_data()

# Create a VectorStoreIndex from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Query the data
response = query_engine.query("Explain the result of Air Liquide in 2023 ? Make a summary")
print(response)

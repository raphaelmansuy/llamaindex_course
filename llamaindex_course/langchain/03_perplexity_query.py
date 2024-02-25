import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# import dotenv
# Import load_dotenv from the dotenv module
from dotenv import load_dotenv

load_dotenv()

PERPLEXITY_API = os.getenv("PERPLEXITY_API")
PERPLEXITY_MODEL_NAME = os.getenv("PERPLEXITY_MODEL_NAME")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_KEY")

print(f"PERPLEXITY_API: {PERPLEXITY_API}")
print(f"PERPLEXITY_MODEL_NAME: {PERPLEXITY_MODEL_NAME}")
# print *only* the first 5 characters of the key
print(f"PERPLEXITY_KEY: {PERPLEXITY_KEY[:5]}")




# Create a prompt
prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic} and {animal}")
# Create a model
model = ChatOpenAI(model=PERPLEXITY_MODEL_NAME,
                   base_url=PERPLEXITY_API, openai_api_key=PERPLEXITY_KEY)
# Create an output parser
output_parser = StrOutputParser()

# Create a chain
chain = prompt | model | output_parser

for chunk in chain.stream({"animal": "bears", "topic": "ice cream"}):
    print(chunk, end="", flush=True)

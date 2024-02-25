from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# import dotenv
# Import load_dotenv from the dotenv module
from dotenv import load_dotenv

load_dotenv()


# Create a prompt
prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic} and {animal}")
# Create a model
model = ChatOpenAI(model="gpt-3.5-turbo")
# Create an output parser
output_parser = StrOutputParser()

# Create a chain
chain = prompt | model | output_parser

for chunk in chain.stream({"animal": "bears", "topic": "ice cream"}):
    print(chunk, end="", flush=True)

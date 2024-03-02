import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "mistral:latest"

MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo-16k"
MODEL_API = os.environ.get("MODEL_API") or None
MODEL_KEY = os.environ.get("MODEL_KEY") or os.environ.get("OPENAI_API_KEY")

print(f"Using model {MODEL_NAME} with API {MODEL_API}")

model = ChatOpenAI(model_name=MODEL_NAME, api_key=MODEL_KEY, base_url=MODEL_API)

prompt = ChatPromptTemplate.from_template(
    "tell me a joke about {foo}, more than {word_count} words"
)
chain = prompt | model | StrOutputParser()

for chunk in chain.stream({"foo": "bears", "word_count": 2000}):
    print(chunk, end="", flush=True)

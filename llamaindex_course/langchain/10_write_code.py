import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL

from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo-16k"
MODEL_API = os.environ.get("MODEL_API") or None
MODEL_KEY = os.environ.get("MODEL_KEY") or os.environ.get("OPENAI_API_KEY")

print(f"Using model {MODEL_NAME} with API {MODEL_API}")

# Create a model

model = ChatOpenAI(
    model_name=MODEL_NAME,
    api_key=MODEL_KEY,
    base_url=MODEL_API,
    temperature=0,
    max_tokens=4096,
)


def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]


TEMPLATE = """Write some python code to solve the user's problem. 

Return only python code in Markdown format, e.g.:

```python
...
```"""
prompt = ChatPromptTemplate.from_messages([("system", TEMPLATE), ("human", "{input}")])


output_parser = StrOutputParser()

chain = prompt | model | StrOutputParser() | _sanitize_output | PythonREPL().run

res = chain.invoke(
    {"input": "Calculate the surface of a cone with a radius of 3 and a height of 5.8"}
)

print(res)

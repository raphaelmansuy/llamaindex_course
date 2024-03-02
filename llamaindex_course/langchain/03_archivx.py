import os
import datetime
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo-16k"
MODEL_API = os.environ.get("MODEL_API") or None
MODEL_KEY = os.environ.get("MODEL_KEY") or os.environ.get("OPENAI_API_KEY")

print(f"Using model {MODEL_NAME} with API {MODEL_API}")

model = ChatOpenAI(model_name=MODEL_NAME, api_key=MODEL_KEY, base_url=MODEL_API)


tools = load_tools(["arxiv"])

agent_chain = initialize_agent(
    tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


agent_chain.run(
    "What articles has published about machine learning ?",
)

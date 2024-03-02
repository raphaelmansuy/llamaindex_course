import os
from langchain import hub

from langchain.agents import AgentExecutor, create_self_ask_with_search_agent

from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilyAnswer

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


tools = [TavilyAnswer(max_results=5, name="Intermediate Answer")]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/self-ask-with-search")

# Construct the Self Ask With Search Agent
agent = create_self_ask_with_search_agent(model, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke(
    {"input": "What is the hometown of the reigning men's U.S. Open champion?"}
)

print(result)
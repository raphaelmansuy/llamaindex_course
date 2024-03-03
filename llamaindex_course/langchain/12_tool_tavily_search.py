from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv

# Load the environment variables
load_dotenv()


search = TavilySearchResults()

result = search.invoke("what is the weather in SF")

print(result)

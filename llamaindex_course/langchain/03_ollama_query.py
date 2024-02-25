
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser





prompt = ChatPromptTemplate.from_template(
    "tell me a joke about {foo}, more than {word_count} words")
chain = prompt | model | StrOutputParser()

for chunk in chain.stream({"foo": "bears", "word_count": 200}):
    print(chunk, end="", flush=True)

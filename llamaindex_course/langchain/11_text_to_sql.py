import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
    max_tokens=8096,
)




## Open sql lite database
db = SQLDatabase.from_uri("sqlite:///./Chinook.db")

table_info = db.get_table_info()

print(table_info)


def get_schema(_):
    """Get the schema of the database"""
    return db.get_table_info()


def run_query(query):
    """Run a query on the database"""
    return db.run(query)


## Create a chain

TEMPLATE_WRITE_SQL_QUERY = """Based on the table schema below,

write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""

prmpt_write_sql_query = ChatPromptTemplate.from_template(TEMPLATE_WRITE_SQL_QUERY)


sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prmpt_write_sql_query
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


TEMPLATE_WRITE_SQL_QUERY = """Based on the table schema below, 
    question, sql query, and sql response, write a natural language response:
{schema}

Reason step by step. The database is case sensitive.


Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(TEMPLATE_WRITE_SQL_QUERY)


full_chain = (
    RunnablePassthrough.assign(query=sql_response).assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | model
    | StrOutputParser()
)

res = full_chain.invoke(
    {
        "question": "What is the name of the last employed person in the company? What is the birth date of the last employed person in the company?"
    }
)

print(res)

#res = full_chain.invoke({"question": "Display the names of employees that are older than Laura Callahan display their birth date and the age of each ?"})

#print(res)

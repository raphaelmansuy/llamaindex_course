import os

# Import BaseModel from Pydantic

from typing import List, Optional


from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser

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


class Person(BaseModel):
    """

    Model to store the information of a person.

    Attributes:

    last_name (str): The last name of the person.
    first_name (str): The first name of the person.
    middle_name (str): The middle name of the person.

    """

    lastname: Optional[str] = Field(description="The last name of the person.", required=False)
    firstname: Optional[str] = Field(description="The first name of the person.", required=False)
    middlename: Optional[str] = Field(
        description="The middle name of the person.", required=False
    )


class Organization(BaseModel):
    """

    Model to store the information of an organization.


    Attributes:

        name (str): The name of the organization.
    """

    name: Optional[str] = Field(description="The name of the organization.", required=True)


class Location(BaseModel):
    """Model to store the information of a location.

    Attributes:
        country (str): The country of the location.
        city (str): The city of the location.
        address (str): The address of the location.
    """

    country: Optional[str] = Field(
        description="The country of the location.", required=False
    )
    city: Optional[str] = Field(
        description="The city of the location.", required=False
    )
    address: Optional[str] = Field(
        description="The address of the location.", required=False
    )


class Entities(BaseModel):
    """

    Model to store the information of the entities.

    attributes:
        people (List[Person]): List of people.
        organizations (List[Organization]): List of organizations.
        locations (List[Location]): List of locations.

    """

    people: List[Person] = Field(description="List of people.", required=False)
    organizations: List[Organization] = Field(
        description="List of organizations.", required=False
    )
    locations: List[Location] = Field(description="List of locations.", required=False)


entities_parser = PydanticOutputParser(pydantic_object=Entities)

extract_prompt = PromptTemplate(
    template="Exctract Persons, Organizations, and Locations from Text. No markdown format !!\n{format_instructions}\nText:{text}\n",
    input_variables=["text"],
    partial_variables={
        "format_instructions": entities_parser.get_format_instructions()
    },
)


chain_extract = extract_prompt | model | entities_parser

instructions = entities_parser.get_format_instructions()

response = chain_extract.invoke(
    {
        "text": "John Doe, CEO of Acme Inc., is currently in New York.",
    }
)

# display the response

print(response)

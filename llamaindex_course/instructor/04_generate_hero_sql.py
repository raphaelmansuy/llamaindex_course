import instructor
from openai import OpenAI
from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session

from rich import print


# Define the model that will serve as a Table for the database
class Hero(SQLModel, instructor.OpenAISchema, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    secret_name: str
    age: Optional[int] = None

client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,  # JSON mode for LLM that don't support response_model with function signature
)




def create_hero() -> Hero:
    return client.chat.completions.create(
        model="mistral:latest",
        response_model=Hero,
        messages=[
            {"role": "user", "content": "Make a new superhero"},
        ],
        temperature=0.7
    )


# Insert the response into the database
engine = create_engine("sqlite:///database.db")
SQLModel.metadata.create_all(engine)

hero = create_hero()
print(hero.model_dump())


with Session(engine) as session:
    session.add(hero)
    session.commit()
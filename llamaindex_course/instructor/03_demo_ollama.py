from typing import List
from pydantic import BaseModel, Field

from openai import OpenAI

import instructor

from rich import print


class Character(BaseModel):
    """A character description"""

    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


# enables `response_model` in create call
client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,  # JSON mode for LLM that don't support response_model with function signature
)

resp = client.chat.completions.create(
    model="mistral:latest",
    messages=[
        {
            "role": "user",
            "content": "Tell me about Harry Potter, reply in JSON format only.",
        }
    ],
    response_model=Character,
    temperature=0.0,
    max_retries=2
)
print(resp.model_dump_json(indent=2))

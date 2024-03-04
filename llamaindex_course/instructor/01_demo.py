import os
import instructor
from openai import OpenAI
from pydantic import BaseModel

from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_API = os.environ.get("MODEL_API")
MODEL_KEY = os.environ.get("MODEL_KEY")

print(f"MODEL_NAME: {MODEL_NAME}")
print(f"MODEL_API: {MODEL_API}")

llm = OpenAI(api_key=MODEL_KEY, base_url=MODEL_API)

# This enables response_model keyword
# from client.chat.completions.create
client = instructor.patch(llm)


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model=MODEL_NAME,
    response_model=UserDetail,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        },
    ],
)

# assert isinstance(user, UserDetail)
# assert user.name == "Jason"
# assert user.age == 25
print(user.model_dump_json(indent=2))

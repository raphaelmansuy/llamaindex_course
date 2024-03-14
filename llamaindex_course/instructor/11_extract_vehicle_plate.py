from openai import OpenAI
import instructor

from pydantic import BaseModel

from rich import print


class Vehicle(BaseModel):
    """A vehicle description"""
    plate: str = None
    color: str = None
    constructor: str = None
    model: str = None


client = instructor.patch(OpenAI(), mode=instructor.Mode.MD_JSON)


def extract(url: str) -> Vehicle:
    result = client.chat.completions.create(
        model="gpt-4-vision-preview",
        max_tokens=4000,
        response_model=Vehicle,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe the vehicle in the image {url}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                ],
            }
        ],
    )
    return result


# Example using a vehicle image
output = extract(
    "https://images.unsplash.com/photo-1704312095035-e84e3edafeff?q=80&w=2787&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
)
print(output)



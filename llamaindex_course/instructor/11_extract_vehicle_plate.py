from openai import OpenAI
import instructor

from pydantic import BaseModel


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
    "https://cdn-dnlfbl.nitrocdn.com/DhDXnMUNlvRaaAgawLsmRnDODzlslqaM/assets/images/optimized/rev-cceb3b6/numberplateclinic.co.uk/wp-content/uploads/Number-Plates-Illegal-1200x480.webp"
)
print(output)


another = extract(
    url="https://www.duxfordautomotive.co.uk/assets/1026714/large/65198c25363a758c400f93010b1ae943_1026714.jpg"
)

print(another)
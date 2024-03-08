import enum
import instructor
from openai import OpenAI

from pydantic import BaseModel

client = instructor.patch(OpenAI())


class Labels(str, enum.Enum):
    SPAM = "spam"
    NOT_SPAM = "not_spam"


class SinglePrediction(BaseModel):
    """
    Correct class label for the given text
    """

    class_label: Labels


def classify(data: str) -> SinglePrediction:
    return client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        response_model=SinglePrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {data}",
            },
        ],
    )  # type: ignore


prediction = classify("Hello there I'm a nigerian prince and I want to give you money")

print(prediction.model_dump_json(indent=2))

prediction = classify(
    "Bonjour Laurent, I hope you are doing well. I wanted to ask you if you could help me with something."
)

print(prediction.model_dump_json(indent=2))

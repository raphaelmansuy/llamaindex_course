import enum
import instructor

from typing import List
from openai import OpenAI
from pydantic import BaseModel

client = instructor.patch(OpenAI())


# Define new Enum class for multiple labels
class MultiLabels(str, enum.Enum):
    BILLING = "billing"
    GENERAL_QUERY = "general_query"
    HARDWARE = "hardware"


# Adjust the prediction model to accommodate a list of labels
class MultiClassPrediction(BaseModel):
    predicted_labels: List[MultiLabels]


# Modify the classify function
def multi_classify(data: str) -> MultiClassPrediction:
    return client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {data}",
            },
        ],
    )  # type: ignore


# Example using a support ticket
ticket = (
    "My account is locked and I can't access my billing info. Phone is also broken."
)
prediction = multi_classify(ticket)
print(prediction)

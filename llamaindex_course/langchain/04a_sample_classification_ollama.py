"" "This script demonstrates how to create a chain to evaluate the urgency of a patient triage and evaluation at an emergency service in a hospital." ""
import os
import json
# Import BaseModel from Pydantic

from typing import List


from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator







class UrgencyEvaluation(BaseModel):
    """
    Model to store the result of a patient triage and evaluation at an emergency service in a hospital.

    Attributes:
        patient_name (str): The unique identifier of the patient.
        triage_level (str): The urgency level assigned during triage (e.g., critical, urgent, non-urgent).
        evaluation_notes (str): Additional notes or observations from the evaluation.
        vital_signs (dict): Vital signs data stored in a dictionary format.
            Example: {'heart_rate': 80, 'blood_pressure': '120/80'}.
        symptoms (List[str]): List of symptoms reported by the patient.
        explanation (str): The explanation of the urgency level assigned during triage (e.g., critical, urgent, non-urgent).
    """
    patient_name: str = Field(description="The unique name of the patient.")
    triage_level: str = Field(
        description="The urgency level assigned during triage (e.g., critical, urgent, non-urgent).")
    evaluation_notes: str = Field(
        description="Additional notes or observations from the evaluation.")
    vital_signs: dict = Field(
        description="Vital signs data stored in a dictionary format. Example: {'heart_rate': 80, 'blood_pressure': '120/80'}.")
    symptoms: List[str] = Field(
        description="List of symptoms reported by the patient.")
    explanation: str = Field(
        "The explanation of the urgency level assigned during triage (e.g., critical, urgent, non-urgent).")


CLINICAL_TRIAGE_PROMPT = """
As a ugentist, evaluate the urgency of the patient {patient_name} with the following symptoms: {symptoms} and vital signs: {vital_signs}. 
The results must formated as follow: {instructions}.
"""


# Create a prompt
clinical_triage_prompt = ChatPromptTemplate.from_template(
    CLINICAL_TRIAGE_PROMPT)
# Create a model
MODEL_NAME = "mistral:latest"

model = Ollama(model=MODEL_NAME)
# Create an output parser
urgency_evaluation_parser = PydanticOutputParser(
    pydantic_object=UrgencyEvaluation)

# Create a chain
evaluation_chain = clinical_triage_prompt | model | urgency_evaluation_parser

PATIENT_NAME = "John Doe"
VITAL_SIGNS = "heart_rate: 80, blood_pressure: 180/100"
SYMPTOMS = "headache, fever, cough, difficulty breathing"


evaluation = evaluation_chain.invoke({"patient_name": PATIENT_NAME,
                                      "vital_signs": VITAL_SIGNS,
                                      "symptoms": SYMPTOMS,
                                      "instructions": urgency_evaluation_parser.get_format_instructions()})

# print the JSON beautifully

print(json.dumps(evaluation.dict(), indent=4))

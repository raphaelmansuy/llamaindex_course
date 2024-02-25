import os
import json
# Import BaseModel from Pydantic

from typing import List


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# The Web interface components
import gradio as gr





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

MODEL_NAME = "mistral:latest"

model = Ollama(model=MODEL_NAME)


urgency_evaluation_parser = PydanticOutputParser(
    pydantic_object=UrgencyEvaluation)

# Create a chain
evaluation_chain = clinical_triage_prompt | model | urgency_evaluation_parser


def evaluate_patient(patient_name: str, vital_signs: str, symptoms: str) -> str:
    """
    Evaluate the urgency of a patient based on symptoms and vital signs.

    Args:
        patient_name (str): The unique identifier of the patient.
        vital_signs (str): The vital signs of the patient.
        symptoms (str): The symptoms reported by the patient.


    Returns:
        dict: A dictionary containing the urgency evaluation results.
    """
    # Create a chain
    evaluation_chain = clinical_triage_prompt | model | urgency_evaluation_parser
    # format instructions
    format_instructions = urgency_evaluation_parser.get_format_instructions()
    # Invoke the chain
    evaluation = evaluation_chain.invoke({"patient_name": patient_name,
                                          "vital_signs": vital_signs,
                                          "symptoms": symptoms,
                                          "instructions": format_instructions})

    result = json.dumps(evaluation.dict(), indent=4)

    # Return the evaluation results
    return result


# Create a Gradio interface
iface = gr.Interface(fn=evaluate_patient,
                     inputs=["text", "text", "text"],
                     outputs="text",
                     title="Patient Evaluation",
                     description="Evaluate patient urgency based on symptoms and vital signs.")


iface.launch()

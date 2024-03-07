import os
import json
from typing import List
from rich.console import Console

# rich is a Python library for rich text and beautiful formatting in the terminal.
# it can be installed with pip install rich
# or poetry add rich


# Example of a FHIR file
FILE_TO_OPEN = "data_json/0d5f5a77-b49d-4f8b-887f-70e9de390751.json"

# Get the full path of the file
FULL_PATH = os.path.join(os.getcwd(), FILE_TO_OPEN)


def read_fhir_file(file_path: str) -> dict:
    """Read a JSON file and return the data as a dictionary
    Args:
        file_path: The path to the file
    Returns:
        The data as a dictionary
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Check if the data is a FHIR Bundle
    if "resourceType" in data and data["resourceType"] == "Bundle":
        return data["entry"]
    else:
        # Error exception
        raise ValueError("The document does not appear to be a FHIR Bundle")


def get_fhir_resource_types(data: dict, resource_type: str) -> List[dict]:
    """Get the FHIR resource type from the data

    Args:
        data: The data
        resource_type: The resource type it can be
        "Patient", "Observation", "Condition", "Procedure",
        "MedicationRequest", "CarePlan", "CareTeam", "Encounter",
        "DiagnosticReport", "Immunization"

    Returns:
        The resource type as a list of dictionaries or an empty list
    """
    entry_list = []
    for entry in data:
        if entry["resource"]["resourceType"] == resource_type:
            entry_list.append(entry["resource"])
    return entry_list


def get_list_of_resources(data: dict) -> List[str]:
    """Get a list of resources from the data"""
    resource_list = []
    for entry in data:
        # Add the resource type to the list if it is not already there
        if entry["resource"]["resourceType"] not in resource_list:
            resource_list.append(entry["resource"]["resourceType"])
    return resource_list


# Example of resources types in HRIR format
# "    'Patient',
#    'Organization',
#    'Practitioner',
#    'Encounter',
#    'Condition',
#    'Observation',
#    'Procedure',
#    'Claim',
#    'ExplanationOfBenefit',
#    'Immunization',
#    'DiagnosticReport',
#    'MedicationRequest',
#    'CareTeam',
#   'CarePlan'


def main():

    # Initialize the Rich Console for nicer printing in the console
    console = Console()

    # Read the FHIR file
    # A FHIR file is a JSON file that contains a FHIR Bundle
    # FHIR stands for Fast Healthcare Interoperability Resources
    # It is a standard for exchanging healthcare information electronically
    # The FHIR Bundle is a collection of resources
    # Each resource is a record in the FHIR format
    # The FHIR Bundle contains a list of entries
    # Example of entry is a Patient, an Observation, a Condition, etc.
    data = read_fhir_file(FULL_PATH)

    # Get list of resources in the data
    # A resource is a data type that is used to represent a record in FHIR
    resource_list = get_list_of_resources(data)
    console.print("The list of resources presents in the data is:")
    console.print(resource_list)

    # Get the patient
    # The patient is a resource that represents a person receiving care
    # It contains information about the patient
    # Example of information is the name
    patient = get_fhir_resource_types(data, "Patient")
    console.print("Patient is:")
    console.print(patient)

    # Get the observations
    # An observation is a resource that represents a measurement or an observation
    # It contains information about the observation
    # Example of information is the value of the observation
    # Example of observation is the blood pressure
    # Example of value is the systolic and diastolic pressure
    # Example of unit is mmHg
    # Example of observation is the temperature
    observations = get_fhir_resource_types(data, "Observation")
    console.print("The Oservation are:",style="bold")
    console.print(observations)

    # Get the conditions
    # A condition is a resource that represents a diagnosis
    # It contains information about the diagnosis
    # Example of information is the code of the diagnosis
    # Example of code is the ICD-10 code 
    # Example of information is the clinical status of the diagnosis
    # Example of clinical status is the active or resolved status
    # Example of information is the verification status of the diagnosis
    # Example of verification status is the confirmed or unconfirmed status
    # Example of information is the onset date of the diagnosis
    # Example of onset date is the date when the diagnosis was made
    # Example of information is the abatement date of the diagnosis
    conditions = get_fhir_resource_types(data, "Condition")
    console.print("Conditions are:")
    console.print(conditions)

    # Get the procedures
    # A procedure is a resource that represents a procedure
    # It contains information about the procedure
    # Example of information is the code of the procedure
    # Example of code is the ICD-10 code, ICD-10 code (eg. 01.23, 01.24)
    # Example of information is the status of the procedure
    # Example of status is the completed or in-progress status
    procedures = get_fhir_resource_types(data, "Procedure")
    console.print("Procedures are:")
    console.print(procedures)

    # Get the medications
    # A medication is a resource that represents a medication
    # It contains information about the medication
    # Example of information is the code of the medication
    # Example of code is the RxNorm code
    # Example of information is the status of the medication
    # Example of status is the active or completed status
    # Example of information is the dosage of the medication
    # Example of dosage is the dose quantity, dose unit, dose timing
    # Example of information is the duration of the medication
    # Example of duration is the duration quantity, duration unit
    medications = get_fhir_resource_types(data, "MedicationRequest")
    console.print("Medications are:")
    console.print(medications)

    # Get the care plans
    # A care plan is a resource that represents a care plan
    # It contains information about the care plan
    # Example of information is the status of the care plan
    # Example of status is the active or completed status
    # Example of information is the intent of the care plan
    # Example of intent is the proposal or plan status
    # Example of information is the category of the care plan
    # Example of category is the treatment or dietary status
    # Example of information is the goal of the care plan
    # Example of goal is the goal description, goal status
    # Example of information is the activity of the care plan
    # Example of activity is the activity description, activity status
    care_plans = get_fhir_resource_types(data, "CarePlan")
    console.print("Care plans are:")
    console.print(care_plans)

    # Get the care teams
    # A care team is a resource that represents a care team
    # It contains information about the care team
    # Example of information is the status of the care team
    # Example of status is the active or suspended status
    # Example of information is the category of the care team
    # Example of category is the treatment or dietary status
    # Example of information is the participant of the care team
    # Example of participant is the role, member, onBehalfOf
    # Example of information is the managing organization of the care team
    # Example of managing organization is the organization name, organization type
    # Example of information is the reason of the care team
    # Example of reason is the reason description, reason code
    care_teams = get_fhir_resource_types(data, "CareTeam")
    console.print("Care teams are:")
    console.print(care_teams)

    # Get the encounters
    # An encounter is a resource that represents an encounter
    # It contains information about the encounter
    # An encounter is a contact between a patient and a healthcare provider
    # Example of information is the status of the encounter
    # Example of status is the finished or in-progress status
    # Example of information is the type of the encounter
    # Example of type is the ambulatory or emergency type
    # Example of information is the class of the encounter
    # Example of class is the inpatient or outpatient class
    # Example of information is the period of the encounter
    # Example of period is the start date, end date
    # Example of information is the reason of the encounter
    # Example of reason is the reason description, reason code
    encounters = get_fhir_resource_types(data, "Encounter")
    console.print("Encounters are:", style="bold")
    console.print(encounters)

    # Get the diagnostic reports
    # A diagnostic report is a resource that represents a diagnostic report
    # It contains information about the diagnostic report
    # Example of information is the status of the diagnostic report
    # Example of status is the final or preliminary status
    # Example of information is the category of the diagnostic report
    # Example of category is the imaging or laboratory category
    # Example of information is the code of the diagnostic report
    # Example of code is the LOINC code
    # Example of information is the result of the diagnostic report
    # Example of result is the result value, result unit
    # Example of information is the conclusion of the diagnostic report
    diagnostic_reports = get_fhir_resource_types(data, "DiagnosticReport")
    console.print("Diagnostic reports are:")
    console.print(diagnostic_reports)

    # Get the immunizations
    # An immunization is a resource that represents an immunization
    # It contains information about the immunization
    # Example of information is the status of the immunization
    # Example of status is the completed or in-progress status
    # Example of information is the vaccine code of the immunization
    # Example of vaccine code is the CVX code
    # Example of information is the date of the immunization
    # Example of date is the date when the immunization was given
    # Example of information is the patient of the immunization
    # Example of patient is the patient name
    # Example of information is the encounter of the immunization
    immunizations = get_fhir_resource_types(data, "Immunization")
    console.print("Immunizations are:")
    console.print(immunizations)


if __name__ == "__main__":
    main()

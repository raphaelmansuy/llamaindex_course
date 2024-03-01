from pydantic import BaseModel
from typing import List, Dict, Any, Union


class Coding(BaseModel):
    system: str
    code: str
    display: str


class Extension(BaseModel):
    url: str
    #    extension: List['Extension'] = []
    valueCoding: Coding = None
    valueString: str = None
    valueCode: str = None
    valueAddress: Dict[str, str] = None
    valueDecimal: float = None


class Identifier(BaseModel):
    system: str
    value: str
    type: Dict[str, Any] = None


class Telecom(BaseModel):
    system: str = None
    value: str = None
    use: str = None


class Name(BaseModel):
    use: str = None
    prefix: Union[str, List[str]] = None
    family: str = None
    given: Union[List[str], str] = None


class Address(BaseModel):
    extension: List[Extension] = []
    line: List[str] = None
    city: str = None
    state: str = None
    postalCode: str = None
    country: str = None


class MaritalStatus(BaseModel):
    coding: List[Coding]
    text: str = None


class Contact(BaseModel):
    name: Name
    telecom: List[Telecom]


class Communication(BaseModel):
    language: Dict[str, Any]


class Resource(BaseModel):
    resourceType: str
    id: str
    text: Dict[str, str] = None
    extension: List[Extension] = []
    identifier: List[Identifier] = []
    name: Union[str, List[Name]] = None
    telecom: List[Telecom] = []
    gender: str = None
    birthDate: str = None
    address: List[Address] = []
    maritalStatus: MaritalStatus = None
    multipleBirthBoolean: bool = None
    contact: List[Contact] = []
    communication: List[Communication] = []
    # Add more fields as needed


class Entry(BaseModel):
    fullUrl: str
    resource: Resource
    request: Dict[str, str] = None


class Bundle(BaseModel):
    resourceType: str
    type: str
    entry: List[Entry]
    # Add more fields as needed


# Usage
# data = json.loads(hl7_fhir_json_string)
# bundle = Bundle.parse_obj(data)


def json_parse_medical_records(json_string: str) -> Bundle:
    """Read a medical file in JSON format and return a Bundle object"""
    bundle = Bundle.model_validate_json(json_string, strict=False)
    return bundle


def read_medical_file(file_path: str) -> Bundle:
    """Read a medical file in JSON format and return a Bundle object"""
    with open(file_path, encoding="utf-8") as f:
        json_string = f.read()
        return json_parse_medical_records(json_string)

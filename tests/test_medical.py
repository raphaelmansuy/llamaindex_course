import logging
import sys
import json

from pathlib import Path

from llamaindex_course.demo_medical.medical_record_format import read_medical_file,Bundle

## get a loggger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "./data_json/0d5f5a77-b49d-4f8b-887f-70e9de390751.json"


def test_json_parse_medical_records1():
    """ Test the json parse medical records function"""
    pass


def test_read_medical_record_format():
    """Test the read_medical_record_format function"""
    logger.info("Loading the documents from the directory %s", DOCUMENTS_DIR)
    bundle: Bundle = read_medical_file(Path(DOCUMENTS_DIR))
    
    logging.debug("Parsing the documents to get the nodes")
    
    ## Print the first entry nicely
    print(json.dumps(bundle.model_dump(), indent=4))
    
    ## Display nicely bundler
    
    for entry in bundle.entry:
        print(entry.resource.resourceType)
        print(entry.resource.id)
        print(entry.resource.name)
        if entry.resource.resourceType == "Observation":
            print(entry.resource)
    
    
    # Display nicely bundler
   # logger.debug(bundle.entry[0])
